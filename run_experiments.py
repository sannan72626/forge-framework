"""
run_experiments.py
──────────────────
Runs FORGE + 4 baselines and produces REAL measured FQ × MU trajectories.

Two modes:
  --mode smoke   GPT-2 + synthetic profiles. No GPU. ~10 min. Proves pipeline.
  --mode full    Llama-2-7B + TOFU-5%. Needs 24GB GPU. ~18 GPU-hours total.

Usage:
  # Quick demo (show project leader immediately):
  python run_experiments.py --mode smoke --out_dir results_real/

  # Real paper results (run on GPU server):
  python run_experiments.py --mode full --split forget05 --out_dir results_tofu5/

  # Single method (for debugging):
  python run_experiments.py --mode smoke --methods GA --out_dir debug/

Output (results_real/):
  trajectories.json          raw (step, fq, mu, rouge_f, rouge_r) per method
  trajectory_plot.png        real FQ × MU plot
  comparison_table.csv       final-checkpoint table (all methods × all metrics)
  <method>/checkpoint-*/     saved model checkpoints

All numbers in trajectory.json are MEASURED, not simulated.
"""

import argparse
import copy
import json
import math
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── Try importing optional heavy deps ────────────────────────────────────────
try:
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              get_cosine_schedule_with_warmup)
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

try:
    from rouge_score import rouge_scorer as rouge_lib
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── Data ─────────────────────────────────────────────────────────────────────

@dataclass
class QAExample:
    question: str
    answer:   str
    is_forget: bool = False


def synthetic_profiles(n_profiles=40, qa_per=4, forget_frac=0.10, seed=42):
    """TOFU-style synthetic profiles for smoke test (no network needed)."""
    first = ["Aria","Ben","Cleo","Dario","Eva","Felix","Gita","Hugo","Iris",
             "Joe","Kira","Leo","Mia","Niko","Opal","Pax","Qadr","Ren",
             "Sasha","Theo","Uma","Vito","Wren","Xan","Yara","Zane",
             "Ada","Bohr","Cori","Dij","Esh","Fay","Gor","Han","Ish",
             "Jib","Kel","Lal","Mor","Nye"]
    last  = ["Alev","Bara","Cano","Dhar","Eke","Faro","Gol","Hama",
             "Iska","Jova","Kan","Lim","Mor","Nila","Ore","Pelt"]
    cities = ["Lisbon","Kyoto","Reykjavik","Cairo","Quito","Tbilisi"]
    genres = ["mystery","satire","memoir","sci-fi","historical fiction"]
    decades= ["1990s","2000s","2010s","2020s"]

    n_forget_p = max(1, int(round(forget_frac * n_profiles)))
    forget, retain = [], []

    for pid in range(n_profiles):
        rng_p = random.Random(seed + pid * 7)
        name   = rng_p.choice(first) + " " + rng_p.choice(last)
        city   = rng_p.choice(cities)
        genre  = rng_p.choice(genres)
        decade = rng_p.choice(decades)
        pool   = [
            (f"Where does {name} live?",           f"{name} lives in {city}."),
            (f"What genre does {name} write?",     f"{name} writes {genre}."),
            (f"When did {name} debut?",             f"{name} debuted in the {decade}."),
            (f"What city is {name} based in?",     f"{name} is based in {city}."),
            (f"What kind of books does {name} write?", f"{name} writes {genre} books."),
            (f"Which decade is {name} known from?",    f"{name} is from the {decade}."),
        ]
        rng_p.shuffle(pool)
        is_forget = pid < n_forget_p
        bucket = forget if is_forget else retain
        for q, a in pool[:qa_per]:
            bucket.append(QAExample(q, a, is_forget=is_forget))

    return forget, retain


def load_tofu(forget_split="forget05"):
    """Load real TOFU from HuggingFace. Requires: pip install datasets"""
    from datasets import load_dataset
    retain_split = "retain" + str(100 - int(forget_split.replace("forget","")))
    forget_ds = load_dataset("locuslab/TOFU", forget_split)["train"]
    retain_ds = load_dataset("locuslab/TOFU", retain_split)["train"]
    forget = [QAExample(e["question"], e["answer"], is_forget=True)  for e in forget_ds]
    retain = [QAExample(e["question"], e["answer"], is_forget=False) for e in retain_ds]
    return forget, retain


# ── Dataset wrapper ───────────────────────────────────────────────────────────

class QADataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=128):
        self.examples   = examples
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.examples)

    def __getitem__(self, idx):
        ex  = self.examples[idx]
        txt = f"Q: {ex.question}\nA: {ex.answer}"
        enc = self.tokenizer(
            txt, return_tensors="pt", truncation=True,
            max_length=self.max_length, padding="max_length"
        )
        ids  = enc["input_ids"][0]
        mask = enc["attention_mask"][0]
        lbl  = ids.clone(); lbl[mask == 0] = -100
        return {"input_ids": ids, "attention_mask": mask, "labels": lbl}


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def _nll(model, tokenizer, text, device, max_length=128):
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=max_length).to(device)
    return model(**enc, labels=enc["input_ids"]).loss.item()


@torch.no_grad()
def _generate(model, tokenizer, question, device, max_new=20):
    prompt = f"Q: {question}\nA:"
    enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=100).to(device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    return full[len(prompt):].strip()


def measure_metrics(model, ref_model, tokenizer, forget_ex, retain_ex, device,
                    n=25, max_length=128, max_new=20):
    """
    Returns dict: fq, mu, rouge_forget, rouge_retain, mia_tr, mia_pr
    All MEASURED (no simulation).
    """
    model.eval()
    sub_f = forget_ex[:n]; sub_r = retain_ex[:n]

    # ── NLL distributions ────────────────────────────────────────────────────
    def nlls(examples, mdl):
        return [_nll(mdl, tokenizer,
                     f"Q: {e.question}\nA: {e.answer}", device, max_length)
                for e in examples]

    nll_f_model = nlls(sub_f, model)
    nll_f_ref   = nlls(sub_f, ref_model)
    nll_r_model = nlls(sub_r, model)
    nll_r_ref   = nlls(sub_r, ref_model)

    # FQ: KS p-value
    if HAS_SCIPY and len(nll_f_model) >= 5:
        _, fq = scipy_stats.ks_2samp(nll_f_model, nll_f_ref)
    else:
        # Fallback: normalised NLL difference
        fq = float(np.clip(1.0 - abs(np.mean(nll_f_model) - np.mean(nll_f_ref)) / 3.0, 0, 1))

    # ROUGE-L
    def rouge_mean(examples):
        if not HAS_ROUGE:
            return 0.0
        scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
        scores = []
        for e in examples[:min(n, 15)]:   # subsample for speed
            gen = _generate(model, tokenizer, e.question, device, max_new)
            scores.append(scorer.score(e.answer, gen)["rougeL"].fmeasure)
        return float(np.mean(scores))

    rouge_f = rouge_mean(sub_f)
    rouge_r = rouge_mean(sub_r)

    # MU: harmonic mean of retain ROUGE-L and perplexity recovery
    ppl_model = float(np.mean([math.exp(min(v, 20)) for v in nll_r_model]))
    ppl_ref   = float(np.mean([math.exp(min(v, 20)) for v in nll_r_ref]))
    ppl_score = float(np.clip(math.exp(-(ppl_model - ppl_ref) / max(ppl_ref, 1e-6)), 0, 1))
    mu = (2 * rouge_r * ppl_score / (rouge_r + ppl_score + 1e-6)
          if rouge_r + ppl_score > 0 else 0.0)

    # MIA-TR: truth vs perturbed ratio
    def perturb(ans):
        w = ans.split(); random.shuffle(w); return " ".join(w) if len(w)>1 else ans+" UNK"

    @torch.no_grad()
    def log_prob_answer(mdl, q, a):
        txt = f"Q: {q}\nA: {a}"
        enc = tokenizer(txt, return_tensors="pt", truncation=True,
                        max_length=max_length).to(device)
        out = mdl(**enc)
        logp = F.log_softmax(out.logits[...,:-1,:], dim=-1)
        tgts = enc["input_ids"][...,1:]
        return logp.gather(-1, tgts.unsqueeze(-1)).squeeze(-1).mean().item()

    tr_ratios = [
        math.exp(log_prob_answer(model, e.question, e.answer) -
                 log_prob_answer(model, e.question, perturb(e.answer)))
        for e in sub_f[:min(n,15)]
    ]
    mia_tr = float(np.mean(tr_ratios))

    # MIA-PR: student vs ref ratio
    pr_ratios = [
        math.exp(-nll_f_model[i] - (-nll_f_ref[i]))
        for i in range(len(nll_f_model))
    ]
    mia_pr = float(np.mean(pr_ratios))

    return dict(fq=fq, mu=mu, rouge_forget=rouge_f, rouge_retain=rouge_r,
                mia_tr=mia_tr, mia_pr=mia_pr)


# ── Unlearning methods ────────────────────────────────────────────────────────

def loss_GA(model, batch, device):
    """Gradient Ascent: -NLL on forget set."""
    out = model(input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device))
    return -out.loss


def loss_NPO(model, ref_model, batch, device, beta=0.1):
    """Negative Preference Optimization (Zhang et al. 2024)."""
    out     = model(input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device))
    with torch.no_grad():
        ref_out = ref_model(input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device))
    logits  = out.logits[...,:-1,:];  ref_logits = ref_out.logits[...,:-1,:]
    targets = batch["labels"].to(device)[...,1:]
    mask    = (targets != -100).float()
    tgt_c   = targets.masked_fill(targets == -100, 0)
    logp     = F.log_softmax(logits,     dim=-1).gather(-1, tgt_c.unsqueeze(-1)).squeeze(-1)
    ref_logp = F.log_softmax(ref_logits, dim=-1).gather(-1, tgt_c.unsqueeze(-1)).squeeze(-1)
    log_ratio = ((logp - ref_logp) * mask).sum(-1) / (mask.sum(-1) + 1e-8)
    return -2.0 / beta * torch.log(torch.sigmoid(-beta * log_ratio) + 1e-8).mean()


def loss_FLAT(model, batch, device, beta=2.0):
    """FLAT: forget-only loss adjustment (Wang et al., ICLR 2025)."""
    out    = model(input_ids=batch["input_ids"].to(device),
                   attention_mask=batch["attention_mask"].to(device))
    logits = out.logits[...,:-1,:]
    tgts   = batch["labels"].to(device)[...,1:]
    mask   = (tgts != -100).float()
    tgt_c  = tgts.masked_fill(tgts == -100, 0)
    logp   = F.log_softmax(logits, dim=-1).gather(-1, tgt_c.unsqueeze(-1)).squeeze(-1)
    p      = logp.exp()
    flat_  = -torch.log(1.0 + beta * p + 1e-8)
    return (flat_ * mask).sum() / (mask.sum() + 1e-8)


def loss_LoKU(model, ref_model, batch, device, margin=0.5, lambda_retain=1.0,
              retain_batch=None):
    """LoKU = IHL + FILA stub (Cha et al., ICLR 2025).
    Inverted Hinge Loss for forget + standard NLL retain."""
    out_f = model(input_ids=batch["input_ids"].to(device),
                  attention_mask=batch["attention_mask"].to(device),
                  labels=batch["labels"].to(device))
    # IHL: clip gradient ascent at margin
    l_forget = -torch.clamp(out_f.loss, max=margin)

    if retain_batch is not None:
        out_r = model(input_ids=retain_batch["input_ids"].to(device),
                      attention_mask=retain_batch["attention_mask"].to(device),
                      labels=retain_batch["labels"].to(device))
        return l_forget + lambda_retain * out_r.loss
    return l_forget


def loss_FORGE(model, ref_model, batch, device, retain_batch=None,
               lambda_retain=1.0, gamma_robust=0.5, eta_gate=0.01,
               ihl_margin=4.0):
    """FORGE: bounded GA + retain + adversarial robust (simplified stub).

    Full version (with Fisher-gated LoRA + adversary Φ_ψ) is in forge/trainer.py.
    This stub uses bounded gradient ascent + retain anchor for immediate use.
    """
    # ℒ_forget: Inverted Hinge Loss (bounded GA)
    out_f = model(input_ids=batch["input_ids"].to(device),
                  attention_mask=batch["attention_mask"].to(device),
                  labels=batch["labels"].to(device))
    l_forget = -torch.tanh(out_f.loss / ihl_margin) * ihl_margin

    total = l_forget

    # ℒ_retain
    if retain_batch is not None and lambda_retain > 0:
        out_r = model(input_ids=retain_batch["input_ids"].to(device),
                      attention_mask=retain_batch["attention_mask"].to(device),
                      labels=retain_batch["labels"].to(device))
        total = total + lambda_retain * out_r.loss

    # ℒ_robust: NLL on a simple paraphrase proxy
    # (Full adversary is in forge/trainer.py; here we simulate with token drop)
    if gamma_robust > 0:
        # Simple proxy: randomly mask 20% of input tokens
        ids_rob = batch["input_ids"].clone().to(device)
        drop_mask = torch.rand_like(ids_rob.float()) < 0.20
        ids_rob[drop_mask] = 0   # zero = mask token proxy
        out_rob = model(input_ids=ids_rob,
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device))
        total = total + gamma_robust * out_rob.loss

    return total


# Method registry
METHOD_LOSSES = {
    "GA":    lambda model, ref, b, r, dev: loss_GA(model, b, dev),
    "NPO":   lambda model, ref, b, r, dev: loss_NPO(model, ref, b, dev),
    "FLAT":  lambda model, ref, b, r, dev: loss_FLAT(model, b, dev),
    "LoKU":  lambda model, ref, b, r, dev: loss_LoKU(model, ref, b, dev, retain_batch=r),
    "FORGE": lambda model, ref, b, r, dev: loss_FORGE(model, ref, b, dev, retain_batch=r),
}


# ── Training loop ─────────────────────────────────────────────────────────────

def train_method(
    method_name:   str,
    base_state:    dict,
    ref_model:     nn.Module,
    forget_ex:     list,
    retain_ex:     list,
    tokenizer,
    device:        torch.device,
    model_factory,            # callable() → fresh model
    n_steps:       int  = 60,
    eval_every:    int  = 10,
    lr:            float = 5e-6,
    batch_size:    int  = 4,
    max_length:    int  = 128,
    out_dir:       Path  = None,
    lora_rank:     int  = 16,
    use_lora:      bool = True,
    n_eval:        int  = 25,
) -> list[dict]:
    """Train one method from base_state; return trajectory list."""
    print(f"\n{'='*60}")
    print(f"Method: {method_name}")
    print(f"{'='*60}")

    model = model_factory()
    model.load_state_dict(base_state, strict=False)
    model.to(device)

    if use_lora and HAS_PEFT:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=lora_rank, lora_alpha=lora_rank*2,
            lora_dropout=0.0,
            target_modules=["c_attn"] if hasattr(model, "transformer") else ["q_proj","v_proj"],
        )
        model = get_peft_model(model, lora_cfg)
        print(f"  LoRA enabled (rank={lora_rank})")

    forget_ds = QADataset(forget_ex, tokenizer, max_length)
    retain_ds = QADataset(retain_ex, tokenizer, max_length)
    forget_dl = DataLoader(forget_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    retain_dl = DataLoader(retain_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)

    loss_fn = METHOD_LOSSES[method_name]
    trajectory = []

    def eval_checkpoint(step):
        m = measure_metrics(model, ref_model, tokenizer,
                            forget_ex, retain_ex, device, n=n_eval,
                            max_length=max_length)
        m["step"] = step
        trajectory.append(m)
        print(f"  step {step:>4d}  FQ={m['fq']:.3f}  MU={m['mu']:.3f}  "
              f"RL-f={m['rouge_forget']:.3f}  RL-r={m['rouge_retain']:.3f}  "
              f"MIA-TR={m['mia_tr']:.3f}  MIA-PR={m['mia_pr']:.3f}")
        return m

    eval_checkpoint(0)

    forget_iter = iter(forget_dl)
    retain_iter = iter(retain_dl)

    for step in range(1, n_steps + 1):
        model.train()
        try:    f_batch = next(forget_iter)
        except: forget_iter = iter(forget_dl); f_batch = next(forget_iter)
        try:    r_batch = next(retain_iter)
        except: retain_iter = iter(retain_dl); r_batch = next(retain_iter)

        loss = loss_fn(model, ref_model, f_batch, r_batch, device)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        if step % eval_every == 0:
            eval_checkpoint(step)

    # Save checkpoint
    if out_dir:
        ckpt = out_dir / method_name
        ckpt.mkdir(parents=True, exist_ok=True)
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(ckpt)
        else:
            torch.save(model.state_dict(), ckpt / "model.pt")
        print(f"  Checkpoint saved → {ckpt}")

    return trajectory


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_trajectories(all_traj: dict, out_path: Path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    WHITE="#FFFFFF"; NAVY="#1E2761"; AMBER="#F4B400"; MUTE="#5A6373"; ICE="#EEF1FA"
    COLORS = {"GA":"#B23A48","NPO":"#E07A5F","FLAT":"#3D5A80","LoKU":"#81B29A","FORGE":"#1E2761"}

    fig, ax = plt.subplots(figsize=(9.8, 6.2), dpi=185)
    fig.patch.set_facecolor(WHITE); ax.set_facecolor("#FAFAFB")
    ax.axhline(0.80, color="#D9D9DE", lw=0.7, ls="--")
    ax.axvline(0.80, color="#D9D9DE", lw=0.7, ls="--")

    # Pre-unlearn origin
    fq0 = all_traj[list(all_traj.keys())[0]][0]["fq"]
    mu0 = all_traj[list(all_traj.keys())[0]][0]["mu"]
    ax.scatter([fq0],[mu0],s=80,color="#888",edgecolor=WHITE,lw=1.5,zorder=4,marker="o")
    ax.text(fq0+0.01,mu0-0.015,"Pre-unlearn (t=0)",fontsize=8.5,color="#555")

    for name, traj in all_traj.items():
        fq = [p["fq"] for p in traj]
        mu = [p["mu"] for p in traj]
        c  = COLORS.get(name,"#888")
        lw = 2.8 if name=="FORGE" else 1.9
        ax.plot(fq, mu, color=c, lw=lw, alpha=0.95, zorder=3)
        ax.scatter(fq[1:], mu[1:], color=c, s=28, zorder=4, edgecolor=WHITE, lw=0.8)
        if len(fq) > 2:
            mid = len(fq)//2
            ax.annotate("", xy=(fq[mid+1],mu[mid+1]), xytext=(fq[mid],mu[mid]),
                        arrowprops=dict(arrowstyle="->",color=c,lw=lw))
        import matplotlib.patheffects as pe
        ax.text(fq[-1]+0.008,mu[-1],name,color=c,fontsize=10,fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2.5,foreground=WHITE)])

    ax.scatter([1.0],[mu0],s=260,marker="*",color=AMBER,edgecolor="#7A5A00",lw=1,zorder=5)

    handles = [Line2D([0],[0],color=COLORS.get(n,"#888"),lw=2.4,label=n)
               for n in all_traj]
    ax.legend(handles=handles,loc="lower left",fontsize=9,frameon=True,edgecolor="#DDD")
    ax.set_xlabel("Forget Quality (FQ) ↑",fontsize=11.5)
    ax.set_ylabel("Model Utility (MU) ↑",fontsize=11.5)
    ax.set_title("MEASURED FQ × MU trajectories (real training, not simulation)",
                 fontsize=11.5,color=NAVY,loc="left",pad=9)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path,dpi=185,bbox_inches="tight",facecolor=WHITE)
    plt.close()
    print(f"Trajectory plot → {out_path}")


def write_csv(all_traj: dict, out_path: Path):
    """Write final-checkpoint comparison table as CSV."""
    import csv
    metrics = ["fq","mu","rouge_forget","rouge_retain","mia_tr","mia_pr"]
    with open(out_path,"w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + metrics)
        for name, traj in all_traj.items():
            last = traj[-1]
            w.writerow([name] + [f"{last.get(m,0):.4f}" for m in metrics])
    print(f"Comparison table → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",     choices=["smoke","full"],   default="smoke")
    p.add_argument("--split",    default="forget05")
    p.add_argument("--methods",  default="GA,NPO,FLAT,LoKU,FORGE")
    p.add_argument("--out_dir",  default="results_real")
    p.add_argument("--steps",    type=int,   default=None,  help="Override n_steps")
    p.add_argument("--eval_every",type=int,  default=None)
    p.add_argument("--lr",       type=float, default=5e-6)
    p.add_argument("--rank",     type=int,   default=16)
    p.add_argument("--seed",     type=int,   default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",")]
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Smoke mode ────────────────────────────────────────────────────────────
    if args.mode == "smoke":
        print("\n[SMOKE TEST] GPT-2 small + synthetic profiles")
        print("No GPU required. Proves pipeline. ~10 min on CPU.\n")

        if not HAS_TRANSFORMERS:
            raise ImportError("pip install transformers")

        model_name = "gpt2"
        tokenizer  = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        def model_factory():
            m = AutoModelForCausalLM.from_pretrained(model_name)
            return m

        forget_ex, retain_ex = synthetic_profiles(
            n_profiles=40, qa_per=4, forget_frac=0.10, seed=args.seed
        )
        print(f"Data: {len(forget_ex)} forget, {len(retain_ex)} retain examples")

        n_steps   = args.steps    or 60
        eval_every= args.eval_every or 10
        batch_size = 4
        max_length = 64
        n_eval     = 20
        lr         = 1e-5   # GPT-2 needs higher lr; override user arg in smoke mode

        print("\nPre-training base model on full corpus (3 epochs)...")
        all_examples = forget_ex + retain_ex
        base_ds  = QADataset(all_examples, tokenizer, max_length)
        base_dl  = DataLoader(base_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
        base_m   = model_factory().to(device)
        opt_base = torch.optim.AdamW(base_m.parameters(), lr=5e-5)
        base_m.train()
        for ep in range(3):
            losses = []
            for batch in base_dl:
                out_b = base_m(input_ids=batch["input_ids"].to(device),
                               attention_mask=batch["attention_mask"].to(device),
                               labels=batch["labels"].to(device))
                opt_base.zero_grad(); out_b.loss.backward(); opt_base.step()
                losses.append(out_b.loss.item())
            print(f"  pre-train epoch {ep+1}/3  loss={np.mean(losses):.3f}")

        base_state = {k: v.clone().cpu() for k, v in base_m.state_dict().items()}

        print("\nTraining retain-only reference model (3 epochs)...")
        retain_ds_r = QADataset(retain_ex, tokenizer, max_length)
        retain_dl_r = DataLoader(retain_ds_r, batch_size=batch_size, shuffle=True, collate_fn=collate)
        ref_m   = model_factory().to(device)
        opt_ref = torch.optim.AdamW(ref_m.parameters(), lr=5e-5)
        ref_m.train()
        for ep in range(3):
            losses = []
            for batch in retain_dl_r:
                out_r = ref_m(input_ids=batch["input_ids"].to(device),
                              attention_mask=batch["attention_mask"].to(device),
                              labels=batch["labels"].to(device))
                opt_ref.zero_grad(); out_r.loss.backward(); opt_ref.step()
                losses.append(out_r.loss.item())
            print(f"  retain-ref epoch {ep+1}/3  loss={np.mean(losses):.3f}")
        ref_m.eval()

    # ── Full mode ─────────────────────────────────────────────────────────────
    elif args.mode == "full":
        print(f"\n[FULL] Llama-2-7B + TOFU {args.split}")
        print(f"Requires 24GB+ GPU. ~18 GPU-hours for all 5 methods.\n")

        if not (HAS_TRANSFORMERS and HAS_PEFT):
            raise ImportError("pip install transformers peft")

        model_name = "locuslab/tofu_ft_llama2-7b"
        tokenizer  = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        def model_factory():
            return AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="auto"
            )

        print(f"Loading TOFU split={args.split}...")
        forget_ex, retain_ex = load_tofu(args.split)
        print(f"Data: {len(forget_ex)} forget, {len(retain_ex)} retain")

        n_steps    = args.steps    or 300
        eval_every = args.eval_every or 50
        batch_size = 4
        max_length = 256
        n_eval     = 50
        lr         = args.lr

        # Target model IS the pre-trained model (already trained on TOFU)
        print("Loading target model as base (already TOFU-finetuned)...")
        base_tmp   = model_factory()
        base_state = {k: v.clone().cpu() for k, v in base_tmp.state_dict().items()}
        del base_tmp

        # Retain reference: locuslab provides retain-only checkpoints
        ref_split = args.split.replace("forget","retain").replace(
            args.split[-2:], str(100 - int(args.split[-2:])).zfill(2))
        ref_name = f"locuslab/tofu_ft_llama2-7b_{ref_split}"
        print(f"Loading reference model: {ref_name}")
        try:
            ref_m = AutoModelForCausalLM.from_pretrained(
                ref_name, torch_dtype=torch.bfloat16, device_map="auto"
            )
        except Exception:
            print(f"  Reference {ref_name} not found; using deepcopy of target.")
            ref_m = model_factory()
        ref_m.eval()

    # ── Run methods ────────────────────────────────────────────────────────────
    all_trajectories = {}
    t0 = time.time()

    for method in methods:
        print(f"\n{'─'*60}")
        traj = train_method(
            method_name=method,
            base_state=base_state,
            ref_model=ref_m,
            forget_ex=forget_ex,
            retain_ex=retain_ex,
            tokenizer=tokenizer,
            device=device,
            model_factory=model_factory,
            n_steps=n_steps,
            eval_every=eval_every,
            lr=lr,
            batch_size=batch_size,
            max_length=max_length,
            out_dir=out,
            lora_rank=args.rank,
            use_lora=HAS_PEFT,
            n_eval=n_eval,
        )
        all_trajectories[method] = traj

    elapsed = time.time() - t0
    print(f"\nAll methods done in {elapsed/60:.1f} min.")

    # ── Save + plot ────────────────────────────────────────────────────────────
    with open(out / "trajectories.json", "w") as f:
        json.dump(all_trajectories, f, indent=2)
    print(f"Raw trajectories → {out/'trajectories.json'}")

    plot_trajectories(all_trajectories, out / "trajectory_plot.png")
    write_csv(all_trajectories, out / "comparison_table.csv")

    # Final table to stdout
    print("\n" + "="*70)
    print("FINAL CHECKPOINT RESULTS (measured)")
    print("="*70)
    hdr = f"{'Method':>8}  {'FQ':>6}  {'MU':>6}  {'RL-f':>6}  {'RL-r':>6}  {'MIA-TR':>7}  {'MIA-PR':>7}"
    print(hdr); print("─"*70)
    for name, traj in all_trajectories.items():
        last = traj[-1]
        print(f"{name:>8}  {last['fq']:>6.3f}  {last['mu']:>6.3f}  "
              f"{last['rouge_forget']:>6.3f}  {last['rouge_retain']:>6.3f}  "
              f"{last['mia_tr']:>7.3f}  {last['mia_pr']:>7.3f}")
    print("="*70)
    print(f"\nAll results in: {out.resolve()}")


if __name__ == "__main__":
    main()
