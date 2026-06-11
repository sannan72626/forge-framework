#!/usr/bin/env python
"""
scripts/run_experiments.py — FORGE Week 4 main experiment runner.

Modes:
  smoke   GPT-2 + synthetic profiles. GPU required. ~5 min on RTX 3050. Proves pipeline end-to-end.
  full    Phi-3-mini (Task 0 checkpoint) + real TOFU splits. RTX 3050 8 GB.

Usage:
  # Smoke test (GPU, ~5 min):
  python scripts/run_experiments.py --mode smoke --methods GA,FORGE --steps 30 --eval_every 10 --out_dir outputs/wk4/smoke/

  # Single method, full TOFU-5%:
  python scripts/run_experiments.py --mode full --method FORGE --split forget05 --seed 42 \
      --config configs/forge_default.yaml --out_dir outputs/wk4/FORGE/forget05/seed42/

  # All methods loop (called from CLAUDE.md week 4/6/8 tasks):
  for METHOD in GA NPO FLAT LoKU FORGE; do
    python scripts/run_experiments.py --mode full --method $METHOD --split forget05 --seed 42 \
        --config configs/forge_default.yaml --out_dir outputs/wk4/$METHOD/forget05/seed42/
  done

Output per run:
  <out_dir>/trajectories.json       [{step, fq, mu, rouge_forget, rouge_retain, mia_tr, mia_pr}, ...]
  <out_dir>/trajectory_plot.png
  <out_dir>/comparison_table.csv
  <out_dir>/<method>/               LoRA adapter checkpoint
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── Optional heavy deps ───────────────────────────────────────────────────────

try:
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                               BitsAndBytesConfig,
                               get_cosine_schedule_with_warmup)
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from peft import (LoraConfig, TaskType, get_peft_model,
                      prepare_model_for_kbit_training)
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

try:
    import bitsandbytes  # noqa: F401
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

# ── Config loader ─────────────────────────────────────────────────────────────

def load_config(path: Optional[str]) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # Parse minimal YAML manually (key: value lines only)
        cfg = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    k, _, v = line.partition(":")
                    v = v.strip()
                    try:
                        cfg[k.strip()] = float(v) if "." in v or "e" in v.lower() else int(v)
                    except ValueError:
                        cfg[k.strip()] = v
        return cfg


# ── Data ──────────────────────────────────────────────────────────────────────

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
            (f"Where does {name} live?",              f"{name} lives in {city}."),
            (f"What genre does {name} write?",        f"{name} writes {genre}."),
            (f"When did {name} debut?",               f"{name} debuted in the {decade}."),
            (f"What city is {name} based in?",        f"{name} is based in {city}."),
            (f"What kind of books does {name} write?",f"{name} writes {genre} books."),
            (f"Which decade is {name} known from?",   f"{name} is from the {decade}."),
        ]
        rng_p.shuffle(pool)
        is_forget = pid < n_forget_p
        bucket = forget if is_forget else retain
        for q, a in pool[:qa_per]:
            bucket.append(QAExample(q, a, is_forget=is_forget))

    return forget, retain


def load_tofu(forget_split="forget05"):
    from datasets import load_dataset
    retain_pct = 100 - int(forget_split.replace("forget", ""))
    retain_split = f"retain{retain_pct:02d}"
    forget_ds = load_dataset("locuslab/TOFU", forget_split)["train"]
    retain_ds = load_dataset("locuslab/TOFU", retain_split)["train"]
    forget = [QAExample(e["question"], e["answer"], is_forget=True)  for e in forget_ds]
    retain = [QAExample(e["question"], e["answer"], is_forget=False) for e in retain_ds]
    return forget, retain


# ── Dataset ───────────────────────────────────────────────────────────────────

class QADataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=128):
        self.examples  = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.examples)

    def __getitem__(self, idx):
        ex  = self.examples[idx]
        txt = f"Q: {ex.question}\nA: {ex.answer}"
        enc = self.tokenizer(txt, return_tensors="pt", truncation=True,
                             max_length=self.max_length, padding="max_length")
        ids  = enc["input_ids"][0]
        mask = enc["attention_mask"][0]
        lbl  = ids.clone(); lbl[mask == 0] = -100
        return {"input_ids": ids, "attention_mask": mask, "labels": lbl}


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ── Metrics ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def _nll(model, tokenizer, text, device, max_length=128):
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=max_length).to(device)
    return model(**enc, labels=enc["input_ids"]).loss.item()


@torch.no_grad()
def _generate(model, tokenizer, question, device, max_new=20, max_length=100):
    prompt = f"Q: {question}\nA:"
    enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=max_length).to(device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    return full[len(prompt):].strip()


def measure_metrics(model, ref_model, tokenizer, forget_ex, retain_ex, device,
                    n=25, max_length=128, max_new=20):
    model.eval()
    sub_f = forget_ex[:n]
    sub_r = retain_ex[:n]

    def nlls(examples, mdl):
        return [_nll(mdl, tokenizer,
                     f"Q: {e.question}\nA: {e.answer}", device, max_length)
                for e in examples]

    nll_f_model = nlls(sub_f, model)
    nll_f_ref   = nlls(sub_f, ref_model)
    nll_r_model = nlls(sub_r, model)
    nll_r_ref   = nlls(sub_r, ref_model)

    # FQ: KS p-value (higher = model's forget NLL distribution shifted away from ref)
    if HAS_SCIPY and len(nll_f_model) >= 5:
        _, fq = scipy_stats.ks_2samp(nll_f_model, nll_f_ref)
    else:
        fq = float(np.clip(1.0 - abs(np.mean(nll_f_model) - np.mean(nll_f_ref)) / 3.0, 0, 1))

    # ROUGE-L on generated answers
    def rouge_mean(examples):
        if not HAS_ROUGE:
            return 0.0
        scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [scorer.score(e.answer,
                               _generate(model, tokenizer, e.question, device, max_new))
                  ["rougeL"].fmeasure for e in examples[:min(n, 15)]]
        return float(np.mean(scores))

    rouge_f = rouge_mean(sub_f)
    rouge_r = rouge_mean(sub_r)

    # MU: harmonic mean of retain ROUGE-L and perplexity recovery
    ppl_model = float(np.mean([math.exp(min(v, 20)) for v in nll_r_model]))
    ppl_ref   = float(np.mean([math.exp(min(v, 20)) for v in nll_r_ref]))
    ppl_score = float(np.clip(math.exp(-(ppl_model - ppl_ref) / max(ppl_ref, 1e-6)), 0, 1))
    mu = 2 * rouge_r * ppl_score / (rouge_r + ppl_score + 1e-6) if (rouge_r + ppl_score) > 0 else 0.0

    # MIA-TR
    def perturb(ans):
        w = ans.split(); random.shuffle(w); return " ".join(w) if len(w) > 1 else ans + " UNK"

    @torch.no_grad()
    def log_prob(q, a):
        txt = f"Q: {q}\nA: {a}"
        enc = tokenizer(txt, return_tensors="pt", truncation=True,
                        max_length=max_length).to(device)
        out = model(**enc)
        logp = F.log_softmax(out.logits[..., :-1, :], dim=-1)
        tgts = enc["input_ids"][..., 1:]
        return logp.gather(-1, tgts.unsqueeze(-1)).squeeze(-1).mean().item()

    tr_ratios = [math.exp(log_prob(e.question, e.answer) -
                          log_prob(e.question, perturb(e.answer)))
                 for e in sub_f[:min(n, 15)]]
    mia_tr = float(np.mean(tr_ratios))

    # MIA-PR
    pr_ratios = [math.exp(-nll_f_model[i] + nll_f_ref[i]) for i in range(len(nll_f_model))]
    mia_pr = float(np.mean(pr_ratios))

    return dict(fq=fq, mu=mu, rouge_forget=rouge_f, rouge_retain=rouge_r,
                mia_tr=mia_tr, mia_pr=mia_pr)


# ── Loss functions ────────────────────────────────────────────────────────────

def loss_GA(model, ref, f_batch, r_batch, device):
    return -model(input_ids=f_batch["input_ids"].to(device),
                  attention_mask=f_batch["attention_mask"].to(device),
                  labels=f_batch["labels"].to(device)).loss


def loss_NPO(model, ref, f_batch, r_batch, device, beta=0.1):
    out     = model(input_ids=f_batch["input_ids"].to(device),
                    attention_mask=f_batch["attention_mask"].to(device))
    with torch.no_grad():
        ref_out = ref(input_ids=f_batch["input_ids"].to(device),
                      attention_mask=f_batch["attention_mask"].to(device))
    tgts   = f_batch["labels"].to(device)[..., 1:]
    mask   = (tgts != -100).float()
    tgt_c  = tgts.masked_fill(tgts == -100, 0)
    logp   = F.log_softmax(out.logits[..., :-1, :], dim=-1).gather(-1, tgt_c.unsqueeze(-1)).squeeze(-1)
    rlogp  = F.log_softmax(ref_out.logits[..., :-1, :], dim=-1).gather(-1, tgt_c.unsqueeze(-1)).squeeze(-1)
    ratio  = ((logp - rlogp) * mask).sum(-1) / (mask.sum(-1) + 1e-8)
    return (-2.0 / beta * torch.log(torch.sigmoid(-beta * ratio) + 1e-8)).mean()


def loss_GD(model, ref, f_batch, r_batch, device):
    l_f   = -model(input_ids=f_batch["input_ids"].to(device),
                   attention_mask=f_batch["attention_mask"].to(device),
                   labels=f_batch["labels"].to(device)).loss
    l_rs  = model(input_ids=r_batch["input_ids"].to(device),
                  attention_mask=r_batch["attention_mask"].to(device),
                  labels=r_batch["labels"].to(device)).loss
    with torch.no_grad():
        l_rr = ref(input_ids=r_batch["input_ids"].to(device),
                   attention_mask=r_batch["attention_mask"].to(device),
                   labels=r_batch["labels"].to(device)).loss
    return l_f + (l_rs - l_rr).abs()


def loss_KL(model, ref, f_batch, r_batch, device):
    l_f   = -model(input_ids=f_batch["input_ids"].to(device),
                   attention_mask=f_batch["attention_mask"].to(device),
                   labels=f_batch["labels"].to(device)).loss
    out_s = model(input_ids=r_batch["input_ids"].to(device),
                  attention_mask=r_batch["attention_mask"].to(device))
    with torch.no_grad():
        out_r = ref(input_ids=r_batch["input_ids"].to(device),
                    attention_mask=r_batch["attention_mask"].to(device))
    kl = F.kl_div(F.log_softmax(out_s.logits, -1),
                  F.softmax(out_r.logits.detach(), -1), reduction="batchmean")
    return l_f + kl


def loss_FLAT(model, ref, f_batch, r_batch, device, beta=2.0):
    out   = model(input_ids=f_batch["input_ids"].to(device),
                  attention_mask=f_batch["attention_mask"].to(device))
    tgts  = f_batch["labels"].to(device)[..., 1:]
    mask  = (tgts != -100).float()
    tgt_c = tgts.masked_fill(tgts == -100, 0)
    logp  = F.log_softmax(out.logits[..., :-1, :], dim=-1).gather(-1, tgt_c.unsqueeze(-1)).squeeze(-1)
    return (-torch.log(1.0 + beta * logp.exp() + 1e-8) * mask).sum() / (mask.sum() + 1e-8)


def loss_LoKU(model, ref, f_batch, r_batch, device, margin=4.0, lam=1.0):
    l_f = -torch.clamp(model(input_ids=f_batch["input_ids"].to(device),
                              attention_mask=f_batch["attention_mask"].to(device),
                              labels=f_batch["labels"].to(device)).loss, max=margin)
    l_r = model(input_ids=r_batch["input_ids"].to(device),
                attention_mask=r_batch["attention_mask"].to(device),
                labels=r_batch["labels"].to(device)).loss
    return l_f + lam * l_r


def loss_FORGE(model, ref, f_batch, r_batch, device,
               lam=1.0, gamma=0.5, eta=0.01, ihl_margin=4.0):
    # ℒ_forget: bounded IHL
    nll_f = model(input_ids=f_batch["input_ids"].to(device),
                  attention_mask=f_batch["attention_mask"].to(device),
                  labels=f_batch["labels"].to(device)).loss
    l_f = -torch.tanh(nll_f / ihl_margin) * ihl_margin

    # ℒ_retain
    l_r = model(input_ids=r_batch["input_ids"].to(device),
                attention_mask=r_batch["attention_mask"].to(device),
                labels=r_batch["labels"].to(device)).loss

    # ℒ_robust: gradient ascent on token-dropped forget (robust forgetting)
    # Negated so the model can't answer even perturbed queries → adversarial robustness
    ids_rob = f_batch["input_ids"].clone().to(device)
    ids_rob[torch.rand_like(ids_rob.float()) < 0.20] = 0
    nll_rob = model(input_ids=ids_rob,
                    attention_mask=f_batch["attention_mask"].to(device),
                    labels=f_batch["labels"].to(device)).loss
    l_rob = -torch.tanh(nll_rob / ihl_margin) * ihl_margin

    return l_f + lam * l_r + gamma * l_rob


def loss_FORGEH(model, ref, f_batch, r_batch, device,
                margin=4.0, lam=1.0, gamma=0.5):
    """FORGE-Hybrid: LoKU's hard-clamp forget + perturbation-robustness term.

    Diagnosis (llama2_7b_report.md): FORGE's tanh-bounded ascent overshoots at scale,
    while LoKU's HARD CLAMP stops forgetting at `margin` → stable at 7B. The hybrid keeps
    LoKU's stable clamp on the forget term AND adds a clamped robustness term on
    input-perturbed forget — aiming to match LoKU's FQ/MU while adding robustness.

        ℒ = −clamp(NLL_f, max=m) + λ·NLL_r + γ·(−clamp(NLL(perturb_f), max=m))
    """
    # ℒ_forget: LoKU hard clamp (stops at margin → no overshoot)
    nll_f = model(input_ids=f_batch["input_ids"].to(device),
                  attention_mask=f_batch["attention_mask"].to(device),
                  labels=f_batch["labels"].to(device)).loss
    l_f = -torch.clamp(nll_f, max=margin)

    # ℒ_retain: utility anchor
    l_r = model(input_ids=r_batch["input_ids"].to(device),
                attention_mask=r_batch["attention_mask"].to(device),
                labels=r_batch["labels"].to(device)).loss

    # ℒ_robust: SAME hard clamp on input-perturbed forget (stable robustness)
    ids_rob = f_batch["input_ids"].clone().to(device)
    ids_rob[torch.rand_like(ids_rob.float()) < 0.20] = 0
    nll_rob = model(input_ids=ids_rob,
                    attention_mask=f_batch["attention_mask"].to(device),
                    labels=f_batch["labels"].to(device)).loss
    l_rob = -torch.clamp(nll_rob, max=margin)

    return l_f + lam * l_r + gamma * l_rob


METHOD_LOSSES = {
    "GA":     loss_GA,
    "GD":     loss_GD,
    "KL":     loss_KL,
    "NPO":    loss_NPO,
    "FLAT":   loss_FLAT,
    "LoKU":   loss_LoKU,
    "FORGE":  loss_FORGE,
    "FORGEH": loss_FORGEH,
}


# ── LoRA target modules by model family ──────────────────────────────────────

def lora_targets(model_name_or_path: str) -> List[str]:
    name = str(model_name_or_path).lower()
    if "phi-3" in name or "phi_3" in name:
        return ["qkv_proj", "o_proj"]   # Phi-3 combined attention
    if "phi" in name:
        return ["q_proj", "v_proj"]     # Phi-1.5 / Phi-2 separate projections
    if "llama" in name or "mistral" in name:
        return ["q_proj", "v_proj"]
    if "opt" in name:
        return ["q_proj", "v_proj"]
    return ["c_attn"]   # GPT-2 default


# ── Training loop ─────────────────────────────────────────────────────────────

def train_method(method_name, base_state, ref_model, forget_ex, retain_ex,
                 tokenizer, device, model_factory,
                 n_steps=60, eval_every=10, lr=5e-6,
                 batch_size=4, max_length=128,
                 out_dir=None, lora_rank=16, use_lora=True,
                 n_eval=25, model_name="gpt2", loss_kwargs=None):
    loss_kwargs = loss_kwargs or {}

    print(f"\n{'='*60}\nMethod: {method_name}\n{'='*60}")

    model = model_factory()
    if base_state is not None:
        model.load_state_dict(base_state, strict=False)

    # device_map="auto" models are already on GPU via accelerate hooks; .to() would fail.
    uses_device_map = bool(getattr(model, "hf_device_map", None))
    if not uses_device_map:
        model.to(device)

    if use_lora and HAS_PEFT:
        if uses_device_map and HAS_BNB:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model)
        targets = lora_targets(model_name)
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=lora_rank, lora_alpha=lora_rank * 2,
            lora_dropout=0.0, target_modules=targets,
        )
        model = get_peft_model(model, lora_cfg)
        print(f"  LoRA rank={lora_rank}, targets={targets}")

    forget_dl = DataLoader(QADataset(forget_ex, tokenizer, max_length),
                           batch_size=batch_size, shuffle=True, collate_fn=collate)
    retain_dl = DataLoader(QADataset(retain_ex, tokenizer, max_length),
                           batch_size=batch_size, shuffle=True, collate_fn=collate)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    loss_fn   = METHOD_LOSSES[method_name]
    trajectory = []

    def checkpoint(step):
        m = measure_metrics(model, ref_model, tokenizer, forget_ex, retain_ex,
                            device, n=n_eval, max_length=max_length)
        m["step"] = step
        trajectory.append(m)
        print(f"  step {step:>4d}  FQ={m['fq']:.3f}  MU={m['mu']:.3f}  "
              f"RL-f={m['rouge_forget']:.3f}  RL-r={m['rouge_retain']:.3f}  "
              f"MIA-TR={m['mia_tr']:.3f}  MIA-PR={m['mia_pr']:.3f}")
        return m

    checkpoint(0)
    f_iter = iter(forget_dl)
    r_iter = iter(retain_dl)

    for step in range(1, n_steps + 1):
        model.train()
        try:    f_b = next(f_iter)
        except: f_iter = iter(forget_dl); f_b = next(f_iter)
        try:    r_b = next(r_iter)
        except: r_iter = iter(retain_dl); r_b = next(r_iter)

        loss = loss_fn(model, ref_model, f_b, r_b, device, **loss_kwargs)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        if step % eval_every == 0:
            checkpoint(step)

    if out_dir:
        ckpt = Path(out_dir) / method_name
        ckpt.mkdir(parents=True, exist_ok=True)
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(ckpt)
        else:
            torch.save(model.state_dict(), ckpt / "model.pt")
        print(f"  Checkpoint → {ckpt}")

    return trajectory


# ── Plotting + IO ─────────────────────────────────────────────────────────────

def plot_trajectories(all_traj, out_path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    WHITE = "#FFFFFF"; NAVY = "#1E2761"
    COLORS = {"GA":"#B23A48","GD":"#C96B5B","KL":"#D4956B",
               "NPO":"#E07A5F","FLAT":"#3D5A80","LoKU":"#81B29A","FORGE":"#1E2761"}

    fig, ax = plt.subplots(figsize=(9.8, 6.2), dpi=185)
    fig.patch.set_facecolor(WHITE); ax.set_facecolor("#FAFAFB")
    ax.axhline(0.80, color="#D9D9DE", lw=0.7, ls="--")
    ax.axvline(0.80, color="#D9D9DE", lw=0.7, ls="--")

    origin = all_traj[list(all_traj)[0]][0]
    ax.scatter([origin["fq"]], [origin["mu"]], s=80, color="#888",
               edgecolor=WHITE, lw=1.5, zorder=4)
    ax.text(origin["fq"] + 0.01, origin["mu"] - 0.015, "t=0", fontsize=8, color="#555")

    for name, traj in all_traj.items():
        fq = [p["fq"] for p in traj]
        mu = [p["mu"] for p in traj]
        c  = COLORS.get(name, "#888")
        lw = 2.8 if name == "FORGE" else 1.9
        ax.plot(fq, mu, color=c, lw=lw, alpha=0.95, zorder=3, label=name)
        ax.scatter(fq[1:], mu[1:], color=c, s=28, zorder=4, edgecolor=WHITE, lw=0.8)
        if len(fq) > 2:
            mid = len(fq) // 2
            ax.annotate("", xy=(fq[mid+1], mu[mid+1]), xytext=(fq[mid], mu[mid]),
                        arrowprops=dict(arrowstyle="->", color=c, lw=lw))
        try:
            import matplotlib.patheffects as pe
            ax.text(fq[-1] + 0.008, mu[-1], name, color=c, fontsize=10, fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2.5, foreground=WHITE)])
        except Exception:
            ax.text(fq[-1] + 0.008, mu[-1], name, color=c, fontsize=10, fontweight="bold")

    handles = [Line2D([0], [0], color=COLORS.get(n, "#888"), lw=2.4, label=n) for n in all_traj]
    ax.legend(handles=handles, loc="lower left", fontsize=9, frameon=True, edgecolor="#DDD")
    ax.set_xlabel("Forget Quality (FQ) ↑", fontsize=11.5)
    ax.set_ylabel("Model Utility (MU) ↑", fontsize=11.5)
    ax.set_title("MEASURED FQ × MU trajectories", fontsize=11.5, color=NAVY, loc="left")
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=185, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"Plot → {out_path}")


def write_csv(all_traj, out_path):
    import csv
    metrics = ["fq", "mu", "rouge_forget", "rouge_retain", "mia_tr", "mia_pr"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + metrics)
        for name, traj in all_traj.items():
            last = traj[-1]
            w.writerow([name] + [f"{last.get(m, 0):.4f}" for m in metrics])
    print(f"CSV → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",       choices=["smoke", "full"], default="smoke")
    p.add_argument("--split",      default="forget05")
    p.add_argument("--methods",    default=None,
                   help="Comma-separated list of methods (overrides --method)")
    p.add_argument("--method",     default=None,
                   help="Single method name; use --methods for multiple")
    p.add_argument("--out_dir",    default="outputs/wk4/smoke")
    p.add_argument("--config",     default=None, help="Path to YAML config file")
    p.add_argument("--base_model", default=None,
                   help="Full-mode: HF model ID or local path "
                        "(default: locuslab/tofu_ft_llama2-7b)")
    p.add_argument("--ref_model",  default=None,
                   help="Full-mode: retain-only reference model "
                        "(default: locuslab/tofu_ft_llama2-7b_retainXX auto-resolved)")
    p.add_argument("--steps",      type=int,   default=None)
    p.add_argument("--eval_every", type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--rank",       type=int,   default=None)
    p.add_argument("--margin",     type=float, default=None,
                   help="Clamp margin for LoKU/FORGEH (set near oracle NLL ~2.3)")
    p.add_argument("--seed",       type=int,   default=42)
    args = p.parse_args()

    # Resolve methods list
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]
    elif args.method:
        methods = [args.method]
    else:
        methods = ["GA", "NPO", "FLAT", "LoKU", "FORGE"]

    # Validate
    for m in methods:
        if m not in METHOD_LOSSES:
            print(f"Unknown method '{m}'. Valid: {list(METHOD_LOSSES.keys())}")
            sys.exit(1)

    cfg = load_config(args.config)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Device: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory // 1024**3} GB)")
    else:
        print("Device: CPU")

    # ── Smoke mode ────────────────────────────────────────────────────────────
    if args.mode == "smoke":
        if device.type != "cuda":
            raise RuntimeError("GPU required. No CUDA device detected.")
        print(f"\n[SMOKE] GPT-2 + synthetic profiles | "
              f"{torch.cuda.get_device_name(0)}\n")
        if not HAS_TRANSFORMERS:
            raise ImportError("pip install transformers")

        model_name = "gpt2"
        tokenizer  = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        def model_factory():
            return AutoModelForCausalLM.from_pretrained(model_name)

        forget_ex, retain_ex = synthetic_profiles(n_profiles=40, qa_per=4,
                                                   forget_frac=0.10, seed=args.seed)
        print(f"Data: {len(forget_ex)} forget, {len(retain_ex)} retain")

        n_steps    = args.steps    or 60
        eval_every = args.eval_every or 10
        batch_size = 4
        max_length = 64
        n_eval     = 20
        lr         = args.lr or 1e-5

        # Pre-train base model on full corpus
        print("\nPre-training base model (3 epochs)...")
        all_ex  = forget_ex + retain_ex
        base_dl = DataLoader(QADataset(all_ex, tokenizer, max_length),
                             batch_size=batch_size, shuffle=True, collate_fn=collate)
        base_m  = model_factory().to(device)
        opt_b   = torch.optim.AdamW(base_m.parameters(), lr=5e-5)
        for ep in range(3):
            losses = []
            for batch in base_dl:
                base_m.train()
                o = base_m(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device),
                           labels=batch["labels"].to(device))
                opt_b.zero_grad(); o.loss.backward(); opt_b.step()
                losses.append(o.loss.item())
            print(f"  base epoch {ep+1}/3  loss={np.mean(losses):.3f}")
        base_state = {k: v.clone().cpu() for k, v in base_m.state_dict().items()}

        # Retain-only reference
        print("\nTraining retain reference model (3 epochs)...")
        ref_dl = DataLoader(QADataset(retain_ex, tokenizer, max_length),
                            batch_size=batch_size, shuffle=True, collate_fn=collate)
        ref_m  = model_factory().to(device)
        opt_r  = torch.optim.AdamW(ref_m.parameters(), lr=5e-5)
        for ep in range(3):
            losses = []
            for batch in ref_dl:
                ref_m.train()
                o = ref_m(input_ids=batch["input_ids"].to(device),
                          attention_mask=batch["attention_mask"].to(device),
                          labels=batch["labels"].to(device))
                opt_r.zero_grad(); o.loss.backward(); opt_r.step()
                losses.append(o.loss.item())
            print(f"  ref epoch {ep+1}/3  loss={np.mean(losses):.3f}")
        ref_m.eval()

    # ── Full mode (Llama-2-7B bfloat16 + TOFU) ───────────────────────────────
    elif args.mode == "full":
        if not (HAS_TRANSFORMERS and HAS_PEFT):
            raise ImportError("pip install transformers peft")
        if device.type != "cuda":
            raise RuntimeError("GPU required for full mode.")

        # Use a local checkpoint if provided, otherwise use locuslab base (no retain-only HF model exists)
        # Build the oracle first: python scripts/finetune_retain_ref.py --split retainXX
        base_path = args.base_model or "locuslab/tofu_ft_llama2-7b"
        retain_pct = 100 - int(args.split.replace("forget", ""))
        local_ref = f"outputs/wk0/llama2_tofu_retain{retain_pct:02d}/"
        import os
        ref_path = args.ref_model or (local_ref if os.path.exists(local_ref) else "locuslab/tofu_ft_llama2-7b")

        model_name = base_path
        # Auto-detect whether to use 4-bit (large models ≥7B) or bfloat16 (small models <3B)
        _name_lower = base_path.lower()
        _small_model = any(k in _name_lower for k in ["phi-1", "phi-2", "phi_1", "phi_2",
                                                        "1.5", "opt-1.3", "gpt2"])
        _dtype_str = "bfloat16" if _small_model else "4-bit NF4"
        print(f"\n[FULL] {base_path} ({_dtype_str}) + TOFU {args.split} | RTX 3090 23 GB")
        print(f"  base: {base_path}")
        print(f"  ref:  {ref_path}")

        if _small_model:
            # bfloat16 — small model fits entirely without quantization
            def model_factory():
                return AutoModelForCausalLM.from_pretrained(
                    base_path, torch_dtype=torch.bfloat16)
        else:
            # 4-bit NF4 for large models (7B+)
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            )
            def model_factory():
                return AutoModelForCausalLM.from_pretrained(
                    base_path, quantization_config=bnb_cfg, device_map="auto")

        try:
            tokenizer = AutoTokenizer.from_pretrained(base_path)
        except OSError:
            # Some HF fine-tune repos don't ship a tokenizer; fall back to base model
            _tok_fallbacks = {"phi-1.5": "microsoft/phi-1_5", "phi-2": "microsoft/phi-2",
                              "llama2": "meta-llama/Llama-2-7b-hf"}
            _tok_src = next((v for k, v in _tok_fallbacks.items() if k in base_path.lower()),
                            "microsoft/phi-1_5")
            print(f"  Tokenizer not found in {base_path}, loading from {_tok_src}")
            tokenizer = AutoTokenizer.from_pretrained(_tok_src)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        print(f"Loading TOFU {args.split}...")
        forget_ex, retain_ex = load_tofu(args.split)
        print(f"  {len(forget_ex)} forget, {len(retain_ex)} retain")

        base_state = None

        print(f"Loading reference model ({_dtype_str}): {ref_path}")
        try:
            if _small_model:
                from peft import PeftModel
                import os as _os
                if _os.path.exists(_os.path.join(ref_path, "adapter_config.json")):
                    _base_ref = AutoModelForCausalLM.from_pretrained(
                        "microsoft/phi-1_5", torch_dtype=torch.bfloat16)
                    ref_m = PeftModel.from_pretrained(_base_ref, ref_path).merge_and_unload()
                else:
                    ref_m = AutoModelForCausalLM.from_pretrained(
                        ref_path, torch_dtype=torch.bfloat16)
                ref_m = ref_m.to(device)
            else:
                ref_m = AutoModelForCausalLM.from_pretrained(
                    ref_path, quantization_config=bnb_cfg, device_map="auto")
        except Exception as e:
            print(f"  Warning: {e}")
            ref_m = model_factory()
        ref_m.eval()

        n_steps    = args.steps    or int(cfg.get("max_steps", 300))
        eval_every = args.eval_every or int(cfg.get("eval_every", 25))
        batch_size = int(cfg.get("per_device_batch", 2))
        max_length = int(cfg.get("max_length", 256))
        n_eval     = 50
        lr         = args.lr or float(cfg.get("lr", 5e-6))

    # ── Run methods ───────────────────────────────────────────────────────────
    lora_rank = args.rank or int(cfg.get("lora_rank", 32))
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
            lora_rank=lora_rank,
            use_lora=HAS_PEFT,
            n_eval=n_eval,
            model_name=model_name,
            loss_kwargs=({"margin": args.margin}
                         if (args.margin is not None and method in ("LoKU", "FORGEH"))
                         else {}),
        )
        all_trajectories[method] = traj

    print(f"\nAll methods done in {(time.time()-t0)/60:.1f} min.")

    traj_path = out / "trajectories.json"
    with open(traj_path, "w") as f:
        json.dump(all_trajectories, f, indent=2)
    print(f"Trajectories → {traj_path}")

    plot_trajectories(all_trajectories, out / "trajectory_plot.png")
    write_csv(all_trajectories, out / "comparison_table.csv")

    # Summary table
    print("\n" + "=" * 70)
    print("FINAL CHECKPOINT (measured)")
    print("=" * 70)
    print(f"{'Method':>8}  {'FQ':>6}  {'MU':>6}  {'RL-f':>6}  "
          f"{'RL-r':>6}  {'MIA-TR':>7}  {'MIA-PR':>7}")
    print("─" * 70)
    for name, traj in all_trajectories.items():
        last = traj[-1]
        print(f"{name:>8}  {last['fq']:>6.3f}  {last['mu']:>6.3f}  "
              f"{last['rouge_forget']:>6.3f}  {last['rouge_retain']:>6.3f}  "
              f"{last['mia_tr']:>7.3f}  {last['mia_pr']:>7.3f}")
    print("=" * 70)
    print(f"\nOutputs in: {out.resolve()}")


if __name__ == "__main__":
    main()
