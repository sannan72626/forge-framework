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
import hashlib
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


# ── Deterministic, non-overlapping validation / final-reporting probe split ───
# Evidence-remediation fix (Phase 1, task 2): the prior pipeline reused the same
# forget_ex[:n]/retain_ex[:n] slice both to drive checkpoint selection (argmax
# FQ*MU over the trajectory) and as "the" reported value at the selected step.
# load_tofu() iterates the HF dataset in a fixed order with no shuffling, so
# index-based slicing here is fully deterministic and reproducible across every
# run of a given split -- no run-to-run randomness, and every method/seed sees
# the identical validation and reporting probes for a given split.

def _question_id(question: str) -> str:
    """Stable, content-addressed ID for a question (first 12 hex chars of sha256)."""
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:12]


def _split_hash(examples) -> str:
    """Fingerprint of a probe split: sha256 over the sorted question IDs it contains."""
    ids = sorted(_question_id(e.question) for e in examples)
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def make_probe_splits(examples, n_val=50, n_report=50):
    """Disjoint validation (checkpoint-selection) and final-reporting probe
    splits from the front of `examples`, deterministic by index position.
    Returns (val_examples, val_ids, val_hash, report_examples, report_ids, report_hash).
    Raises if there are not enough examples to carve both splits disjointly.
    """
    if len(examples) < n_val + n_report:
        raise ValueError(
            f"Not enough examples ({len(examples)}) to carve disjoint validation "
            f"(n={n_val}) and reporting (n={n_report}) probes without overlap."
        )
    val_ex = examples[:n_val]
    report_ex = examples[n_val:n_val + n_report]
    val_ids = [_question_id(e.question) for e in val_ex]
    report_ids = [_question_id(e.question) for e in report_ex]
    assert not (set(val_ids) & set(report_ids)), "validation/reporting probe overlap detected"
    return val_ex, val_ids, _split_hash(val_ex), report_ex, report_ids, _split_hash(report_ex)


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


def loss_NPO(model, ref, f_batch, r_batch, device, beta=2.5):
    # beta=0.1 (this repo's original default) amplifies the loss by 2/beta=20x under
    # the shared lr=2e-5 config, causing catastrophic collapse within ~100-150 steps
    # (confirmed empirically: MU/ROUGE crash to 0.000, see outputs/probe/npo_beta*.
    # beta=2.5 keeps NPO's peak joint FQ*MU operating point (~0.33 around step 35-65
    # of 250) intact instead of blowing past it before any checkpoint captures it.
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


def loss_SimNPO(model, ref, f_batch, r_batch, device, beta=2.5, gamma=0.0, lam=1.0):
    """SimNPO (Fan et al., NeurIPS 2025, arXiv:2410.07163) — reference-free,
    length-normalized, margin-scaled saturating ascent. No reference model needed.

    L_f = E[-2/beta * log sigmoid(beta*u - gamma)],  u = per-sample length-normalized forget NLL.
    """
    out   = model(input_ids=f_batch["input_ids"].to(device),
                  attention_mask=f_batch["attention_mask"].to(device))
    tgts  = f_batch["labels"].to(device)[..., 1:]
    mask  = (tgts != -100).float()
    tgt_c = tgts.masked_fill(tgts == -100, 0)
    logp  = F.log_softmax(out.logits[..., :-1, :], dim=-1).gather(-1, tgt_c.unsqueeze(-1)).squeeze(-1)
    u     = -((logp * mask).sum(-1) / (mask.sum(-1) + 1e-8))  # length-normalized forget NLL per sample
    l_f   = (-2.0 / beta * torch.log(torch.sigmoid(beta * u - gamma) + 1e-8)).mean()
    l_r   = model(input_ids=r_batch["input_ids"].to(device),
                  attention_mask=r_batch["attention_mask"].to(device),
                  labels=r_batch["labels"].to(device)).loss
    return l_f + lam * l_r


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
    "SimNPO": loss_SimNPO,
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
                 n_eval=25, model_name="gpt2", loss_kwargs=None,
                 val_probes=None, report_probes=None,
                 save_all_steps=False, run_id=None, seed=None, base_model_revision=None,
                 tokenizer_revision=None, code_sha256=None, config_sha256=None):
    """
    Evidence-remediation fix (Phase 1, tasks 1-3): previously this function (a)
    saved model.save_pretrained() unconditionally at the FINAL training step
    regardless of which step was later identified as the best validation
    checkpoint, and (b) evaluated every trajectory point -- including the one
    eventually "selected" -- on the same forget_ex[:n]/retain_ex[:n] slice used
    throughout training, with no probe held out purely for final reporting.

    Now:
      - `val_probes` = (val_forget_ex, val_retain_ex, val_forget_ids, val_retain_ids,
        val_forget_hash, val_retain_hash): used for every trajectory checkpoint
        and for argmax(FQ*MU) selection. Falls back to forget_ex/retain_ex
        (legacy behavior) if not provided, e.g. for the synthetic smoke mode.
      - `report_probes`: same shape, DISJOINT from val_probes, used exactly
        once -- after training ends and the best-validation LoRA state is
        restored -- to compute the number that should actually be reported.
        Never used for selection.
      - Both the final-step and the validation-selected LoRA state are saved
        to disk as separate, clearly-named checkpoints; neither overwrites
        the other. A JSON manifest records selected/final step, checkpoint
        paths, LoRA-weight sha256, and the probe split identities/hashes.

    Phase 2C addition (Task B): when `save_all_steps=True`, every scheduled
    `checkpoint()` call ALSO saves that step's live LoRA state to
    `{out_dir}/{method_name}_step{N}/`, in addition to the existing
    final/selected saves below -- purely additive I/O, no change to the
    training loop, loss, optimizer, scheduler, or data order (see
    PHASE2C_PROTOCOL.md Sec. 12 for the verified claim that in-loop eval does
    not perturb training-mode or RNG state). Only the adapter state_dict and
    its config are saved per step -- NOT optimizer/scheduler/RNG/dataloader
    state, so this does not support exact bit-for-bit training resumption,
    only post-hoc evaluation of each step's weights. Per-step identity
    records (manifest["all_step_manifests"]) carry method/seed/run_id/step,
    adapter path+sha256, base-model/tokenizer revisions, and code/config
    hashes, all passed in by the caller rather than re-derived here.

    Returns (trajectory, manifest_dict).
    """
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

    if val_probes is not None:
        val_f, val_r, val_f_ids, val_r_ids, val_f_hash, val_r_hash = val_probes
    else:
        val_f, val_r = forget_ex, retain_ex
        val_f_ids = val_r_ids = val_f_hash = val_r_hash = None

    best_j = -1.0
    best_step = None
    best_state = None  # LoRA-only state_dict at the best validation checkpoint
    step_manifests = []  # Phase 2C, Task B: one entry per saved intermediate step

    def checkpoint(step):
        nonlocal best_j, best_step, best_state
        m = measure_metrics(model, ref_model, tokenizer, val_f, val_r,
                            device, n=n_eval, max_length=max_length)
        m["step"] = step
        trajectory.append(m)
        j = m["fq"] * m["mu"]
        if j > best_j:
            best_j = j
            best_step = step
            best_state = {k: v.detach().clone().cpu()
                          for k, v in model.state_dict().items() if "lora" in k.lower()}
        print(f"  step {step:>4d}  FQ={m['fq']:.3f}  MU={m['mu']:.3f}  J={j:.4f}  "
              f"RL-f={m['rouge_forget']:.3f}  RL-r={m['rouge_retain']:.3f}  "
              f"MIA-TR={m['mia_tr']:.3f}  MIA-PR={m['mia_pr']:.3f}")

        if save_all_steps and out_dir:
            step_dir = Path(out_dir) / f"{method_name}_step{step}"
            step_dir.mkdir(parents=True, exist_ok=True)
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(step_dir)
            else:
                torch.save(model.state_dict(), step_dir / "model.pt")
            step_manifests.append({
                "method": method_name, "seed": seed, "run_id": run_id, "step": step,
                "adapter_path": str(step_dir), "adapter_sha256": _hash_checkpoint_dir(step_dir),
                "base_model_revision": base_model_revision, "tokenizer_revision": tokenizer_revision,
                "code_sha256": code_sha256, "config_sha256": config_sha256,
                "fq_nll": m["fq"], "mu_ret": m["mu"], "j_diag": j,
                "resumable_for_exact_training_replay": False,
                "resumable_for_exact_training_replay_note": (
                    "adapter weights + config only; no optimizer/scheduler/RNG/dataloader "
                    "state saved -- NOT a full resumable training checkpoint"),
            })
            print(f"    saved step-{step} checkpoint -> {step_dir}")
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

    manifest = {
        "method": method_name,
        "selected_step": best_step,
        "selected_j_diag_val": best_j,
        "final_step": n_steps,
        "validation_probe": {
            "n_forget": len(val_f), "n_retain": len(val_r),
            "forget_question_ids": val_f_ids, "retain_question_ids": val_r_ids,
            "forget_split_hash": val_f_hash, "retain_split_hash": val_r_hash,
        },
        "report_probe": None,
        "final_checkpoint_path": None, "final_checkpoint_sha256": None,
        "selected_checkpoint_path": None, "selected_checkpoint_sha256": None,
        "reported_metrics": {
            "note": "computed on the report_probes (held out from checkpoint selection) "
                    "at the validation-selected checkpoint; this, not the validation-probe "
                    "value above, is the number that should be published.",
        },
        "all_step_manifests": step_manifests,
        "run_id": run_id, "seed": seed,
        "base_model_revision": base_model_revision, "tokenizer_revision": tokenizer_revision,
        "code_sha256": code_sha256, "config_sha256": config_sha256,
    }

    if out_dir:
        out_dir = Path(out_dir)

        # 1. Save the FINAL-step state (model is still here right after the loop).
        final_ckpt = out_dir / f"{method_name}_final"
        final_ckpt.mkdir(parents=True, exist_ok=True)
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(final_ckpt)
        else:
            torch.save(model.state_dict(), final_ckpt / "model.pt")
        manifest["final_checkpoint_path"] = str(final_ckpt)
        manifest["final_checkpoint_sha256"] = _hash_checkpoint_dir(final_ckpt)
        print(f"  Final-step checkpoint -> {final_ckpt}")

        # 2. Restore the best-validation LoRA state (if one was tracked) and save it
        #    separately -- this is the checkpoint diagnostic AND official-metric
        #    evaluation should both load.
        if best_state is not None:
            sd = model.state_dict()
            for k, v in best_state.items():
                sd[k] = v.to(device)
            model.load_state_dict(sd, strict=False)

        selected_ckpt = out_dir / f"{method_name}_selected"
        selected_ckpt.mkdir(parents=True, exist_ok=True)
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(selected_ckpt)
        else:
            torch.save(model.state_dict(), selected_ckpt / "model.pt")
        manifest["selected_checkpoint_path"] = str(selected_ckpt)
        manifest["selected_checkpoint_sha256"] = _hash_checkpoint_dir(selected_ckpt)
        print(f"  Selected (t*={best_step}) checkpoint -> {selected_ckpt}")

        # 3. Final-reporting measurement on the restored, selected-checkpoint model,
        #    using the probe set that was NEVER used for selection.
        if report_probes is not None:
            rep_f, rep_r, rep_f_ids, rep_r_ids, rep_f_hash, rep_r_hash = report_probes
            rep_m = measure_metrics(model, ref_model, tokenizer, rep_f, rep_r,
                                    device, n=n_eval, max_length=max_length)
            manifest["report_probe"] = {
                "n_forget": len(rep_f), "n_retain": len(rep_r),
                "forget_question_ids": rep_f_ids, "retain_question_ids": rep_r_ids,
                "forget_split_hash": rep_f_hash, "retain_split_hash": rep_r_hash,
            }
            manifest["reported_metrics"].update({
                "fq_nll": rep_m["fq"], "mu_ret": rep_m["mu"],
                "j_diag": rep_m["fq"] * rep_m["mu"],
                "rouge_forget": rep_m["rouge_forget"], "rouge_retain": rep_m["rouge_retain"],
            })
            print(f"  Held-out report-probe metrics @ t*={best_step}: "
                  f"FQ={rep_m['fq']:.4f}  MU={rep_m['mu']:.4f}  J={rep_m['fq']*rep_m['mu']:.4f}")
        else:
            manifest["reported_metrics"]["note"] += (
                " WARNING: no report_probes given to train_method(); no held-out "
                "reported_metrics were computed. Do not substitute the validation-probe "
                "value for a 'reported' number -- that is exactly the leakage this fix "
                "exists to remove. Treat this run as incomplete for publication purposes."
            )

        json.dump(manifest, open(out_dir / f"{method_name}_checkpoint_manifest.json", "w"), indent=2)
        print(f"  Checkpoint manifest -> {out_dir / f'{method_name}_checkpoint_manifest.json'}")

    return trajectory, manifest


def _hash_checkpoint_dir(ckpt_dir: Path) -> str:
    """sha256 of the actual LoRA weight file in a saved checkpoint directory
    (adapter_model.safetensors, or model.pt as a fallback), so checkpoint
    identity can be verified independently of the directory path/mtime."""
    for name in ("adapter_model.safetensors", "model.pt", "adapter_model.bin"):
        p = ckpt_dir / name
        if p.exists():
            h = hashlib.sha256()
            h.update(p.read_bytes())
            return h.hexdigest()
    return None


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


def _build_loss_kwargs(method, args):
    kwargs = {}
    if args.margin is not None and method in ("LoKU", "FORGEH"):
        kwargs["margin"] = args.margin
    if args.beta is not None and method in ("NPO", "SimNPO", "FLAT"):
        kwargs["beta"] = args.beta
    if args.gamma_npo is not None and method == "SimNPO":
        kwargs["gamma"] = args.gamma_npo
    if args.gamma_forge is not None and method in ("FORGE", "FORGEH"):
        kwargs["gamma"] = args.gamma_forge
    return kwargs


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
    p.add_argument("--beta",       type=float, default=None,
                   help="Beta for NPO/SimNPO/FLAT (overrides loss function default)")
    p.add_argument("--gamma_npo",  type=float, default=None,
                   help="Gamma margin for SimNPO (overrides loss function default)")
    p.add_argument("--gamma_forge", type=float, default=None,
                   help="Robust-term weight for FORGE/FORGEH (0.0 disables L_rob for the ablation)")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--save_all_steps", action="store_true", default=False,
                   help="Phase 2C, Task B: save every scheduled checkpoint's LoRA state "
                        "(not just final+selected). Purely additive I/O; off by default so "
                        "every other caller of this script keeps existing behavior unchanged.")
    p.add_argument("--run_id", default=None,
                   help="Unique identity label for this training execution, recorded in "
                        "per-step manifests. Auto-generated (uuid4 hex) if omitted.")
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

    # val_probes/report_probes default to None (legacy fallback inside train_method);
    # only "full" mode currently constructs a real disjoint split (task 2 of the
    # evidence-remediation fix). Smoke mode uses synthetic data and is not used for
    # reported results, so it keeps the legacy single-probe behavior.
    val_probes = None
    report_probes = None

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

        # Evidence-remediation fix (Phase 1, task 2): carve disjoint validation
        # (checkpoint-selection) and final-reporting probes, once per run, shared
        # identically across every method in `methods` for comparability. Sized
        # adaptively so small splits (e.g. forget01, 40 examples) still get a
        # non-overlapping split instead of raising.
        def _probe_sizes(n_examples, target=n_eval):
            n_val = min(target, max(1, n_examples // 2))
            n_report = min(target, n_examples - n_val)
            return n_val, n_report

        nv_f, nr_f = _probe_sizes(len(forget_ex))
        nv_r, nr_r = _probe_sizes(len(retain_ex))
        val_f_ex, val_f_ids, val_f_hash, rep_f_ex, rep_f_ids, rep_f_hash = make_probe_splits(forget_ex, nv_f, nr_f)
        val_r_ex, val_r_ids, val_r_hash, rep_r_ex, rep_r_ids, rep_r_hash = make_probe_splits(retain_ex, nv_r, nr_r)
        val_probes = (val_f_ex, val_r_ex, val_f_ids, val_r_ids, val_f_hash, val_r_hash)
        report_probes = (rep_f_ex, rep_r_ex, rep_f_ids, rep_r_ids, rep_f_hash, rep_r_hash)
        print(f"Probe split: validation forget={len(val_f_ex)} retain={len(val_r_ex)} "
              f"(hash {val_f_hash[:8]}/{val_r_hash[:8]}) | reporting forget={len(rep_f_ex)} "
              f"retain={len(rep_r_ex)} (hash {rep_f_hash[:8]}/{rep_r_hash[:8]}) -- disjoint, verified in make_probe_splits()")

    # ── Run methods ───────────────────────────────────────────────────────────
    lora_rank = args.rank or int(cfg.get("lora_rank", 32))
    all_trajectories = {}
    all_manifests = {}
    t0 = time.time()

    run_id = None
    base_model_revision = tokenizer_revision = code_sha256 = config_sha256 = None
    if args.save_all_steps:
        import uuid
        run_id = args.run_id or uuid.uuid4().hex
        try:
            from huggingface_hub import HfApi
            _api = HfApi()
            base_model_revision = _api.model_info(model_name).sha
        except Exception as e:
            base_model_revision = f"UNRESOLVED ({e})"
        try:
            tok_src = getattr(tokenizer, "name_or_path", None)
            tokenizer_revision = HfApi().model_info(tok_src).sha if tok_src else None
        except Exception as e:
            tokenizer_revision = f"UNRESOLVED ({e})"
        code_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        config_sha256 = (hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
                         if args.config and os.path.exists(args.config) else None)
        print(f"Phase 2C full-coverage mode: run_id={run_id}, "
              f"base_model_revision={base_model_revision}, tokenizer_revision={tokenizer_revision}, "
              f"code_sha256={code_sha256[:12]}..., config_sha256={(config_sha256 or '')[:12]}...")

    for method in methods:
        print(f"\n{'─'*60}")
        traj, manifest = train_method(
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
            loss_kwargs=_build_loss_kwargs(method, args),
            val_probes=val_probes,
            report_probes=report_probes,
            save_all_steps=args.save_all_steps,
            run_id=run_id, seed=args.seed,
            base_model_revision=base_model_revision, tokenizer_revision=tokenizer_revision,
            code_sha256=code_sha256, config_sha256=config_sha256,
        )
        all_trajectories[method] = traj
        all_manifests[method] = manifest

    print(f"\nAll methods done in {(time.time()-t0)/60:.1f} min.")

    traj_path = out / "trajectories.json"
    with open(traj_path, "w") as f:
        json.dump(all_trajectories, f, indent=2)
    print(f"Trajectories → {traj_path}")

    manifests_path = out / "checkpoint_manifests.json"
    with open(manifests_path, "w") as f:
        json.dump(all_manifests, f, indent=2)
    print(f"Checkpoint manifests → {manifests_path}")

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
