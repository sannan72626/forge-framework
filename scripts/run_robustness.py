#!/usr/bin/env python
"""
scripts/run_robustness.py — Perturbed-prompt robustness evaluation (addressing
review [R11]/[R4]: "Robustness is a named contribution and a design requirement,
yet it is never evaluated" / "An ablation (gamma=0 vs gamma=0.5) plus an actual
perturbed-prompt evaluation is required to justify this term.")

For a trained checkpoint, evaluates whether forget-set answers remain suppressed
under THREE query perturbations relative to the canonical (training-time) prompt:

  1. token_drop   — same 20% token-drop used in FORGE's L_rob term (in-distribution
                    check: did the model actually learn to generalize the robustness
                    it was explicitly trained for?)
  2. word_shuffle — local word-order permutation (out-of-distribution perturbation
                    type never seen during training)
  3. truncation   — drop the last ~30% of the prompt (incomplete-query probe)

For each condition, reports:
  - rouge_leak : ROUGE-L(generated answer, TRUE memorized answer) — higher = more
                 leaked/recoverable information (LOWER is better post-unlearning)
  - exact_leak : fraction of examples where rouge_leak > 0.5 (coarse "still knows it")

Usage:
  # Single checkpoint:
  python scripts/run_robustness.py --checkpoint outputs/wk8/FORGE/forget05/seed42/FORGE \
      --base_model locuslab/tofu_ft_phi-1.5 --split forget05 --out outputs/wk_robustness/FORGE_g05.json

  # Ablation comparison (requires two checkpoints already trained: gamma=0.5 default
  # and gamma=0 ablation — see scripts/run_experiments.py --gamma_forge 0.0):
  python scripts/run_robustness.py --compare \
      --checkpoint_a outputs/wk8/FORGE/forget05/seed42/FORGE --label_a "FORGE (gamma=0.5)" \
      --checkpoint_b outputs/wk_ablation/FORGE_gamma0/forget05/seed42/FORGE --label_b "FORGE (gamma=0)" \
      --base_model locuslab/tofu_ft_phi-1.5 --split forget05 --out outputs/wk_robustness/ablation.json
"""

import argparse, json, random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from rouge_score import rouge_scorer as rouge_lib
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

TOK_MODEL = "microsoft/phi-1_5"


def load_forget_examples(split, n=60):
    ds = load_dataset("locuslab/TOFU", split)["train"]
    return [{"question": e["question"], "answer": e["answer"]} for e in list(ds)[:n]]


# ── Perturbation operators ────────────────────────────────────────────────────

def perturb_token_drop(text, rho=0.20, seed=0):
    rng = random.Random(seed)
    words = text.split()
    kept = [w for w in words if rng.random() >= rho]
    return " ".join(kept) if kept else text


def perturb_word_shuffle(text, window=3, seed=0):
    rng = random.Random(seed)
    words = text.split()
    for i in range(0, len(words) - window, window):
        chunk = words[i:i + window]
        rng.shuffle(chunk)
        words[i:i + window] = chunk
    return " ".join(words)


def perturb_truncation(text, keep_frac=0.7):
    words = text.split()
    n_keep = max(1, int(len(words) * keep_frac))
    return " ".join(words[:n_keep])


PERTURBATIONS = {
    "canonical":   lambda t, i: t,
    "token_drop":  lambda t, i: perturb_token_drop(t, seed=i),
    "word_shuffle": lambda t, i: perturb_word_shuffle(t, seed=i),
    "truncation":  lambda t, i: perturb_truncation(t),
}


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_robustness(model, tok, device, examples, max_new=30):
    scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
    model.eval()
    results = {name: [] for name in PERTURBATIONS}
    for name, fn in PERTURBATIONS.items():
        for i, e in enumerate(examples):
            q = fn(e["question"], i)
            prompt = f"Q: {q}\nA:"
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=200).to(device)
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            leak = scorer.score(e["answer"], gen)["rougeL"].fmeasure
            results[name].append(leak)
    return {
        name: {
            "rouge_leak_mean": float(np.mean(vals)),
            "exact_leak_frac": float(np.mean([v > 0.5 for v in vals])),
            "n": len(vals),
        }
        for name, vals in results.items()
    }


def load_checkpoint_model(base_model, checkpoint, device):
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, checkpoint).to(device)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="locuslab/tofu_ft_phi-1.5")
    p.add_argument("--split", default="forget05")
    p.add_argument("--n_examples", type=int, default=60)
    p.add_argument("--out", required=True)

    p.add_argument("--checkpoint", default=None, help="Single-checkpoint mode")
    p.add_argument("--label", default="model")

    p.add_argument("--compare", action="store_true", help="Two-checkpoint ablation mode")
    p.add_argument("--checkpoint_a", default=None)
    p.add_argument("--label_a", default="A")
    p.add_argument("--checkpoint_b", default=None)
    p.add_argument("--label_b", default="B")

    p.add_argument("--multi_seed_compare", action="store_true",
                   help="Multi-seed ablation: --checkpoint_a/_b are templates with {seed}, "
                        "e.g. outputs/wk8/FORGE/forget05/seed{seed}/FORGE")
    p.add_argument("--seeds", default="42,123,456")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU required")
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)

    tok = AutoTokenizer.from_pretrained(TOK_MODEL)
    tok.pad_token = tok.eos_token

    examples = load_forget_examples(args.split, args.n_examples)
    print(f"Loaded {len(examples)} forget examples from TOFU {args.split}", flush=True)

    out_data = {"split": args.split, "n_examples": len(examples), "results": {}}

    if args.multi_seed_compare:
        assert args.checkpoint_a and args.checkpoint_b, "--multi_seed_compare requires --checkpoint_a/_b templates with {seed}"
        seeds = [int(s) for s in args.seeds.split(",")]
        per_seed = {args.label_a: {}, args.label_b: {}}
        for seed in seeds:
            for label, template in [(args.label_a, args.checkpoint_a), (args.label_b, args.checkpoint_b)]:
                ckpt = template.format(seed=seed)
                print(f"\n{'='*55}\nRobustness eval: {label} seed={seed} ({ckpt})\n{'='*55}", flush=True)
                model = load_checkpoint_model(args.base_model, ckpt, device)
                res = eval_robustness(model, tok, device, examples)
                per_seed[label][seed] = res
                for name, r in res.items():
                    print(f"  {name:>12}: rouge_leak={r['rouge_leak_mean']:.3f}  "
                          f"exact_leak={r['exact_leak_frac']:.3f}", flush=True)
                del model
                torch.cuda.empty_cache()

        # aggregate mean+std across seeds per perturbation condition
        agg = {}
        for label in [args.label_a, args.label_b]:
            agg[label] = {}
            for cond in PERTURBATIONS:
                rouge_vals = [per_seed[label][s][cond]["rouge_leak_mean"] for s in seeds]
                exact_vals = [per_seed[label][s][cond]["exact_leak_frac"] for s in seeds]
                agg[label][cond] = {
                    "rouge_leak_mean": float(np.mean(rouge_vals)),
                    "rouge_leak_std": float(np.std(rouge_vals)),
                    "exact_leak_mean": float(np.mean(exact_vals)),
                    "exact_leak_std": float(np.std(exact_vals)),
                    "n_seeds": len(seeds),
                    "per_seed_rouge_leak": rouge_vals,
                }
        out_data["seeds"] = seeds
        out_data["per_seed"] = per_seed
        out_data["results"] = agg
        print(f"\n{'='*55}\nMulti-seed delta ({args.label_a} - {args.label_b}), mean over {len(seeds)} seeds\n{'='*55}")
        for cond in PERTURBATIONS:
            d = agg[args.label_a][cond]["rouge_leak_mean"] - agg[args.label_b][cond]["rouge_leak_mean"]
            print(f"  {cond:>12}: delta_rouge_leak={d:+.3f} "
                  f"(std_a={agg[args.label_a][cond]['rouge_leak_std']:.3f}, "
                  f"std_b={agg[args.label_b][cond]['rouge_leak_std']:.3f})")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out_data, open(args.out, "w"), indent=2)
        print(f"\nSaved -> {args.out}")
        return

    checkpoints = []
    if args.compare:
        assert args.checkpoint_a and args.checkpoint_b, "--compare requires --checkpoint_a and --checkpoint_b"
        checkpoints = [(args.label_a, args.checkpoint_a), (args.label_b, args.checkpoint_b)]
    else:
        assert args.checkpoint, "provide --checkpoint (or use --compare with two checkpoints)"
        checkpoints = [(args.label, args.checkpoint)]

    for label, ckpt in checkpoints:
        print(f"\n{'='*55}\nRobustness eval: {label} ({ckpt})\n{'='*55}", flush=True)
        model = load_checkpoint_model(args.base_model, ckpt, device)
        res = eval_robustness(model, tok, device, examples)
        out_data["results"][label] = res
        for name, r in res.items():
            print(f"  {name:>12}: rouge_leak={r['rouge_leak_mean']:.3f}  "
                  f"exact_leak={r['exact_leak_frac']:.3f}  (n={r['n']})", flush=True)
        del model
        torch.cuda.empty_cache()

    if args.compare:
        a, b = out_data["results"][args.label_a], out_data["results"][args.label_b]
        print(f"\n{'='*55}\nDelta ({args.label_a} - {args.label_b}) per perturbation "
              f"(positive = A leaks MORE than B)\n{'='*55}")
        for name in PERTURBATIONS:
            d = a[name]["rouge_leak_mean"] - b[name]["rouge_leak_mean"]
            print(f"  {name:>12}: delta_rouge_leak={d:+.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_data, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
