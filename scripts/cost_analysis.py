#!/usr/bin/env python
"""
scripts/cost_analysis.py — Wk9 computational cost comparison.

Measures per-method: trainable params, peak VRAM, wall-clock per step, FLOPs estimate.
Compares against a full fine-tuning retrain baseline.

Usage:
  python -u scripts/cost_analysis.py --out outputs/wk9/cost_analysis.json
"""

import argparse, json, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

BASE_MODEL = "locuslab/tofu_ft_phi-1.5"
TOK_MODEL  = "microsoft/phi-1_5"


def measure_method(use_lora, rank, tok, device, n_warmup=3, n_measure=10, batch=4, seqlen=128):
    """Measure VRAM + wall-clock for one configuration."""
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16).to(device)
    if use_lora:
        cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=rank*2,
                         lora_dropout=0.0, target_modules=["q_proj","v_proj"])
        model = get_peft_model(model, cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())

    # Dummy batch
    ids  = torch.randint(0, 50000, (batch, seqlen), device=device)
    mask = torch.ones_like(ids)
    lbl  = ids.clone()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)

    # Warmup
    model.train()
    for _ in range(n_warmup):
        out = model(input_ids=ids, attention_mask=mask, labels=lbl)
        opt.zero_grad(); out.loss.backward(); opt.step()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(n_measure):
        out = model(input_ids=ids, attention_mask=mask, labels=lbl)
        opt.zero_grad(); out.loss.backward(); opt.step()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    peak_gb   = torch.cuda.max_memory_allocated(device) / 1e9
    step_sec  = (t1 - t0) / n_measure

    del model, opt; torch.cuda.empty_cache()
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 4),
        "peak_vram_gb": round(peak_gb, 2),
        "sec_per_step": round(step_sec, 4),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/wk9/cost_analysis.json")
    p.add_argument("--steps", type=int, default=250, help="steps used in actual unlearning runs")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("GPU required")
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)

    tok = AutoTokenizer.from_pretrained(TOK_MODEL)
    tok.pad_token = tok.eos_token

    results = {}

    # LoRA unlearning (FORGE/LoKU/etc all use same rank-32 LoRA)
    print("Measuring LoRA rank=32 unlearning cost...", flush=True)
    results["forge_lora_r32"] = measure_method(True, 32, tok, device)
    print(f"  {results['forge_lora_r32']}", flush=True)

    # Full fine-tuning retrain (all params trainable)
    print("Measuring full fine-tune retrain cost...", flush=True)
    results["full_retrain"] = measure_method(False, 0, tok, device)
    print(f"  {results['full_retrain']}", flush=True)

    # Derived comparisons
    lora = results["forge_lora_r32"]
    full = results["full_retrain"]

    # Unlearning: 250 steps. Full retrain TOFU: ~5 epochs × 4000 ex / batch 4 = 5000 steps
    unlearn_steps = args.steps
    retrain_steps = 5 * 4000 // 4  # 5 epochs, 4000 examples, batch 4

    lora_wall = lora["sec_per_step"] * unlearn_steps
    full_wall = full["sec_per_step"] * retrain_steps

    results["comparison"] = {
        "unlearn_steps": unlearn_steps,
        "retrain_steps": retrain_steps,
        "lora_wallclock_sec": round(lora_wall, 1),
        "full_retrain_wallclock_sec": round(full_wall, 1),
        "speedup": round(full_wall / lora_wall, 1),
        "trainable_param_ratio": round(lora["trainable_params"] / full["trainable_params"], 5),
        "vram_ratio": round(lora["peak_vram_gb"] / full["peak_vram_gb"], 3),
    }

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)

    print("\n" + "="*60)
    print("Cost Analysis Summary")
    print("="*60)
    c = results["comparison"]
    print(f"  Trainable params:  LoRA {lora['trainable_params']:,} "
          f"({lora['trainable_pct']}%) vs Full {full['trainable_params']:,}")
    print(f"  Peak VRAM:         LoRA {lora['peak_vram_gb']} GB vs Full {full['peak_vram_gb']} GB")
    print(f"  Sec/step:          LoRA {lora['sec_per_step']} vs Full {full['sec_per_step']}")
    print(f"  Wall-clock:        LoRA {c['lora_wallclock_sec']}s ({unlearn_steps} steps) vs "
          f"Full {c['full_retrain_wallclock_sec']}s ({retrain_steps} steps)")
    print(f"  → Speedup:         {c['speedup']}× faster than full retrain")
    print(f"  → Param ratio:     {c['trainable_param_ratio']*100:.3f}% of full params trained")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
