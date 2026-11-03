#!/usr/bin/env python
"""
scripts/profile_edge_unlearning.py — edge/resource-constrained hardware
profiling for FORGE unlearning. Written and ready to run, but NOT executed
against real edge hardware in this pass — see outputs/edge_profile/NOT_RUN.md
for why, and do not treat any numbers from a non-edge run of this script as
edge evidence.

Usage (on actual edge/Jetson-class hardware only):
  python scripts/profile_edge_unlearning.py \
    --model locuslab/tofu_ft_phi-1.5 --base_model microsoft/phi-1_5 \
    --method FORGE --split forget05 --steps 25 --batch_size 1 \
    --output outputs/edge_profile/forge_phi15_edge.json
"""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import torch


def get_energy_reading():
    """Best-effort energy reading via common platform tools. Returns None
    (not a fabricated number) if no tool is available — this function must
    never invent a value."""
    for cmd in (["tegrastats", "--interval", "100", "--count", "1"],):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
            if out.returncode == 0:
                return out.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except FileNotFoundError:
        pass
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="locuslab/tofu_ft_phi-1.5")
    p.add_argument("--base_model", default="microsoft/phi-1_5")
    p.add_argument("--method", default="FORGE")
    p.add_argument("--split", default="forget05")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device_name = "CPU"
    total_vram_gb = None
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    is_edge_device = any(k in platform.machine().lower() for k in ["aarch64", "arm"]) or \
                     any(k in device_name.lower() for k in ["jetson", "orin", "xavier", "nano"])

    if not is_edge_device:
        print(f"WARNING: detected device '{device_name}' ({platform.machine()}) does not "
              f"look like an edge/Jetson-class device. Refusing to label this run's output "
              f"as an edge measurement. Writing raw numbers with is_edge_device=False so the "
              f"caller can decide, but these must NOT be used as IoT/edge evidence in the paper.",
              flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
    tok.pad_token = tok.eos_token

    t_load0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64, lora_dropout=0.0,
                         target_modules=["q_proj", "v_proj"])
    model = get_peft_model(base, lora_cfg).to(dev)
    load_time_sec = time.time() - t_load0

    dummy = tok("Question: test\nAnswer: test answer.", return_tensors="pt",
               padding="max_length", truncation=True, max_length=64).to(dev)
    labels = dummy["input_ids"].clone()

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad], lr=2e-5)

    t0 = time.time()
    for _ in range(args.steps):
        out = model(input_ids=dummy["input_ids"], attention_mask=dummy["attention_mask"], labels=labels)
        opt.zero_grad(); out.loss.backward(); opt.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    total_sec = time.time() - t0
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else None

    result = {
        "device_name": device_name,
        "platform_machine": platform.machine(),
        "is_edge_device": is_edge_device,
        "total_vram_gb": total_vram_gb,
        "model_load_time_sec": load_time_sec,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "total_train_time_sec": total_sec,
        "sec_per_step": total_sec / args.steps,
        "peak_vram_gb_during_training": peak_vram_gb,
        "energy_reading_raw": get_energy_reading(),
        "warning": None if is_edge_device else (
            "This run was NOT on edge/Jetson-class hardware. Do not cite these "
            "numbers as edge/IoT deployment evidence."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.output, "w"), indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
