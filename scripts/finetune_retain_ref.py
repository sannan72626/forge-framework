#!/usr/bin/env python
"""
scripts/finetune_retain_ref.py — Fine-tune the retain-only oracle reference model.

locuslab does NOT provide a Llama-2-7B retain-only checkpoint, so we build it by
continued fine-tuning of locuslab/tofu_ft_llama2-7b on the retain split only.
The result is used as the oracle for Forget Quality (FQ = KS test p-value).

Usage:
  # Build oracle for forget-5% experiments (retain95 split, ~1 h on RTX 3090):
  python scripts/finetune_retain_ref.py --split retain95 \
      --out outputs/wk0/llama2_tofu_retain95/

  # For forget-1% (retain99):
  python scripts/finetune_retain_ref.py --split retain99 \
      --out outputs/wk0/llama2_tofu_retain99/

  # For forget-10% (retain90):
  python scripts/finetune_retain_ref.py --split retain90 \
      --out outputs/wk0/llama2_tofu_retain90/

Then pass to run_experiments.py:
  python scripts/run_experiments.py --mode full --method FORGE --split forget05 \
      --ref_model outputs/wk0/llama2_tofu_retain95/ ...
"""

import argparse
import json
import math
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model
    HAS_PEFT = True
except ImportError:
    raise ImportError("pip install peft")

BASE_MODEL = "microsoft/phi-1_5"  # public base; no HF token needed

# bfloat16 — oracle fine-tunes only ONE model, so 14 GB fits with 9 GB headroom.
# No quantization needed; avoids prepare_model_for_kbit_training complications.
LORA_CFG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,
    lora_alpha=64,
    lora_dropout=0.0,
    target_modules=["q_proj", "v_proj"],  # Phi-1.5 separate Q/V projections
    bias="none",
)


class TOFUDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        txt = f"Q: {ex['question']}\nA: {ex['answer']}"
        enc = self.tokenizer(
            txt, return_tensors="pt", truncation=True,
            max_length=self.max_length, padding="max_length",
        )
        ids  = enc["input_ids"][0]
        mask = enc["attention_mask"][0]
        lbl  = ids.clone(); lbl[mask == 0] = -100
        return {"input_ids": ids, "attention_mask": mask, "labels": lbl}


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required.")
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory // 1024**3} GB)")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print(f"Loading TOFU split: {args.split}", flush=True)
    ds = load_dataset("locuslab/TOFU", args.split)["train"]
    examples = list(ds)
    print(f"  {len(examples)} examples (retain-only oracle)", flush=True)

    base_model = args.base_model
    print(f"Loading tokenizer from {base_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading {base_model} (bfloat16, single model)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16
    ).to(device)
    model = get_peft_model(model, LORA_CFG)
    model.print_trainable_parameters()

    loader = DataLoader(
        TOFUDataset(examples, tokenizer, args.max_length),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    steps_per_epoch = math.ceil(len(examples) / args.batch_size / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, total_steps // 10), total_steps
    )

    print(f"\nFine-tuning on {args.split}: epochs={args.epochs}, "
          f"steps≈{total_steps}, lr={args.lr}, rank=32", flush=True)

    log = []
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0; n_micro = 0

        for i, batch in enumerate(loader):
            out_b = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            (out_b.loss / args.grad_accum).backward()
            epoch_loss += out_b.loss.item(); n_micro += 1

            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                global_step += 1
                if global_step % 100 == 0:
                    print(f"  epoch {epoch+1}/{args.epochs}  step {global_step}  "
                          f"loss={out_b.loss.item():.4f}  "
                          f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

        avg_loss = epoch_loss / max(n_micro, 1)
        print(f"Epoch {epoch+1}/{args.epochs} — avg_loss={avg_loss:.4f}", flush=True)
        log.append({"epoch": epoch + 1, "avg_loss": avg_loss})

    if args.save_merged:
        print(f"\nMerging LoRA into base and saving full model to {out}", flush=True)
        merged = model.merge_and_unload()
        merged.save_pretrained(out)
    else:
        print(f"\nSaving oracle adapter to {out}", flush=True)
        model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    json.dump({"base": base_model, "split": args.split, "merged": args.save_merged, "log": log},
              open(out / "train_log.json", "w"), indent=2)
    print(f"Done. Oracle checkpoint: {out.resolve()}")


def main():
    p = argparse.ArgumentParser(description="Fine-tune retain-only oracle for FQ eval")
    p.add_argument("--base_model", default="microsoft/phi-1_5",
                   help="Base model to fine-tune (e.g. NousResearch/Llama-2-7b-hf)")
    p.add_argument("--split",      default="retain95",
                   choices=["retain90", "retain95", "retain99"],
                   help="TOFU retain split to train on")
    p.add_argument("--out",        default="outputs/wk0/llama2_tofu_retain95/")
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--lr",         type=float, default=5e-6)
    p.add_argument("--batch_size", type=int,   default=2)
    p.add_argument("--grad_accum", type=int,   default=2)
    p.add_argument("--max_length", type=int,   default=256)
    p.add_argument("--save_merged", action="store_true",
                   help="Merge LoRA into base and save full model (for 4-bit runner loading)")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
