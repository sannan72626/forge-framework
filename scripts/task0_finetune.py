#!/usr/bin/env python
"""
Week 6 cross-model fine-tuner — QLoRA fine-tune Phi-3-mini-4k-instruct on TOFU.

Used for Wk6 cross-model generalization experiments (Phi-3-mini as a smaller model
alongside Llama-2-7B). NOT needed for Wk4 main experiments — those use the
locuslab/tofu_ft_llama2-7b checkpoint directly.

Hardware: RTX 3090 23 GB (4-bit NF4 QLoRA still used for Phi-3 to leave headroom)
  rank=16, batch=1, grad_accum=8, max_length=128
  ~1-2 h per split

Usage:
  # Target model (fine-tune on everything)
  python scripts/task0_finetune.py --split full --out outputs/wk0/phi3_tofu_full/

  # Reference model (fine-tune on retain set only; needed for FQ eval)
  python scripts/task0_finetune.py --split retain95 --out outputs/wk0/phi3_tofu_retain95/

  # Run both before Wk6 cross-model experiments:
  #   python scripts/run_experiments.py --mode full --method FORGE --split forget05 \
  #       --base_model outputs/wk0/phi3_tofu_full/ \
  #       --ref_model  outputs/wk0/phi3_tofu_retain95/ \
  #       --out_dir outputs/wk6/cross_model/phi3/
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
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    raise ImportError("pip install peft")

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

BNB_CFG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

LORA_CFG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["qkv_proj", "o_proj"],
    bias="none",
)


class TOFUDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        txt = f"Q: {ex['question']}\nA: {ex['answer']}"
        enc = self.tokenizer(
            txt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        ids = enc["input_ids"][0]
        mask = enc["attention_mask"][0]
        lbl = ids.clone()
        lbl[mask == 0] = -100
        return {"input_ids": ids, "attention_mask": mask, "labels": lbl}


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def train(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Task 0 requires a CUDA GPU. No GPU detected.")
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory // 1024**3} GB)")

    print(f"Loading TOFU split: {args.split}")
    ds = load_dataset("locuslab/TOFU", args.split)["train"]
    examples = list(ds)
    print(f"  {len(examples)} examples")

    print(f"Loading tokenizer from {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading model (4-bit NF4 QLoRA)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=BNB_CFG,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LORA_CFG)
    model.print_trainable_parameters()

    dataset = TOFUDataset(examples, tokenizer, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=0)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    steps_per_epoch = math.ceil(len(dataset) / args.batch_size / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, total_steps // 10)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\nFine-tuning: split={args.split}, epochs={args.epochs}, "
          f"total_steps≈{total_steps}, lr={args.lr}")
    print(f"  batch={args.batch_size}, grad_accum={args.grad_accum}, "
          f"max_len={args.max_length}, rank=16")

    log = []
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_micro = 0

        for i, batch in enumerate(loader):
            out_b = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = out_b.loss / args.grad_accum
            loss.backward()
            epoch_loss += out_b.loss.item()
            n_micro += 1

            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 100 == 0:
                    cur_lr = scheduler.get_last_lr()[0]
                    print(f"  epoch {epoch+1}/{args.epochs}  step {global_step}  "
                          f"loss={out_b.loss.item():.4f}  lr={cur_lr:.2e}")

        avg_loss = epoch_loss / max(n_micro, 1)
        print(f"Epoch {epoch+1}/{args.epochs} — avg_loss={avg_loss:.4f}")
        log.append({"epoch": epoch + 1, "avg_loss": avg_loss})

    # Flush any remaining gradient accumulation
    if n_micro % args.grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad()

    print(f"\nSaving adapter to {out}")
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)

    summary = {
        "model": MODEL_NAME,
        "split": args.split,
        "epochs": args.epochs,
        "lr": args.lr,
        "rank": 16,
        "final_loss": log[-1]["avg_loss"] if log else None,
        "log": log,
    }
    with open(out / "train_log.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved train_log.json — final_loss={summary['final_loss']:.4f}")
    print(f"\nDone. Checkpoint: {out.resolve()}")


def main():
    p = argparse.ArgumentParser(description="QLoRA fine-tune Phi-3-mini on TOFU")
    p.add_argument("--split", default="full",
                   choices=["full", "retain99", "retain95", "retain90"],
                   help="TOFU split to train on")
    p.add_argument("--out", default="outputs/wk0/phi3_tofu_full/",
                   help="Output directory for the LoRA adapter")
    p.add_argument("--epochs",     type=int,   default=5)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--batch_size", type=int,   default=1)
    p.add_argument("--grad_accum", type=int,   default=8,
                   help="Gradient accumulation steps (effective batch = batch_size × grad_accum)")
    p.add_argument("--max_length", type=int,   default=128)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
