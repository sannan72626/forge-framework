#!/usr/bin/env python
"""
scripts/run_sequential.py — Wk9 sequential unlearning stability.

Question: does unlearning a SECOND batch degrade the forgetting of the FIRST?

  Stage 1: unlearn forget01 (40 examples) → measure FQ on forget01
  Stage 2: continue from stage-1 model, unlearn forget05∖forget01 (the other 160)
           → re-measure FQ on forget01 (stage-1 set)

  ΔFQ(stage1) = FQ_after_stage1 − FQ_after_stage2
  Target: ΔFQ < 0.05  (stage-1 forgetting is retained)

Usage:
  python -u scripts/run_sequential.py --method FORGE --out_dir outputs/wk9/sequential_FORGE/
"""

import argparse, json, os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy import stats as sc
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

BASE_MODEL = "locuslab/tofu_ft_phi-1.5"
TOK_MODEL  = "microsoft/phi-1_5"
# retain95 oracle has seen NEITHER half of forget05 → valid FQ reference for both halves
ORACLE_F05 = "outputs/wk0/phi15_retain95"


class QADataset(Dataset):
    def __init__(self, examples, tok, max_len=128):
        self.ex=examples; self.tok=tok; self.ml=max_len
    def __len__(self): return len(self.ex)
    def __getitem__(self, i):
        e=self.ex[i]; t=f"Q: {e['question']}\nA: {e['answer']}"
        enc=self.tok(t,return_tensors="pt",truncation=True,max_length=self.ml,padding="max_length")
        ids=enc["input_ids"][0]; mask=enc["attention_mask"][0]
        lbl=ids.clone(); lbl[mask==0]=-100
        return {"input_ids":ids,"attention_mask":mask,"labels":lbl}

def collate(b): return {k:torch.stack([x[k] for x in b]) for k in b[0]}


# ── Loss functions (subset) ───────────────────────────────────────────────────

def loss_GA(m,fb,rb,dev): return -m(**{k:v.to(dev) for k,v in fb.items()}).loss

def loss_LoKU(m,fb,rb,dev,margin=4.0,lam=1.0):
    lf=-torch.clamp(m(**{k:v.to(dev) for k,v in fb.items()}).loss,max=margin)
    return lf+lam*m(**{k:v.to(dev) for k,v in rb.items()}).loss

def loss_FORGE(m,fb,rb,dev,lam=1.0,gamma=0.5,margin=4.0):
    nf=m(**{k:v.to(dev) for k,v in fb.items()}).loss
    lf=-torch.tanh(nf/margin)*margin
    lr_=m(**{k:v.to(dev) for k,v in rb.items()}).loss
    ids=fb["input_ids"].clone().to(dev); ids[torch.rand_like(ids.float())<0.20]=0
    nrob=m(input_ids=ids,attention_mask=fb["attention_mask"].to(dev),labels=fb["labels"].to(dev)).loss
    return lf+lam*lr_+gamma*(-torch.tanh(nrob/margin)*margin)

LOSSES={"GA":loss_GA,"LoKU":loss_LoKU,"FORGE":loss_FORGE}


@torch.no_grad()
def nll(model, tok, text, device):
    enc=tok(text,return_tensors="pt",truncation=True,max_length=128).to(device)
    return model(**enc,labels=enc["input_ids"]).loss.item()


def compute_fq(model, tok, forget_ex, oracle_nlls, device):
    texts=[f"Q: {e['question']}\nA: {e['answer']}" for e in forget_ex]
    model_nlls=[nll(model,tok,t,device) for t in texts]
    _,fq=sc.ks_2samp(model_nlls, oracle_nlls)
    return float(fq)


def train_stage(model, loss_fn, forget_ex, retain_ex, tok, device,
                n_steps=250, lr=2e-5, fq_eval=None):
    """Train one stage. If fq_eval=(eval_ex, oracle_nlls) given, track peak-FQ
    checkpoint and restore it at the end (FORGE FQ peaks then overshoots)."""
    from copy import deepcopy
    f_dl=DataLoader(QADataset(forget_ex,tok),batch_size=4,shuffle=True,collate_fn=collate)
    r_dl=DataLoader(QADataset(retain_ex,tok),batch_size=4,shuffle=True,collate_fn=collate)
    trainable=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(trainable,lr=lr)
    f_it=iter(f_dl); r_it=iter(r_dl)
    best_fq=-1.0; best_state=None
    for step in range(1,n_steps+1):
        model.train()
        try: fb=next(f_it)
        except: f_it=iter(f_dl); fb=next(f_it)
        try: rb=next(r_it)
        except: r_it=iter(r_dl); rb=next(r_it)
        loss=loss_fn(model,fb,rb,device)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable,1.0); opt.step()
        if fq_eval is not None and step % 25 == 0:
            eval_ex, oracle_nlls = fq_eval
            fq = compute_fq(model, tok, eval_ex, oracle_nlls, device)
            if fq > best_fq:
                best_fq = fq
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items() if "lora" in k.lower()}
    # Restore peak-FQ checkpoint
    if best_state is not None:
        sd = model.state_dict()
        for k, v in best_state.items():
            sd[k] = v.to(device)
        model.load_state_dict(sd, strict=False)
    return model, best_fq


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--method", default="FORGE", choices=list(LOSSES.keys()))
    p.add_argument("--out_dir", default="outputs/wk9/sequential_FORGE/")
    p.add_argument("--n_steps", type=int, default=250)
    args=p.parse_args()

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type!="cuda": raise RuntimeError("GPU required")
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)

    tok=AutoTokenizer.from_pretrained(TOK_MODEL)
    tok.pad_token=tok.eos_token; tok.padding_side="right"

    # Data: forget05 split into two disjoint halves (stage1=A, stage2=B), retain
    print("Loading TOFU splits...", flush=True)
    f05=list(load_dataset("locuslab/TOFU","forget05")["train"])
    retain=list(load_dataset("locuslab/TOFU","retain95")["train"])
    half=len(f05)//2
    stage1_ex=f05[:half]    # 100 examples — measured for stage-1 retention
    stage2_ex=f05[half:]    # 100 disjoint examples
    print(f"  stage1 (forget05 half-A): {len(stage1_ex)}  stage2 (half-B): {len(stage2_ex)}  retain: {len(retain)}", flush=True)

    # retain95 oracle has seen neither half → valid FQ reference for both halves
    print("Loading forget05 oracle (retain95)...", flush=True)
    base_o=AutoModelForCausalLM.from_pretrained(BASE_MODEL,torch_dtype=torch.bfloat16)
    oracle=PeftModel.from_pretrained(base_o,ORACLE_F05).merge_and_unload().to(device).eval()
    oracle_nlls  =[nll(oracle,tok,f"Q: {e['question']}\nA: {e['answer']}",device) for e in stage1_ex]
    oracle_nlls_B=[nll(oracle,tok,f"Q: {e['question']}\nA: {e['answer']}",device) for e in stage2_ex]
    del oracle,base_o; torch.cuda.empty_cache()
    print(f"  half-A oracle NLL mean={np.mean(oracle_nlls):.3f}  "
          f"half-B oracle NLL mean={np.mean(oracle_nlls_B):.3f}", flush=True)

    loss_fn=LOSSES[args.method]

    # ── Stage 1: unlearn forget05 half-A ──
    print(f"\n=== Stage 1: {args.method} unlearns forget05 half-A ===", flush=True)
    base=AutoModelForCausalLM.from_pretrained(BASE_MODEL,torch_dtype=torch.bfloat16)
    lora_cfg=LoraConfig(task_type=TaskType.CAUSAL_LM,r=32,lora_alpha=64,
                        lora_dropout=0.0,target_modules=["q_proj","v_proj"])
    model=get_peft_model(base,lora_cfg).to(device)
    # Track peak FQ on stage-1 set; restore peak checkpoint (FORGE FQ peaks then overshoots)
    model,fq_s1=train_stage(model,loss_fn,stage1_ex,retain,tok,device,args.n_steps,
                            fq_eval=(stage1_ex, oracle_nlls))
    print(f"  Peak FQ on half-A after stage 1: {fq_s1:.4f}", flush=True)

    # ── Stage 2: continue from peak checkpoint, unlearn forget05 half-B ──
    print(f"\n=== Stage 2: {args.method} unlearns forget05 half-B (from stage-1 peak) ===", flush=True)
    # Stage 2 stops at its own peak FQ on half-B (a well-tuned stage-2 unlearning)
    model,fq_s2=train_stage(model,loss_fn,stage2_ex,retain,tok,device,args.n_steps,
                            fq_eval=(stage2_ex, oracle_nlls_B))
    print(f"  Peak FQ on half-B after stage 2: {fq_s2:.4f}", flush=True)
    # Re-measure stage-1 (half-A) FQ: did stage-2 training degrade it?
    fq_s1_after=compute_fq(model,tok,stage1_ex,oracle_nlls,device)
    print(f"  FQ on half-A after stage 2: {fq_s1_after:.4f}", flush=True)

    delta=abs(fq_s1 - fq_s1_after)
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    result={"method":args.method,
            "fq_halfA_after_stage1":fq_s1,
            "fq_halfB_after_stage2":fq_s2,
            "fq_halfA_after_stage2":fq_s1_after,
            "delta_fq_stage1":delta,
            "pass":delta<0.05}
    json.dump(result,open(out/f"{args.method}_sequential.json","w"),indent=2)
    print(f"\n  ΔFQ(stage1) = |{fq_s1:.3f} − {fq_s1_after:.3f}| = {delta:.4f}", flush=True)
    print(f"  Gating (ΔFQ < 0.05): {'PASS' if delta<0.05 else 'WARN'}", flush=True)
    print(f"  Saved → {out/(args.method+'_sequential.json')}", flush=True)


if __name__=="__main__":
    main()
