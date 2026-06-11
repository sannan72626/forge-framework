#!/usr/bin/env python
"""
scripts/run_muse.py — Wk7 MUSE-News unlearning experiments.

Two-step process:
  Step 1: Fine-tune Phi-1.5 on MUSE-News forget texts (memorise them)
  Step 2: Apply FORGE/baselines to unlearn those texts
  Eval  : VerbMem (5-gram overlap) + KnowMem (ROUGE-L on QA pairs)

Usage:
  python -u scripts/run_muse.py --methods GA,FLAT,LoKU,FORGE \
      --config configs/forge_default.yaml --out_dir outputs/wk7/muse/
"""

import argparse, json, os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

BASE_MODEL = "locuslab/tofu_ft_phi-1.5"
TOK_MODEL  = "microsoft/phi-1_5"


# ── Data ──────────────────────────────────────────────────────────────────────

def load_muse_data(n_forget=50, n_retain=50):
    try:
        from datasets import load_dataset
        ds = load_dataset("jaechan-repo/muse-bench", "news", trust_remote_code=True)
        forget_texts = [e["text"] for e in list(ds["forget"])[:n_forget]]
        forget_qa    = [(e.get("question","Who?"), e.get("answer","Unknown"))
                        for e in list(ds["forget"])[:n_forget]]
        retain_texts = [e["text"] for e in list(ds["retain"])[:n_retain]]
    except Exception as e:
        print(f"  MUSE dataset load failed ({e}), using synthetic fallback", flush=True)
        forget_texts = [f"The news article about event {i} contains details about incident {i} "
                        f"which occurred in city {i} affecting {i*10} people. " * 10
                        for i in range(n_forget)]
        forget_qa    = [(f"What happened in event {i}?", f"Incident {i} occurred in city {i}.")
                        for i in range(n_forget)]
        retain_texts = [f"General information about topic {i} and its implications for society. " * 10
                        for i in range(n_retain)]
    return forget_texts, forget_qa, retain_texts


class TextDataset(Dataset):
    def __init__(self, texts, tok, max_len=256):
        self.texts=texts; self.tok=tok; self.ml=max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], return_tensors="pt", truncation=True,
                       max_length=self.ml, padding="max_length")
        ids=enc["input_ids"][0]; mask=enc["attention_mask"][0]
        lbl=ids.clone(); lbl[mask==0]=-100
        return {"input_ids":ids,"attention_mask":mask,"labels":lbl}

def collate(b): return {k:torch.stack([x[k] for x in b]) for k in b[0]}


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def verbatim_mem(model, tok, src_texts, device, ngram=5, max_gen=100):
    def ngrams(t,n):
        w=t.lower().split(); return set(zip(*[w[i:] for i in range(n)]))
    src_ng = set()
    for t in src_texts: src_ng.update(ngrams(t, ngram))
    scores = []
    model.eval()
    for txt in src_texts[:30]:
        enc = tok(txt[:150], return_tensors="pt", max_length=64,
                  truncation=True).to(device)
        out = model.generate(**enc, max_new_tokens=max_gen, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        gen_ng = ngrams(gen, ngram)
        scores.append(len(gen_ng & src_ng) / max(len(gen_ng),1))
    return float(np.mean(scores)) if scores else 0.0


@torch.no_grad()
def knowledge_mem(model, tok, qa_pairs, device, max_samples=30):
    from rouge_score import rouge_scorer as rl
    scorer = rl.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    model.eval()
    for q, ref_a in list(qa_pairs)[:max_samples]:
        enc = tok(f"Q: {q}\nA:", return_tensors="pt", max_length=128, truncation=True).to(device)
        out = model.generate(**enc, max_new_tokens=30, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        scores.append(scorer.score(ref_a, gen)["rougeL"].fmeasure)
    return float(np.mean(scores)) if scores else 0.0


# ── Loss functions (same as WMDP) ─────────────────────────────────────────────

def loss_GA(m,_,fb,rb,dev): return -m(**{k:v.to(dev) for k,v in fb.items()}).loss

def loss_FLAT(m,_,fb,rb,dev,beta=2.0):
    out=m(input_ids=fb["input_ids"].to(dev),attention_mask=fb["attention_mask"].to(dev))
    t=fb["labels"].to(dev)[...,1:]; mask=(t!=-100).float(); tc=t.masked_fill(t==-100,0)
    lp=F.log_softmax(out.logits[...,:-1,:],-1).gather(-1,tc.unsqueeze(-1)).squeeze(-1)
    return (-torch.log(1+beta*lp.exp()+1e-8)*mask).sum()/(mask.sum()+1e-8)

def loss_LoKU(m,_,fb,rb,dev,margin=4.0,lam=1.0):
    lf=-torch.clamp(m(**{k:v.to(dev) for k,v in fb.items()}).loss,max=margin)
    lr_=m(**{k:v.to(dev) for k,v in rb.items()}).loss
    return lf+lam*lr_

def loss_FORGE(m,_,fb,rb,dev,lam=1.0,gamma=0.5,margin=4.0):
    nf=m(**{k:v.to(dev) for k,v in fb.items()}).loss
    lf=-torch.tanh(nf/margin)*margin
    lr_=m(**{k:v.to(dev) for k,v in rb.items()}).loss
    ids_rob=fb["input_ids"].clone().to(dev)
    ids_rob[torch.rand_like(ids_rob.float())<0.20]=0
    nrob=m(input_ids=ids_rob,attention_mask=fb["attention_mask"].to(dev),
            labels=fb["labels"].to(dev)).loss
    lrob=-torch.tanh(nrob/margin)*margin
    return lf+lam*lr_+gamma*lrob

METHOD_LOSSES={"GA":loss_GA,"FLAT":loss_FLAT,"LoKU":loss_LoKU,"FORGE":loss_FORGE}


# ── Fine-tune on MUSE forget texts (memorise) ─────────────────────────────────

def finetune_on_muse(forget_texts, retain_texts, tok, device, out_path, cfg):
    if (out_path/"adapter_config.json").exists():
        print(f"  Memorised checkpoint exists at {out_path}", flush=True)
        return
    print("Step 1: Fine-tuning on MUSE forget texts to increase memorisation...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
                          lora_dropout=0.0, target_modules=["q_proj","v_proj"])
    model = get_peft_model(base, lora_cfg).to(device)
    all_texts = forget_texts + retain_texts
    dl = DataLoader(TextDataset(all_texts, tok, 256), batch_size=2,
                    shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
    for ep in range(3):
        losses=[]
        for batch in dl:
            model.train()
            out=model(**{k:v.to(device) for k,v in batch.items()})
            opt.zero_grad(); out.loss.backward(); opt.step()
            losses.append(out.loss.item())
        print(f"  Memorise epoch {ep+1}/3  loss={np.mean(losses):.4f}", flush=True)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_path)
    tok.save_pretrained(out_path)
    print(f"  Memorised model saved → {out_path}", flush=True)


# ── Run one unlearning method ─────────────────────────────────────────────────

def run_method(method, forget_texts, forget_qa, retain_texts,
               memorised_path, tok, device, cfg, out_dir):
    print(f"\n{'='*55}\nMUSE {method}\n{'='*55}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, memorised_path).merge_and_unload().to(device)
    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=int(cfg.get("lora_rank",32)),
                          lora_alpha=64, lora_dropout=0.0, target_modules=["q_proj","v_proj"])
    from peft import get_peft_model as gpm
    model = gpm(model, lora_cfg)

    f_dl = DataLoader(TextDataset(forget_texts, tok, int(cfg.get("max_length",128))),
                      batch_size=int(cfg.get("per_device_batch",4)), shuffle=True, collate_fn=collate)
    r_dl = DataLoader(TextDataset(retain_texts, tok, int(cfg.get("max_length",128))),
                      batch_size=int(cfg.get("per_device_batch",4)), shuffle=True, collate_fn=collate)
    trainable=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable, lr=float(cfg.get("lr",2e-5)))
    n_steps=int(cfg.get("max_steps",200)); eval_every=int(cfg.get("eval_every",50))

    loss_fn=METHOD_LOSSES[method]
    f_it=iter(f_dl); r_it=iter(r_dl)

    # Baseline
    vm0=verbatim_mem(model,tok,forget_texts,device)
    km0=knowledge_mem(model,tok,forget_qa,device)
    print(f"  step    0  VerbMem={vm0:.3f}  KnowMem={km0:.3f}", flush=True)

    for step in range(1, n_steps+1):
        model.train()
        try: fb=next(f_it)
        except: f_it=iter(f_dl); fb=next(f_it)
        try: rb=next(r_it)
        except: r_it=iter(r_dl); rb=next(r_it)
        loss=loss_fn(model,None,fb,rb,device)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable,1.0); optimizer.step()
        if step%eval_every==0:
            vm=verbatim_mem(model,tok,forget_texts,device)
            km=knowledge_mem(model,tok,forget_qa,device)
            print(f"  step {step:>4}  VerbMem={vm:.3f}  KnowMem={km:.3f}", flush=True)

    vm_f=verbatim_mem(model,tok,forget_texts,device)
    km_f=knowledge_mem(model,tok,forget_qa,device)
    out_dir.mkdir(parents=True, exist_ok=True)
    result={"method":method,"verbmem":vm_f,"knowmem":km_f,
            "pass_verbmem":vm_f<=0.08,"pass_knowmem":km_f<=0.08}
    json.dump(result, open(out_dir/"muse_result.json","w"), indent=2)
    print(f"  Final: VerbMem={vm_f:.3f} ({'PASS' if vm_f<=0.08 else 'WARN'})  "
          f"KnowMem={km_f:.3f} ({'PASS' if km_f<=0.08 else 'WARN'})", flush=True)
    return vm_f, km_f


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods",  default="GA,FLAT,LoKU,FORGE")
    p.add_argument("--config",   default="configs/forge_default.yaml")
    p.add_argument("--out_dir",  default="outputs/wk7/muse")
    p.add_argument("--n_forget", type=int, default=50)
    p.add_argument("--n_retain", type=int, default=50)
    args = p.parse_args()

    cfg = {}
    if os.path.exists(args.config):
        try:
            import yaml; cfg = yaml.safe_load(open(args.config)) or {}
        except ImportError:
            for line in open(args.config):
                line=line.strip()
                if ":" in line and not line.startswith("#"):
                    k,_,v=line.partition(":"); v=v.strip()
                    try: cfg[k.strip()]=float(v) if "."in v or "e"in v.lower() else int(v)
                    except: cfg[k.strip()]=v

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("GPU required")
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)

    tok = AutoTokenizer.from_pretrained(TOK_MODEL)
    tok.pad_token = tok.eos_token; tok.padding_side = "right"

    print("Loading MUSE-News data...", flush=True)
    forget_texts, forget_qa, retain_texts = load_muse_data(args.n_forget, args.n_retain)
    print(f"  {len(forget_texts)} forget texts, {len(retain_texts)} retain texts")

    out = Path(args.out_dir)
    memorised_path = out / "memorised_checkpoint"
    finetune_on_muse(forget_texts, retain_texts, tok, device, memorised_path, cfg)

    methods = [m.strip() for m in args.methods.split(",")]
    summary = {}
    for method in methods:
        if method not in METHOD_LOSSES: continue
        vm, km = run_method(method, forget_texts, forget_qa, retain_texts,
                            memorised_path, tok, device, cfg, out/method)
        summary[method] = {"verbmem": vm, "knowmem": km,
                           "pass": vm<=0.08 and km<=0.08}

    json.dump(summary, open(out/"muse_summary.json","w"), indent=2)
    print("\n" + "="*55 + "\nMUSE Summary\n" + "="*55)
    print(f"{'Method':>8}  {'VerbMem':>8}  {'KnowMem':>8}  {'Pass':>6}")
    for m,r in summary.items():
        print(f"{m:>8}  {r['verbmem']:>8.3f}  {r['knowmem']:>8.3f}  "
              f"{'PASS' if r['pass'] else 'WARN':>6}")
    print(f"\nSaved → {out/'muse_summary.json'}")

if __name__ == "__main__":
    main()
