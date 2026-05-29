# FORGE Experimental Protocol (Phi-3-mini-4B · RTX 3050 8 GB)

## Model
- Unlearning target: outputs/wk0/phi3_tofu_full  (Phi-3-mini TOFU fine-tuned, 4-bit)
- Retain reference: same checkpoint frozen
- Alternative: locuslab/tofu_ft_llama2-7b with 4-bit quant (Llama-2-7B)

## Datasets
- TOFU: forget-1%, forget-5%, forget-10% (locuslab/TOFU on HuggingFace)
- WMDP-bio: MCQ biosecurity subset (cais/wmdp)
- MUSE-News: copyright news corpus (jaechan-repo/muse-bench)

## Metrics (5 per run)
1. FQ — Forget Quality (KS-test p-value; higher = better)
2. MU — Model Utility (retain ROUGE-L + perplexity; higher = better)
3. ROUGE-L forget (lower = better)
4. MIA-TR — Membership Inference Truth Ratio (target ≈ 1.0)
5. MIA-PR — Membership Inference Probability Ratio (target ≈ 1.0)

## Training protocol (8 GB constraints)
- Seeds: 3 per (method, split): 42, 123, 456
- Steps: 200 (reduced from 300 for memory budget)
- Batch: 1, gradient_accumulation_steps: 4 (effective batch 4)
- max_length: 128 tokens
- LoRA rank: 16 (not 32 — OOM on 8 GB)
- Config: configs/forge_default.yaml (lr=5e-6, rank=16, ep=10)
- Eval: every 40 steps; checkpoint saved at each eval

## Statistical validation (Week 8)
- Paired t-test FORGE vs each baseline; α=0.05
- 95% bootstrap CI (1000 resamples) on all metrics
- Report mean ± CI; mark p<0.05 with †, p<0.01 with ‡

## Trajectory visualisation
- X-axis: FQ, Y-axis: MU
- Each method = one curve from shared origin (pre-unlearn)
- Direction arrows at midpoint; endpoint labelled
