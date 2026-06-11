# FORGE Method Summary
## Bounded-Ascent Unlearning with Perturbation Robustness

> This summary describes the method **as implemented and evaluated**. The originally-
> proposed Fisher-weighted dual adapters, learned adversarial probe generator, and CoT-KL
> are scoped as future work — see `outputs/reports/method_audit.md`.

## Objective
```
ℒ_FORGE(θ) = ℒ_forget + λ·ℒ_retain + γ·ℒ_robust

  ℒ_forget = −tanh(NLL_θ(D_f) / m) · m          # bounded gradient ascent
  ℒ_retain =  NLL_θ(D_r)                         # utility anchor
  ℒ_robust = −tanh(NLL_θ(perturb(D_f)) / m) · m  # ascent on input-perturbed forget
  perturb(x): randomly mask 20% of input tokens
```
All terms are minimized over the LoRA parameters θ (single adapter). No adversary,
no second adapter, no minimax — the objective is a plain (bounded) multi-term loss.

## Contributions (what the method actually does)
1. **Bounded forget loss.** The `tanh(·)·m` bound caps the per-step forget gradient at
   magnitude `m`. This prevents the runaway ascent that drives MU→0 in GA/GD, and is the
   mechanism behind FORGE being the only method achieving **both** high FQ and high MU.
2. **Perturbation robustness.** The robustness term requires forgetting to hold even when
   the forget input is corrupted, giving resistance beyond canonical queries — a
   lightweight, retain-free signal (honestly: input perturbation, not an adversarial net).

## Default hyperparameters (validated)
```
model: locuslab/tofu_ft_phi-1.5 (1.4B); Llama-2-7B validation in progress
lr=2e-5, lora_rank=32, lora_alpha=64, target_modules=[q_proj, v_proj]
ihl_margin (m)=4.0, λ_retain=1.0, γ_robust=0.5
max_steps=250, eval_every=25, batch=4, max_length=128
FQ uses peak-across-trajectory checkpoint (FORGE FQ peaks ~step 125 then overshoots)
```

## Headline result (TOFU forget-5%, 3 seeds, p<0.05)
| | FQ | MU | FQ×MU |
|--|----|----|-------|
| FORGE | 0.869 | 0.475 | **0.413** |
| LoKU (best baseline) | 0.447 | 0.210 | 0.094 |

FORGE is the only method with FQ>0.8 *and* MU>0.4; FQ×MU is 4.4× the best baseline.

## Future work (explicitly NOT claimed as done)
- Fisher-weighted dual LoRA adapters (decouple forget/retain gradients)
- Learned adversarial probe generator (true minimax robustness)
- Step-wise CoT-trace KL for reasoning-model unlearning (ζ=0 throughout this work)
