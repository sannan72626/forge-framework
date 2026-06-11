# Method Audit — Claims vs Implementation
## FORGE · reconciliation for publication
**Date:** 2026-06-02

This document records the exact gap between the originally-claimed method (Wk1–3 design
docs) and the implementation that produced all experimental results (Wk4–9, and the
Llama-2-7B run). It then defines the **canonical, honest method specification** that the
paper will describe — the objective that actually produced the results.

---

## 1. What was originally claimed (Wk3 method_summary.md)

```
ℒ(θ,g,ψ) = ℒ_forget + λ·ℒ_retain + γ·ℒ_robust(ψ) + ζ·ℒ_reason + η·ℒ_gate
           min over (θ,g);  max over ψ
```
- **Gap A:** dual LoRA adapters (Δ_f, Δ_r) + Fisher-weighted merge + token gate g_φ(h)
- **Gap B:** adversarial inner loop; probe generator Φ_ψ updated k=2 steps (minimax)
- **Gap C:** step-wise CoT KL (ζ=0 TOFU, 0.2 R-TOFU)

## 2. What was actually implemented and run

`scripts/run_experiments.py :: loss_FORGE` (the version that produced every result):
```
ℒ_FORGE = ℒ_forget + λ·ℒ_retain + γ·ℒ_robust
  ℒ_forget = −tanh(NLL_f / m) · m            # bounded gradient ascent (IHL-style)
  ℒ_retain = NLL(model, retain)              # utility anchor
  ℒ_robust = −tanh(NLL(perturb(forget)) / m) · m   # ascent on input-perturbed forget
  perturb(x) = randomly zero 20% of input tokens
  m = 4.0 (ihl_margin), λ = 1.0, γ = 0.5
```

### Discrepancies found
| Claimed | Status | Action |
|---------|--------|--------|
| Dual adapters + Fisher merge | **not implemented** (single adapter) | **drop claim** — single adapter is what works |
| Token gate g_φ (η·ℒ_gate) | **dead code** (computed under `no_grad`) | **remove** from code + claims |
| Adversary Φ_ψ minimax (γ·ℒ_robust) | **not a minimax** — random token drop | **reframe** as input-perturbation robustness (honest) |
| CoT KL (ζ·ℒ_reason) | **absent** | **mark explicit future work** (ζ=0 on TOFU anyway) |
| `trainer.py` l_rob sign | **bug** (minimized, not ascended) | **fix** to match validated runner |

---

## 3. Canonical method specification (what the paper will claim)

**FORGE = Bounded-ascent unlearning with perturbation robustness.**

```
ℒ_FORGE(θ) = ℒ_forget + λ·ℒ_retain + γ·ℒ_robust
  ℒ_forget = −tanh(NLL_θ(D_f) / m) · m
  ℒ_retain =  NLL_θ(D_r)
  ℒ_robust = −tanh(NLL_θ(perturb(D_f)) / m) · m
```

**Two honest, defensible contributions:**
1. **Bounded forget loss.** The `tanh`-bounded ascent caps the per-step forget gradient,
   preventing the runaway divergence that makes GA/GD collapse utility (MU→0). This is the
   mechanism behind FORGE being the *only* method with both high FQ and high MU
   (FQ×MU = 0.413 vs 0.094 best baseline, p<0.05).
2. **Perturbation robustness.** The robustness term forces forgetting to hold under input
   corruption, not just on canonical queries — a lightweight, retain-free robustness signal.
   (Honestly described as input-perturbation, NOT an adversarial T5 minimax.)

**Explicitly scoped as future work (not claimed as done):**
- Dual Fisher-weighted adapters (Gap A original ambition)
- Learned adversarial probe generator (Gap B original ambition)
- CoT-trace KL for reasoning models (Gap C; ζ=0 throughout this work)

---

## 4. Other reviewer concerns + remediation status

| Concern | Status | Remediation |
|---------|--------|-------------|
| Model scale (1.4B only) | **being fixed** | Llama-2-7B grid running (6 methods × 3 seeds, forget05) |
| Method ≠ code | **fixed here** | Code + claims reconciled (this document) |
| Custom FQ vs official TOFU harness | **documented limitation** | Note divergence from official KS-test eval; plan to cross-check on official harness |
| Baselines are reruns, not published numbers | **documented** | Report our reruns + cite published LoKU/FLAT numbers for context, explain the gap (model = Phi-1.5/Llama-2 vs their Llama-2; our oracle is scratch-trained) |
| Single benchmark headline (forget05) | partial | forget01/05/10 reported; forget05 is the standard split; per-split tuning noted as limitation |
| WMDP misses target (0.37 vs 0.30) | **honest limitation** | MCQ-format training added; if still short, report as negative result |

---

*This audit is the basis for the rewritten method_summary.md and the paper method section.*
