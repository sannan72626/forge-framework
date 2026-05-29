# Research Gap Analysis

## Gap A — Selective LoRA Forgetting (addressed by FORGE component 1)

**Problem:** All existing PEFT unlearning methods use a single LoRA adapter that must
simultaneously suppress D_f and preserve D_r, creating gradient conflicts.
LoKU (ICLR'25) partially addresses this with Fisher initialisation but still uses one adapter.

**Evidence:** At forget-10%, LoKU MU drops from 0.878 (forget-1%) to 0.819 — a 5.9pp drop.
This is the best published result; all other methods drop ≥19pp (Yuan et al. ICLR'25).

**Proposed solution:** Dual LoRA adapters (Δ_f, Δ_r) with separate objectives, merged via
diagonal Fisher weighting: Δ_merged = (F_f⊙Δ_f + F_r⊙Δ_r)/(F_f+F_r+ε).
Token gate g_φ(h)∈[0,1] routes the merged delta selectively at token level.

**Novel contribution:** First PEFT method to separate forget/retain optimisation trajectories.

---

## Gap B — Adversarial Robustness (addressed by FORGE component 2)

**Problem:** No existing unlearning method trains against distributional probes.
All methods are evaluated on canonical queries only; paraphrase / entity-swap / CoT-prefix
attacks recover "forgotten" content in LoKU (mean reactivation 0.228) and FLAT (0.145).

**Evidence:** Thaker et al. (SaTML'25) show that TOFU FQ can be high while paraphrase
probes still elicit correct answers. CMU critique demonstrates 5 attack templates.

**Proposed solution:** Adversarial inner loop: probe generator Φ_ψ (prefix-tuned soft prompt)
updated k steps to maximise ℒ_robust; unlearner minimises it in outer step.
Minimax: min_{θ,g} max_ψ ℒ_robust(ψ).

**Novel contribution:** First unlearning method with adversarially-trained robustness guarantee.

---

## Gap C — CoT Trace Leakage (addressed by FORGE component 3)

**Problem:** R-TOFU (Yoon et al. EMNLP'25) shows answer-level FQ can reach 1.0 while
reasoning traces still reconstruct forbidden content under CoT-prefix or temperature > 0.
No existing method targets intermediate reasoning steps.

**Evidence:** R-TOFU evaluation: FLAT achieves FQ=0.91 on canonical queries but trace-level
ROUGE-L remains 0.34 — meaning the reasoning path still generates memorised content.

**Proposed solution:** Step-wise KL divergence on CoT traces:
ℒ_reason = Σ_k KL(p_θ(z_k|z_{<k},x) ‖ p_ref(z_k|z_{<k},x))
Active when ζ > 0 (TOFU baseline: ζ=0; R-TOFU extension: ζ=0.2).

**Novel contribution:** First method targeting per-step reasoning trace distributions.
