# Research Gap Recommendation

> **⚠️ IMPLEMENTATION SCOPE NOTE (added 2026-06-02).** This document is the *original
> research proposal*. The method actually implemented and evaluated is a simpler,
> validated objective — bounded-ascent unlearning with perturbation robustness. The
> Fisher-weighted dual adapters, learned adversarial minimax, and CoT-KL below are
> **scoped as future work**, not claimed as completed. See
> `outputs/reports/method_audit.md` and `outputs/wk3/method_summary.md` for the
> canonical method that the paper describes.

**Chosen approach:** Address all three gaps (A + B + C) in a single unified objective.

**Rationale:** The three gaps are orthogonal (A = capacity/selectivity, B = distribution shift,
C = reasoning traces) and can be combined additively in one training objective without
interference. Each gap corresponds to one loss term and one ablation run.

**Method name:** FORGE — Fisher-Orchestrated Robust Gating for Erasure

**Full objective:**
ℒ(θ,g,ψ) = ℒ_forget + λ·ℒ_retain + γ·ℒ_robust(ψ) + ζ·ℒ_reason + η·ℒ_gate
            min over (θ,g);   max over ψ

**Ablation plan:**
- γ=0 → Gap B contribution (adversarial loop)
- λ=0 → Gap A retain adapter contribution
- η=0 → Gap A gate sparsity contribution
- k=0 → adversary inactive (inner loop disabled)
- r=8  → lower-rank Fisher merge
- ζ=0  → Gap C CoT loss (always 0 in TOFU phase)
