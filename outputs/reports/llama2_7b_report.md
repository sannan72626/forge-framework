# Llama-2-7B Validation Report — HONEST VERDICT
## FORGE: scale-generalization test
**Date completed:** 2026-06-04
**Hardware:** RTX 3090 (23 GB) | Llama-2-7B (4-bit NF4) | base mirror NousResearch/Llama-2-7b-hf

---

## 1. Goal

Test whether FORGE's headline advantage over LoKU (established on Phi-1.5) **replicates at
7B scale** — the #1 reviewer concern. This is the make-or-break experiment for the paper's
central claim.

## 2. Setup
- **Unlearning target:** `locuslab/tofu_ft_llama2-7b`
- **Oracle:** `NousResearch/Llama-2-7b-hf` base fine-tuned on TOFU retain-95% (scratch, merged)
- **Grid:** 6 methods × 3 seeds on forget05 (18 runs) at lr=2e-5/250 steps
- **Sweep:** FORGE & LoKU at lr∈{3e-5,5e-5}/400 steps + fine-eval (every 5 steps) at lr=2e-5

## 3. Main result (3 seeds, forget05)

| Method | FQ (peak) | MU (final) | FQ×MU |
|--------|-----------|-----------|-------|
| **LoKU** | **0.313** | **0.599** | **0.188** |
| FORGE | 0.272 | 0.593 | 0.161 |
| KL | 0.179 | 0.420 | 0.075 |
| FLAT | 0.022 | 0.552 | 0.012 |
| GD | 0.037 | 0.002 | 0.000 |
| GA | 0.022 | 0.000 | 0.000 |

**FORGE vs LoKU: FQ p=0.42, MU p=0.56 — statistically tied; LoKU nominally ahead on both.**
FORGE significantly beats GA/GD/KL/FLAT (all p<0.05) but NOT LoKU.

## 4. Hyperparameter sweep + fine-eval (does the advantage re-emerge?)

| Config | FORGE peak FQ | LoKU peak FQ |
|--------|--------------|--------------|
| lr=2e-5, fine-eval (every 5 steps) | 0.272 @ step45 | **0.396 @ step55** |
| lr=3e-5, 400 steps | 0.112 | 0.068 |
| lr=5e-5, 400 steps | 0.272 | **0.396** |

The fine-grained eval (every 5 steps) confirms FORGE's true peak is **0.272** — not a
sampling artifact. LoKU reaches **0.396** on both FQ and MU. Higher lr makes FORGE overshoot
faster (lower peak). **No operating point was found where FORGE beats LoKU at 7B.**

## 5. Why the advantage doesn't transfer

FORGE's `tanh`-bounded ascent bounds the gradient *magnitude*, but the model still drifts
past the oracle NLL over successive steps — at 7B this overshoot is fast, so the FQ peak is
brief and low (0.272). LoKU's **hard clamp at margin** stops forgetting once NLL reaches the
threshold, which is inherently more stable at scale. On the small Phi-1.5 model the overshoot
was slower, letting FORGE's bounded ascent sit near the oracle NLL longer (FQ peak 0.869) —
that operating-point advantage **does not exist at 7B**.

## 6. HONEST VERDICT

**The central claim ("FORGE uniquely balances FQ and MU, beating LoKU") is Phi-1.5-specific
and does NOT hold at Llama-2-7B.** At 7B, LoKU is better on both FQ and MU.

What remains true at BOTH scales:
- FORGE robustly beats the naive baselines (GA, GD, KL, FLAT) with significance.
- FORGE *matches* (does not beat) LoKU, the best PEFT baseline, at 7B.

## 7. Implications for publication

The paper **cannot** claim FORGE > LoKU as a general result. Viable honest framings:
1. **Scale-dependence study:** "bounded-ascent unlearning is competitive at small scale but
   its advantage over hard-clamp methods (LoKU) does not persist at 7B" — an honest
   empirical contribution about operating-point sensitivity in unlearning. Modest; TMLR or
   a workshop.
2. **Method fix:** add the hard-clamp / early-stop mechanism that makes LoKU stable, i.e.
   actually combine FORGE's robustness term with LoKU's clamp, and re-test. This is new
   method work, not just tuning.
3. **Do not publish the "FORGE wins" story** — it is not supported at scale.

## 8. Outputs
| File | Description |
|------|-------------|
| `outputs/wk_llama/{METHOD}/forget05/seed{42,123,456}/` | 18-run grid trajectories |
| `outputs/wk_llama/stats_summary.json` | 3-seed stats + paired t-tests |
| `outputs/wk_llama_sweep/` | lr sweep + fine-eval trajectories |
| `outputs/wk0/llama2_retain95_merged/` | 7B retain oracle (merged, 13 GB) |

---

*Report generated: 2026-06-04 | Honest negative result on scale generalization.*

---

## 9. ADDENDUM — Option 2 (FORGE-Hybrid) tested and FAILED

Implemented `FORGEH` = LoKU's hard-clamp forget + clamped perturbation-robustness term.
Fair head-to-head at the principled margin = 2.3 (≈ oracle NLL), fine-eval, seed 42:

| Method | peak FQ | MU@peak | Trajectory |
|--------|---------|---------|-----------|
| LoKU | 0.396 @ step55 | 0.673 | identical |
| FORGEH | 0.396 @ step55 | 0.673 | **bit-for-bit identical** |

**Root cause:** the perturbation term clamps `NLL(perturb(forget))` at the margin. Token
corruption makes the input harder than the oracle target, so `NLL(perturb(forget)) > margin`
from step 0 → clamp saturates → **the robustness term contributes zero gradient**. FORGEH
reduces exactly to LoKU. At a higher margin (4.0) the term becomes active but causes overshoot
(FORGEH peak 0.179 < LoKU 0.396).

**Conclusion:** No FORGE-family configuration beats LoKU at 7B. The perturbation-robustness
mechanism is either inert (low margin) or harmful (high margin). Option 2 does not yield a
publishable improvement. Recommend Option 1 (honest scale-dependence framing) OR a
fundamentally different robustness mechanism (e.g. output-consistency KL between clean and
paraphrased forget, not a clamped NLL term) — the latter is genuine new method research.
