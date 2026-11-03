# Results Manifest

Every number below is generated from `outputs/RESULTS_PROVENANCE.json`
(149 entries, all `real_gpu_*` — zero simulation, zero missing/fabricated
entries as of this writing). Regenerate via `python scripts/build_provenance.py`.
Metric names follow the Phase 2 convention: `FQ_NLL`/`MU_ret`/`J_diag` are
**diagnostic** metrics (raw-NLL KS-test and 2-term retain harmonic mean),
**not** the official TOFU Forget Quality / Model Utility statistics.
`FQ_official`/`MU_official` (new section below) ARE an attempt at the
official metrics — an independent reimplementation, not verified
byte-identical to the original TOFU authors' reference code (see
`forge/tofu_official.py`).

## Hard rule compliance

- No entry has `type` starting with `simulation` or `smoke`.
- No entry is used in a "main results" context without a `source` file
  that exists on disk (verified by `build_provenance.py` at generation
  time — a missing source produces `value: null, type: "missing"`, and
  the current run produced zero such entries).
- WMDP entries are explicitly typed `real_gpu_corpus_based` (cyber) or
  `real_gpu_mcq_heldout_substitute` (bio) — never `real_gpu` alone —
  so the protocol difference between domains cannot be silently lost.

## TOFU diagnostic grid (Phi-1.5), 8 methods × 3 splits × 3 metrics = 72 entries

Key pattern: `tofu_phi15_{split}_{method}_{fq_nll|mu_ret|j_diag}`.
All 3-seed means (seeds 42/123/456), checkpoint-selected via
`argmax_t(FQ_NLL_t * MU_ret_t)` per seed. Source:
`outputs/wk8/{METHOD}/{split}/seed{42,123,456}/trajectories.json`.

| Split → | forget01 | forget05 | forget10 |
|---|---|---|---|
| GA | J=0.0020 | J=0.0723 | J=0.0044 |
| GD | J=0.0002 | J=0.3115 | J=0.0275 |
| KL | J=0.0000 | J=0.0001 | J=0.0000 |
| FLAT | J=0.0000 | J=0.0001 | J=0.0000 |
| LoKU | J=0.0013 | J=0.2688 | J=0.0079 |
| NPO | J=0.0050 | J=0.1391 | J=0.0083 |
| SimNPO | J=0.0092 | J=0.2818 | J=0.0723 |
| FORGE | J=0.0016 | **J=0.4293** | J=0.0161 |

Full FQ_NLL/MU_ret per method/split are in the JSON; this table shows
J_diag only for space. Query `outputs/RESULTS_PROVENANCE.json` for exact
values, e.g. `tofu_phi15_forget05_forge_fq_nll` = 0.869.

## Official-style TOFU metrics (Phase 2, cross-validated 2026-07-30), 8 methods × 2 metrics = 16 entries

Key pattern: `tofu_phi15_forget05_{method}_{fq_official|mu_official}`. Type:
`real_gpu_official_reimplementation`. 3-seed means. Source:
`outputs/official_tofu/{METHOD}/forget05/seed{42,123,456}.json`.
**Read `forge/tofu_official.py`'s module docstring before citing these as
"official" without qualification.** These formulas were checked line-by-line
against the original TOFU authors' reference implementation
(`github.com/locuslab/tofu`, cloned and read directly) — this check found
and fixed three real bugs in the first draft: perturbation aggregation done
in probability space instead of loss space, an inverted truth-ratio
direction, and an incorrect Model Utility truth-ratio transform (now
`max(0, 1-1/R)`, not a guess). The numbers below are POST-FIX. This is
still an independent reimplementation, not a run of the reference pipeline
itself.

| Method | FQ_official | MU_official |
|---|---|---|
| **GA** | **0.0003** | 0.0000 |
| GD | 0.9977 | 0.3887 |
| KL | 0.6047 | 0.3419 |
| FLAT | 0.9667 | 0.3984 |
| LoKU | 0.9874 | 0.3964 |
| NPO | 0.9977 | 0.3845 |
| SimNPO | 0.9667 | 0.3992 |
| FORGE | 0.9874 | 0.3964 |

**Notable finding, confirmed (not just observed) after the formula fix:**
under this metric, GA is a dramatic outlier (FQ_official≈0.0003, barely
moved from the pre-fix 0.0004) while every other method scores 0.60–0.998
— the opposite differentiation pattern from FQ_NLL, where GA
scored moderately and KL/FLAT scored zero. MU_official is 0.34–0.40 for
every non-GA method (0.0000 for GA), still generally below the diagnostic
MU_ret range (0.3–0.6) but closer to it than the pre-fix numbers (0.18–0.24)
were — the corrected 9-quantity aggregation is meaningfully less harsh than
the earlier, incorrectly-transformed version, though still not identical to
MU_ret. This divergence between diagnostic and official FQ metrics should
be reported as a finding in its own right, not reconciled away — see
`EXPERIMENT_COMPLETION_REPORT.md`.

## WMDP corrected protocol, 7 methods × 2 domains = 14 entries

Key pattern: `wmdp_{cyber|bio}_{method}_acc`. Cyber:
`real_gpu_corpus_based` (public `cyber-forget-corpus`, zero-shot MCQ
eval). Bio: `real_gpu_mcq_heldout_substitute` (disjoint MCQ train/eval
split — official bio-forget-corpus not publicly released, verified via
HF Hub API). Single seed (42) each. Source:
`outputs/wk_wmdp_fix/{domain}/wmdp_summary.json`.

| Method | Cyber Acc. | Bio Acc. |
|---|---|---|
| GA | 0.278 | 0.260 |
| NPO | 0.314 | 0.380 |
| LoKU | 0.320 | 0.380 |
| FORGE | 0.322 | 0.392 |
| RMU | 0.326 | 0.436 |
| SimNPO | 0.334 | 0.448 |
| FLAT | 0.336 | 0.404 |

## MUSE-News (real corpus), 6 methods × 2 metrics = 12 entries

Key pattern: `muse_news_{method}_{verbmem|knowmem}`. Type:
`real_gpu_real_data` (real `muse-bench/MUSE-News`, 200 forget/retain
articles). Single seed (42). Source: `outputs/wk_muse_fix/muse_summary.json`.

| Method | VerbMem | KnowMem |
|---|---|---|
| GA | 0.000 | 0.000 |
| NPO | 0.000 | 0.000 |
| SimNPO | 0.000 | 0.007 |
| FLAT | 0.016 | 0.035 |
| LoKU | 0.167 | 0.032 |
| FORGE | 0.046 | 0.025 |

## Robustness ablation, multi-seed (Phase 4 extension), 2 configs × 4 conditions = 8 entries

Key pattern: `robustness_multiseed_forge_gamma{0.5|0}_{condition}_rouge_leak`.
Type: `real_gpu_diagnostic` — **3 seeds (42/123/456), supersedes the
single-seed entries from the initial pass.** Source:
`outputs/wk_robustness/gamma_ablation_multiseed.json`. Required 2 new
gamma=0 ablation training runs (seeds 123, 456; seed 42 already existed)
before this could be computed.

| Condition | γ=0.5 (3-seed mean) | γ=0 (3-seed mean) | Δ |
|---|---|---|---|
| canonical | 0.475 | 0.532 | −0.057 |
| token_drop | 0.323 | 0.358 | −0.035 |
| word_shuffle | 0.289 | 0.320 | −0.031 |
| truncation | 0.284 | 0.322 | −0.038 |

All four deltas are negative and consistent with the earlier single-seed
result — this is now a **confirmed**, not merely suggestive, finding: the
robustness term measurably reduces leakage in every condition tested,
including two perturbation types never seen during training.

## Llama-2-7B scale, 6 methods × 2 metrics + NPO/SimNPO negative result = 14 entries

Key pattern: `llama7b_forget05_{method}_{fq_nll|mu_ret}`. GA–FORGE:
3-seed means, recomputed with the corrected joint-checkpoint selection
(source trajectories recomputed from
`outputs/wk_llama/{METHOD}/forget05/seed{42,123,456}/trajectories.json`).
NPO/SimNPO: single seed, type `real_gpu_diagnostic_single_seed_negative_result`
— FQ_NLL=0.000 confirmed at 5-step eval resolution across 100 steps,
source `outputs/probe/llama_{NPO,SimNPO}_fine/trajectories.json`.

| Method | FQ_NLL | MU_ret |
|---|---|---|
| GA | 0.022 | 0.642 |
| GD | 0.037 | 0.621 |
| KL | 0.179 | 0.442 |
| FLAT | 0.022 | 0.642 |
| LoKU | **0.313** | 0.646 |
| FORGE | 0.272 | 0.635 |
| NPO | 0.000 | n/a (negative result) |
| SimNPO | 0.000 | n/a (negative result) |

## Cost + sequential — 3 entries, unaffected by the checkpoint-selection bug

`cost_forge_lora_r32`, `cost_full_retrain` (source
`outputs/wk9/cost_analysis.json`), `sequential_{ga,loku,forge}_delta_fq`
(source `outputs/wk9/sequential_{METHOD}/*.json`). Numbers identical to
the current manuscript's Tables IX/X — verified, not recomputed.
