# FORGE — Parameter-Efficient Bounded-Ascent Machine Unlearning

Companion code for the FORGE parameter-efficient bounded-ascent machine
unlearning research line. The manuscript itself and its generated LaTeX
tables are maintained separately pending submission and peer review, and
are **not included in this repository** — see "Release scope and
limitations" below.

**Note on naming:** earlier development iterations of this project used
the name "Fisher-Orchestrated Robust Gating for Erasure" and an
ambitious Fisher-weighted dual-adapter / adversarial-probe / CoT-KL
architecture (see git history and `forge/trainer.py`'s in-file history
note). That design was scoped down during implementation to the simpler,
validated objective described below and in the manuscript. This README
describes the actual, validated method — not the earlier proposal.

## What FORGE actually is

FORGE unlearns a LoRA adapter on top of a frozen base LLM using a
tanh-bounded forget-ascent loss, a retain-likelihood anchor, and a
token-drop robustness term:

```
L_FORGE = -m*tanh(l_f/m) + lambda*l_r - gamma*m*tanh(l_drop/m)
```

It is one member of a family of saturating/bounded forget-ascent
objectives that also includes NPO, SimNPO, LoKU's hard clamp, and Stable
Forgetting — see Section III-G of the manuscript for a direct comparison.

## Quick start

```bash
pip install -r requirements.txt
```

GPU is required for all real experiments in this repo — no script
silently falls back to CPU or synthetic data (missing GPU or missing
dataset fields raise an error rather than substitute a fake result).

Train FORGE on TOFU forget-5% (Phi-1.5):
```bash
python scripts/run_experiments.py \
  --mode full --method FORGE --split forget05 --seed 42 \
  --base_model locuslab/tofu_ft_phi-1.5 --ref_model outputs/wk0/phi15_retain95/ \
  --config configs/forge_default.yaml --eval_every 10 \
  --out_dir outputs/wk8/FORGE/forget05/seed42/
```

See `REPRODUCIBILITY.md` for the complete command reference (official
TOFU metrics, NPO/SimNPO baselines, WMDP, MUSE, robustness ablation,
cost/edge profiling, and paper artifact generation).

## Repository map

```
forge/
  trainer.py           Loss functions (GA/GD/KL/NPO/SimNPO/FLAT/LoKU/FORGE).
                        NOT currently imported by the working pipeline —
                        see the in-file status note. scripts/run_experiments.py
                        has its own copy and is what actually produced every
                        real result in this repo.
  tofu_official.py      Independent reimplementation of the official TOFU
                        Truth-Ratio Forget Quality and 9-quantity Model
                        Utility. Not cross-validated against the original
                        authors' reference code — see module docstring.
  stats.py               Bootstrap CI + paired t-tests, single-checkpoint-
                        consistent selection (argmax FQ*MU per seed).
  metrics.py, adaptive_lr.py   Supporting utilities.

scripts/
  run_experiments.py     Main TOFU training/eval runner (all methods).
  finetune_retain_ref.py  Builds the retain-only oracle checkpoint that
                          --ref_model/--oracle_adapter point to below — run
                          this first; the oracle checkpoint itself is not
                          included in this repo.
  run_wmdp.py             WMDP-cyber (corpus-based) / WMDP-bio (held-out
                          MCQ split) — has a hard guard against train-on-test.
  run_muse.py              MUSE-News on the real muse-bench/MUSE-News corpus.
  run_robustness.py       Perturbed-prompt robustness eval + gamma ablation.
  verify_muse_dataset.py   Confirms the MUSE data source is real, not proxy.
  evaluate_tofu_official.py, run_official_tofu_grid.sh   Official metrics.
  profile_edge_unlearning.py   Edge/Jetson profiling (not yet run on real
                                edge hardware — see outputs/edge_profile/NOT_RUN.md).
  aggregate_results.py    Per-seed CSV, mean/std/median, bootstrap CI,
                          effect sizes, Pareto-dominance figure.
  build_provenance.py     Builds outputs/RESULTS_PROVENANCE.json — every
                          number used anywhere must trace back through this.
  make_paper_artifacts.py Generates paper_tables/*.tex from verified data
                          only. Included for completeness; its output
                          (paper_tables/) and the manuscript itself
                          (paper/main.tex) are not part of this repository.

outputs/                  Experiment outputs; see RESULTS_MANIFEST.md. Raw
                          per-step checkpoints/trajectories behind the main
                          tables (wk0/, wk4/, wk6/–wk9/) are excluded by
                          .gitignore (too large for git) — only derived
                          summaries/JSON/CSV are tracked.
tests/                    Loss-function sanity tests (NPO/SimNPO, GPU-only)
                          and CPU-only regression tests for checkpoint
                          restoration, probe separation, and metric-
                          aggregation correctness.
```

## Data integrity policy

This repo follows a strict no-fabrication policy, enforced structurally
where possible:
- `scripts/run_wmdp.py` raises if asked to train on the evaluation MCQ set.
- `scripts/run_muse.py` raises if the real dataset fails to load (no
  synthetic-text fallback).
- `forge/tofu_official.py` raises if required official TOFU fields are
  missing rather than substituting diagnostic data.
- `outputs/RESULTS_PROVENANCE.json` tags every reported number with an
  explicit type (`real_gpu_diagnostic`, `real_gpu_official_reimplementation`,
  `real_gpu_mcq_heldout_substitute`, etc.) and its exact source file.
- Where a claim (edge/IoT deployment validation) cannot currently be
  supported with real data, the repo says so explicitly
  (`outputs/edge_profile/NOT_RUN.md`) rather than fabricating a number.

## Release scope and limitations

This repository contains the training/evaluation code, configuration, and
derived result summaries described above. It intentionally does **not**
include:

- Trained LoRA adapter weights for any method/seed/split (`*.safetensors`
  etc. are excluded by `.gitignore` — too large for git). Any command
  below that reads or writes `outputs/wk0/`, `wk4/`, `wk6/`–`wk9/` requires
  you to run the corresponding training command yourself first (e.g.
  `scripts/finetune_retain_ref.py` for the retain-only oracle, see
  `REPRODUCIBILITY.md` §0) — none of those checkpoint directories exist in
  this repo.
- The manuscript (`paper/main.tex`) and the generated LaTeX tables
  (`paper_tables/`), maintained separately pending submission and peer
  review.
- Internal correction/audit working notes produced while finding and
  fixing the issues listed below. Only the resulting code fixes and
  derived result files are published here.

Publishing this code does not by itself mean every historical result has
been independently re-verified against every fix below — see
`RESULTS_MANIFEST.md` / `outputs/RESULTS_PROVENANCE.json` for the
per-number status (`real_gpu_diagnostic`, `real_gpu_official_reimplementation`,
etc.) of each reported value.

## Known historical issues and corrections

- **Checkpoint selection vs. final-step save.** Reported FQ/MU/J values
  for a given seed should come from the *same* checkpoint, selected by
  `argmax_t(FQ_NLL_t * MU_ret_t)` (`forge/stats.py::_select_checkpoint`).
  Some historical runs — including the robustness-ablation checkpoints
  under `outputs/wk_ablation/` — were trained and saved before this rule
  was enforced throughout the pipeline; their included JSON holds the
  full per-step trajectory, not only the selected step, so the rule can be
  re-applied independently. These have not been retroactively
  recomputed in this release.
- **Diagnostic vs. official TOFU metrics.** `FQ_NLL`/`MU_ret`/`J_diag`
  (from `scripts/run_experiments.py::measure_metrics`) are a lighter-weight
  diagnostic (raw-NLL KS-test; 2-term retain harmonic mean) — **not** the
  official TOFU Truth-Ratio Forget Quality / 9-quantity Model Utility.
  `forge/tofu_official.py` / `scripts/evaluate_tofu_official.py` are an
  independent reimplementation of the official metrics and are **not
  verified byte-identical** to the original TOFU authors' reference code.
  `outputs/official_tofu/` holds this reimplementation's output, kept
  separate from the diagnostic numbers.
- **Training-time vs. evaluation-time perturbations.** Both default to a
  0.20 drop-rate / 0.7 keep-fraction numeral, but they are not the same
  operation: training's token-drop (`scripts/run_experiments.py`) zero-fills
  tokenizer tokens across the full "Q: ...\nA: ..." string (question and
  answer); evaluation's token-drop/word-shuffle/truncation
  (`scripts/run_robustness.py`) perturb whitespace-split words of the
  question only. Sharing the same numeral does not mean they test the
  same perturbation.
- **MUSE-News synthetic-fallback (fixed).** An earlier version of
  `scripts/run_muse.py` pointed at a dataset id that does not exist on the
  HF Hub (`jaechan-repo/muse-bench`), so its synthetic-text fallback fired
  silently on 100% of runs — any MUSE result predating this fix is not
  real-data evidence. `scripts/verify_muse_dataset.py` now checks and
  records this explicitly, and `outputs/wk_muse_fix/` holds the corrected,
  verified-real-data results.
- **`outputs/wk_llama/stats_summary.json` is superseded.** It pairs each
  seed's peak-FQ step with its final-step MU rather than reading both from
  the same checkpoint, producing misleading values (e.g. GA's `mu_mean`
  reads `0.0`, not its actual single-checkpoint value of ≈0.64). Do not
  use it — use `outputs/wk_llama/stats_summary_corrected.json`, which
  applies the single-checkpoint-consistent rule above. See
  `outputs/wk_llama/STATS_SUMMARY_NOTICE.md`. The original file is kept
  unmodified, not deleted, only superseded.

None of the above is being claimed as fully resolved simply because this
updated code is now public — full retroactive checkpoint reselection and
byte-identical official-metric validation, among other items, remain open.
See `RESULTS_MANIFEST.md` for what has and has not been independently
reproduced.

## Key results (see `RESULTS_MANIFEST.md` for full provenance)

- On TOFU forget-5% (Phi-1.5), FORGE has the strongest diagnostic joint
  score of 8 methods tested; SimNPO is strongest on forget-1%/10%.
- Under an independent official-metric reimplementation, nearly all
  bounded/preference-based methods score similarly high, while plain
  gradient ascent is a marked outlier — a different ranking than the
  diagnostic metric produces.
- On WMDP (corrected, non-train-on-test protocol), plain gradient ascent
  dominates hazard suppression over every retain-preserving method,
  including FORGE.
- FORGE's robustness term is validated by a 3-seed ablation: it reduces
  leakage under every perturbation condition tested, including two never
  seen during training.
- At Llama-2-7B, FORGE remains competitive with but does not dominate
  LoKU; NPO/SimNPO fail to reach a working operating point at this scale.
- Deployment cost (0.444% trainable params, 4.61 GB peak VRAM, 43.0x
  speedup) is measured on a workstation GPU only — not validated on edge
  hardware.
