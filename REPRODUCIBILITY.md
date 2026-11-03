# Reproducibility

Clean commands for every experiment referenced in `RESULTS_MANIFEST.md`.
All commands assume the repo root as working directory and a CUDA GPU
(this repo is GPU-first throughout — no script silently falls back to CPU
or synthetic data; missing GPU or missing dataset fields raise, they
don't substitute).

**None of the checkpoint/oracle directories referenced below
(`outputs/wk0/`, `wk4/`, `wk6/`–`wk9/`) are included in this repository**
— they are excluded by `.gitignore` as too large for git. Every command
that reads from one of those paths assumes you have already produced it
yourself (training commands below, starting with §0).

```bash
pip install -r requirements.txt
```

## 0. Prerequisite: retain-only oracle checkpoint

Most commands below pass `--ref_model`/`--oracle_adapter` pointing at
`outputs/wk0/phi15_retain{95,99,90}/` — build these first (none exist in
this repo):
```bash
python scripts/finetune_retain_ref.py --split retain95 --out outputs/wk0/phi15_retain95/
python scripts/finetune_retain_ref.py --split retain99 --out outputs/wk0/phi15_retain99/
python scripts/finetune_retain_ref.py --split retain90 --out outputs/wk0/phi15_retain90/
```

## 1. Official TOFU evaluation

Single checkpoint:
```bash
python scripts/evaluate_tofu_official.py \
  --base_model microsoft/phi-1_5 \
  --adapter outputs/wk8/FORGE/forget05/seed42/FORGE \
  --oracle_adapter outputs/wk0/phi15_retain95 \
  --split forget05 \
  --n_forget_probe 50 --n_mu_probe 30 \
  --output_json outputs/official_tofu/FORGE/forget05/seed42.json
```

Full grid (8 methods × 3 seeds, forget-5%):
```bash
bash scripts/run_official_tofu_grid.sh
```

## 2. NPO/SimNPO training

Single method/split/seed:
```bash
python scripts/run_experiments.py \
  --mode full --method NPO --split forget05 --seed 42 \
  --base_model locuslab/tofu_ft_phi-1.5 --ref_model outputs/wk0/phi15_retain95/ \
  --config configs/forge_default.yaml --beta 2.5 --eval_every 10 \
  --out_dir outputs/wk8/NPO/forget05/seed42/
```
```bash
python scripts/run_experiments.py \
  --mode full --method SimNPO --split forget05 --seed 42 \
  --base_model locuslab/tofu_ft_phi-1.5 --ref_model outputs/wk0/phi15_retain95/ \
  --config configs/forge_default.yaml --beta 2.5 --gamma_npo 5.0 --eval_every 10 \
  --out_dir outputs/wk8/SimNPO/forget05/seed42/
```

Full 3-split × 3-seed grid: loop the above over
`--split {forget01,forget05,forget10}` and `--seed {42,123,456}`, with
`--ref_model` set to the matching retain oracle
(`outputs/wk0/phi15_retain{99,95,90}/` for forget01/05/10 respectively).

Unit tests:
```bash
python -m pytest tests/test_npo_simnpo_losses.py -v
```

## 3. Robustness ablation

Train the γ=0 ablation checkpoint (γ=0.5 default checkpoints already exist
under `outputs/wk8/FORGE/`):
```bash
python scripts/run_experiments.py \
  --mode full --method FORGE --split forget05 --seed 42 \
  --base_model locuslab/tofu_ft_phi-1.5 --ref_model outputs/wk0/phi15_retain95/ \
  --config configs/forge_default.yaml --gamma_forge 0.0 --eval_every 10 \
  --out_dir outputs/wk_ablation/FORGE_gamma0/forget05/seed42/
```
Repeat for seeds 123, 456.

Multi-seed comparison:
```bash
python scripts/run_robustness.py --multi_seed_compare \
  --checkpoint_a 'outputs/wk8/FORGE/forget05/seed{seed}/FORGE' --label_a FORGE_gamma0.5 \
  --checkpoint_b 'outputs/wk_ablation/FORGE_gamma0/forget05/seed{seed}/FORGE' --label_b FORGE_gamma0 \
  --base_model locuslab/tofu_ft_phi-1.5 --split forget05 --seeds 42,123,456 \
  --out outputs/wk_robustness/gamma_ablation_multiseed.json
```

## 4. WMDP corrected protocol (or removal statement)

```bash
python scripts/run_wmdp.py --domain cyber --methods GA,NPO,SimNPO,FLAT,LoKU,FORGE,RMU \
  --config configs/forge_wmdp.yaml --out_dir outputs/wk_wmdp_fix/cyber/ --n_corpus 500

python scripts/run_wmdp.py --domain bio --methods GA,NPO,SimNPO,FLAT,LoKU,FORGE,RMU \
  --config configs/forge_wmdp.yaml --out_dir outputs/wk_wmdp_fix/bio/ --n_mcq 500 --n_corpus 500
```
The train-on-test guard (`--train_source wmdp_eval_mcq`) is rejected by
design — see `scripts/run_wmdp.py::_assert_no_train_test_overlap`.
If corrected WMDP cannot be run in your environment (e.g. `cais/wmdp-corpora`
unreachable), the fallback statement is: *"WMDP was not used for benchmark
claims because a corpus-based protocol was outside the current experimental
scope."* — do not substitute the old train-on-test result.

## 5. MUSE verification

```bash
python scripts/verify_muse_dataset.py --output outputs/muse_dataset_verification.json

python scripts/run_muse.py --methods GA,NPO,SimNPO,FLAT,LoKU,FORGE \
  --n_forget 200 --n_retain 200 --config configs/forge_default.yaml \
  --out_dir outputs/wk_muse_fix
```

## 6. Cost / edge profiling

Cost (workstation GPU, already run, Table IX unaffected by any fix in this
pass):
```bash
python scripts/cost_analysis.py  # see outputs/wk9/cost_analysis.json
```

Edge profiling (only meaningful on actual edge/Jetson-class hardware —
will print a warning and refuse to be labeled as edge evidence otherwise):
```bash
python scripts/profile_edge_unlearning.py \
  --model locuslab/tofu_ft_phi-1.5 --base_model microsoft/phi-1_5 \
  --method FORGE --split forget05 --steps 25 --batch_size 1 \
  --output outputs/edge_profile/forge_phi15_edge.json
```

## 7. Statistical aggregation and paper artifact generation

```bash
python scripts/build_provenance.py      # outputs/RESULTS_PROVENANCE.json
python scripts/aggregate_results.py     # outputs/revision_stats/*, figures/fig_pareto_dominance.pdf
python scripts/make_paper_artifacts.py  # paper_tables/*.tex
```

## 8. Extra seeds for the main TOFU forget-5% comparison (5-seed upgrade)

```bash
for METHOD in GA GD KL FLAT LoKU NPO SimNPO FORGE; do
  for SEED in 789 1000; do
    EXTRA=""
    [ "$METHOD" = "NPO" ] && EXTRA="--beta 2.5"
    [ "$METHOD" = "SimNPO" ] && EXTRA="--beta 2.5 --gamma_npo 5.0"
    python scripts/run_experiments.py \
      --mode full --method $METHOD --split forget05 --seed $SEED \
      --base_model locuslab/tofu_ft_phi-1.5 --ref_model outputs/wk0/phi15_retain95/ \
      --config configs/forge_default.yaml $EXTRA --eval_every 10 \
      --out_dir outputs/wk8/$METHOD/forget05/seed$SEED/
  done
done
```
`scripts/build_provenance.py` and `scripts/aggregate_results.py` pick up
seeds 789/1000 automatically (their seed lists already include them; both
skip missing seed directories, so forget-1%/forget-10%, which only have
seeds 42/123/456, are unaffected).

## 9. Manuscript

The manuscript and its generated LaTeX tables are maintained outside this
repository pending submission and peer review, and are not included here.
`scripts/make_paper_artifacts.py` (§7 above) is included for completeness
— it regenerates `paper_tables/*.tex` from `outputs/RESULTS_PROVENANCE.json`
only — but that output directory itself is not part of this release.

## 10. Pre-flight audit (Phase 0, safe to re-run anytime)

```bash
git rev-parse HEAD
python3 --version
python -m pytest tests -q || true
python3 - <<'PY'
import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
PY
```
