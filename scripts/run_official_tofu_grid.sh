#!/usr/bin/env bash
# scripts/run_official_tofu_grid.sh — batch official-style TOFU FQ/MU across
# all 8 methods, forget05 split, seeds 42/123/456. See forge/tofu_official.py
# for the honesty caveat (independent reimplementation, not cross-validated
# against the original TOFU reference code).
set -e

METHODS="GA GD KL FLAT LoKU NPO SimNPO FORGE"
SEEDS="42 123 456"
SPLIT="forget05"
ORACLE="outputs/wk0/phi15_retain95"
N_FORGET_PROBE=50
N_MU_PROBE=30

mkdir -p outputs/official_tofu logs

for METHOD in $METHODS; do
  for SEED in $SEEDS; do
    ADAPTER="outputs/wk8/${METHOD}/${SPLIT}/seed${SEED}/${METHOD}"
    OUT="outputs/official_tofu/${METHOD}/${SPLIT}/seed${SEED}.json"
    if [ ! -d "$ADAPTER" ]; then
      echo "SKIP: adapter missing for ${METHOD} seed${SEED}: ${ADAPTER}"
      continue
    fi
    echo "=== ${METHOD} ${SPLIT} seed${SEED} ==="
    python scripts/evaluate_tofu_official.py \
      --base_model microsoft/phi-1_5 \
      --adapter "$ADAPTER" \
      --oracle_adapter "$ORACLE" \
      --split "$SPLIT" \
      --n_forget_probe $N_FORGET_PROBE --n_mu_probe $N_MU_PROBE \
      --output_json "$OUT" \
      || echo "FAILED: ${METHOD} seed${SEED} (see traceback above)"
  done
done

echo "OFFICIAL_TOFU_GRID_DONE"
