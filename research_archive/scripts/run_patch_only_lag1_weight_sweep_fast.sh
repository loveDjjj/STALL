#!/usr/bin/env bash
set -euo pipefail

cd /data/OneDay/STALL_project

CSV="STALL/cache/indexes/comgenvid.csv"
PATCH_CACHE="STALL/cache/patch_embeddings/comgenvid"
PARAMS="STALL/precomputed/patch_params_comgenvid_real_samegrid_lag1_v2.npz"
OUT_DIR="STALL/results/patch_only_lag1_weight_sweep"

mkdir -p "$OUT_DIR"

weights=(
  "1.0 0.0 spat_only"
  "0.8 0.2 spat80_temp20"
  "0.7 0.3 spat70_temp30"
  "0.6 0.4 spat60_temp40"
  "0.5 0.5 spat50_temp50"
  "0.4 0.6 spat40_temp60"
  "0.2 0.8 spat20_temp80"
  "0.0 1.0 temp_only"
)

for item in "${weights[@]}"; do
  read -r spat_w temp_w name <<<"$item"
  echo
  echo "== Patch-only lag1 weight: ${name} (spat=${spat_w}, temp=${temp_w}) =="
  PYTHONNOUSERSITE=1 conda run --no-capture-output -n stall \
    python STALL/src/eval_patch_fast.py \
      --csv "$CSV" \
      --patch-emb-cache "$PATCH_CACHE" \
      --patch-params "$PARAMS" \
      --output-csv "$OUT_DIR/comgenvid_patch_lag1_${name}.csv" \
      --duration 2 \
      --compact \
      --score-device cuda \
      --score-batch-size 32 \
      --patch-temp-mode same_grid_lag1 \
      --patch-spat-weight "$spat_w" \
      --patch-temp-weight "$temp_w"
done

echo
echo "Weight sweep outputs: $OUT_DIR"
