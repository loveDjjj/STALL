#!/usr/bin/env bash
set -euo pipefail

cd /data/OneDay/STALL_project

CSV="STALL/cache/indexes/comgenvid.csv"
PATCH_CACHE="STALL/cache/patch_embeddings/comgenvid"
PARAMS="STALL/precomputed/patch_params_comgenvid_real_second_order_v2.npz"
OUT_DIR="STALL/results/patch_only_second_order_weight_sweep"

mkdir -p "$OUT_DIR"

weights=(
  "1.0 0.0 spat_only"
  "0.8 0.2 spat80_temp20"
  "0.6 0.4 spat60_temp40"
  "0.5 0.5 spat50_temp50"
  "0.4 0.6 spat40_temp60"
  "0.3 0.7 spat30_temp70"
  "0.2 0.8 spat20_temp80"
  "0.1 0.9 spat10_temp90"
  "0.0 1.0 temp_only"
)

for item in "${weights[@]}"; do
  read -r spat_w temp_w name <<<"$item"
  echo
  echo "== Patch-only second-order weight: ${name} (spat=${spat_w}, temp=${temp_w}) =="
  PYTHONPATH= \
  PYTHONNOUSERSITE=1 \
  TORCH_HOME=/tmp/torch_cache \
  XDG_CACHE_HOME=/tmp \
  conda run --no-capture-output -n stall \
    python STALL/src/eval_patch_fast.py \
      --csv "$CSV" \
      --patch-emb-cache "$PATCH_CACHE" \
      --patch-params "$PARAMS" \
      --output-csv "$OUT_DIR/comgenvid_patch_second_order_${name}.csv" \
      --duration 2 \
      --compact \
      --score-device cuda \
      --score-batch-size 64 \
      --patch-temp-mode same_grid_second_order \
      --patch-spat-weight "$spat_w" \
      --patch-temp-weight "$temp_w"
done

echo
echo "Second-order weight sweep outputs: $OUT_DIR"
