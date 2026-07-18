#!/usr/bin/env bash
set -euo pipefail

cd /data/OneDay/STALL_project

CSV="STALL/cache/indexes/comgenvid.csv"
PATCH_CACHE="STALL/cache/patch_embeddings/comgenvid"
PARAMS="STALL/precomputed/patch_params_comgenvid_real_second_order_region2_bottomk0p20_v2.npz"
OUT_DIR="STALL/results/patch_region2_weight_sweep"

mkdir -p "$OUT_DIR"

weights=(
  "0.0 1.0 temp_only"
  "0.05 0.95 spat05_temp95"
  "0.1 0.9 spat10_temp90"
  "0.2 0.8 spat20_temp80"
  "0.3 0.7 spat30_temp70"
)

for item in "${weights[@]}"; do
  read -r spat_w temp_w name <<<"$item"
  echo
  echo "== Region2 second-order weight: ${name} (spat=${spat_w}, temp=${temp_w}) =="
  PYTHONPATH= \
  PYTHONNOUSERSITE=1 \
  TORCH_HOME=/tmp/torch_cache \
  XDG_CACHE_HOME=/tmp \
  conda run --no-capture-output -n stall \
    python STALL/src/eval_patch_fast.py \
      --csv "$CSV" \
      --patch-emb-cache "$PATCH_CACHE" \
      --patch-params "$PARAMS" \
      --output-csv "$OUT_DIR/comgenvid_patch_region2_${name}.csv" \
      --duration 2 \
      --compact \
      --score-device cuda \
      --score-batch-size 64 \
      --patch-temp-mode same_grid_second_order \
      --patch-spat-weight "$spat_w" \
      --patch-temp-weight "$temp_w"
done

echo
echo "Region2 weight sweep outputs: $OUT_DIR"
