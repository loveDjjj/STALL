#!/usr/bin/env bash
set -euo pipefail

cd /data/OneDay/STALL_project

CSV="STALL/cache/indexes/comgenvid.csv"
PATCH_CACHE="STALL/cache/patch_embeddings/comgenvid"
OUT_DIR="STALL/results/patch_region3_bottomk_sweep"
mkdir -p "$OUT_DIR"

bottomks=(0.10 0.15 0.25 0.30)

for bottomk in "${bottomks[@]}"; do
  tag="${bottomk/./p}"
  params="STALL/precomputed/patch_params_comgenvid_real_second_order_region3_bottomk${tag}_v2.npz"
  result="$OUT_DIR/comgenvid_patch_region3_second_order_bottomk${tag}_spat10_temp90.csv"

  echo
  echo "== Region3 second-order bottomk=${bottomk}: fit params =="
  PYTHONPATH= \
  PYTHONNOUSERSITE=1 \
  TORCH_HOME=/tmp/torch_cache \
  XDG_CACHE_HOME=/tmp \
  conda run --no-capture-output -n stall \
    python STALL/src/create_patch_params.py \
      --csv "$CSV" \
      --patch-emb-cache "$PATCH_CACHE" \
      --output "$params" \
      --duration 2 \
      --compact \
      --real-only \
      --max-patches-for-fit 300000 \
      --patch-temp-mode same_grid_second_order \
      --patch-region-size 3 \
      --aggregation bottomk_mean \
      --bottomk-ratio "$bottomk"

  echo
  echo "== Region3 second-order bottomk=${bottomk}: eval spat10/temp90 =="
  PYTHONPATH= \
  PYTHONNOUSERSITE=1 \
  TORCH_HOME=/tmp/torch_cache \
  XDG_CACHE_HOME=/tmp \
  conda run --no-capture-output -n stall \
    python STALL/src/eval_patch_fast.py \
      --csv "$CSV" \
      --patch-emb-cache "$PATCH_CACHE" \
      --patch-params "$params" \
      --output-csv "$result" \
      --duration 2 \
      --compact \
      --score-device cuda \
      --score-batch-size 64 \
      --patch-temp-mode same_grid_second_order \
      --patch-spat-weight 0.10 \
      --patch-temp-weight 0.90
done

echo
echo "Region3 bottom-k sweep outputs: $OUT_DIR"
