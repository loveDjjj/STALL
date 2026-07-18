#!/usr/bin/env bash
set -euo pipefail

cd /data/OneDay/STALL_project

CSV="STALL/cache/indexes/comgenvid.csv"
PATCH_CACHE="STALL/cache/patch_embeddings/comgenvid"
PARAM_DIR="STALL/precomputed"
RESULT_DIR="STALL/results/patch_second_order_bottomk_sweep"

mkdir -p "$RESULT_DIR"

ratios=("0.01" "0.03" "0.10")

for ratio in "${ratios[@]}"; do
  ratio_tag="${ratio/./p}"
  params="${PARAM_DIR}/patch_params_comgenvid_real_second_order_bottomk${ratio_tag}_v2.npz"
  output="${RESULT_DIR}/comgenvid_patch_second_order_bottomk${ratio_tag}_spat10_temp90.csv"

  echo
  echo "== Fit params: second-order bottomk=${ratio} =="
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
      --aggregation bottomk_mean \
      --bottomk-ratio "$ratio"

  echo
  echo "== Eval: second-order bottomk=${ratio}, spat10/temp90 =="
  PYTHONPATH= \
  PYTHONNOUSERSITE=1 \
  TORCH_HOME=/tmp/torch_cache \
  XDG_CACHE_HOME=/tmp \
  conda run --no-capture-output -n stall \
    python STALL/src/eval_patch_fast.py \
      --csv "$CSV" \
      --patch-emb-cache "$PATCH_CACHE" \
      --patch-params "$params" \
      --output-csv "$output" \
      --duration 2 \
      --compact \
      --score-device cuda \
      --score-batch-size 64 \
      --patch-temp-mode same_grid_second_order \
      --patch-spat-weight 0.1 \
      --patch-temp-weight 0.9
done

echo
echo "Bottom-k sweep outputs: $RESULT_DIR"
