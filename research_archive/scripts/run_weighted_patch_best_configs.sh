#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/OneDay/STALL_project/STALL}"
OUT_DIR="${OUT_DIR:-$ROOT/results/patch_reliability_weighted}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCORE_DEVICE="${SCORE_DEVICE:-cpu}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-64}"

mkdir -p "$OUT_DIR"

run_one() {
  local dataset="$1"
  local csv="$2"
  local cache="$3"
  local params="$4"
  local mode="$5"
  local region="$6"
  local bottomk="$7"

  local out="$OUT_DIR/${dataset}_patch_reliability_weighted.csv"
  echo "============================================================"
  echo "Dataset=$dataset"
  echo "CSV:     $csv"
  echo "Cache:   $cache"
  echo "Params:  $params"
  echo "Output:  $out"
  echo "============================================================"

  "$PYTHON_BIN" "$ROOT/tools/eval_patch_reliability_weighted.py" \
    --csv "$csv" \
    --patch-emb-cache "$cache" \
    --patch-params "$params" \
    --output-csv "$out" \
    --duration 2 \
    --compact \
    --score-device "$SCORE_DEVICE" \
    --score-batch-size "$SCORE_BATCH_SIZE" \
    --patch-temp-mode "$mode" \
    --patch-region-size "$region" \
    --bottomk-ratio "$bottomk" \
    --patch-spat-weight 0.10 \
    --patch-temp-weight 0.90 \
    --temporal-weight 1.0 \
    --spatial-weight 1.0
}

run_one \
  "comgenvid" \
  "$ROOT/cache/indexes/comgenvid.csv" \
  "$ROOT/cache/patch_embeddings/comgenvid" \
  "$ROOT/precomputed/patch_params_comgenvid_real_second_order_region3_bottomk0p50_v2.npz" \
  "same_grid_second_order" \
  3 \
  0.50

run_one \
  "videofeedback" \
  "$ROOT/cache/indexes/videofeedback.csv" \
  "$ROOT/cache/patch_embeddings/videofeedback" \
  "$ROOT/precomputed/patch_params_videofeedback_real_same_grid_second_order_region1_mean_v2.npz" \
  "same_grid_second_order" \
  1 \
  0.50

run_one \
  "genvideo" \
  "$ROOT/cache/indexes/genvideo.csv" \
  "$ROOT/cache/patch_embeddings/genvideo" \
  "$ROOT/precomputed/patch_params_genvideo_real_same_grid_second_order_region2_mean_v2.npz" \
  "same_grid_second_order" \
  2 \
  0.50

echo "Done."
