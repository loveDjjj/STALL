#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/OneDay/STALL_project/STALL}"
OUT_DIR="${OUT_DIR:-$ROOT/results/likelihood_tail_features}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCORE_DEVICE="${SCORE_DEVICE:-cpu}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-128}"

mkdir -p "$OUT_DIR"

run_one() {
  local dataset="$1"
  local csv="$2"
  local cache="$3"
  local params="$4"
  local region="$5"
  local out="$OUT_DIR/${dataset}_likelihood_tail_features.csv"

  "$PYTHON_BIN" "$ROOT/tools/patch_likelihood_tail_features.py" \
    --csv "$csv" \
    --patch-emb-cache "$cache" \
    --patch-params "$params" \
    --output-csv "$out" \
    --duration 2 \
    --compact \
    --score-device "$SCORE_DEVICE" \
    --score-batch-size "$SCORE_BATCH_SIZE" \
    --patch-temp-mode same_grid_second_order \
    --patch-region-size "$region"
}

run_one \
  "comgenvid" \
  "$ROOT/cache/indexes/comgenvid.csv" \
  "$ROOT/cache/patch_embeddings/comgenvid" \
  "$ROOT/precomputed/patch_params_comgenvid_real_second_order_region3_bottomk0p50_v2.npz" \
  3

run_one \
  "videofeedback" \
  "$ROOT/cache/indexes/videofeedback.csv" \
  "$ROOT/cache/patch_embeddings/videofeedback" \
  "$ROOT/precomputed/patch_params_videofeedback_real_same_grid_second_order_region1_mean_v2.npz" \
  1

run_one \
  "genvideo" \
  "$ROOT/cache/indexes/genvideo.csv" \
  "$ROOT/cache/patch_embeddings/genvideo" \
  "$ROOT/precomputed/patch_params_genvideo_real_same_grid_second_order_region2_mean_v2.npz" \
  2

echo "Done."
