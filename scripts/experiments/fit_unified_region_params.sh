#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET="${1:?usage: fit_unified_region_params.sh DATASET}"
OUTPUT_DIR="$ROOT/results/unified_multiscale_layers/region_params"
mkdir -p "$OUTPUT_DIR"

case "$DATASET" in
  comgenvid)
    INDEX="$ROOT/cache/indexes/comgenvid_calib_real200.csv"
    CACHE="$ROOT/cache/patch_embeddings/comgenvid"
    AGGREGATION="bottomk_mean"
    BOTTOMK="0.2"
    ;;
  videofeedback)
    INDEX="$ROOT/cache/indexes/videofeedback_small_calib_real200.csv"
    CACHE="$ROOT/cache/patch_embeddings/videofeedback"
    AGGREGATION="mean"
    BOTTOMK="0.5"
    ;;
  genvideo)
    INDEX="$ROOT/cache/indexes/genvideo_calib_real200.csv"
    CACHE="$ROOT/cache/patch_embeddings/genvideo"
    AGGREGATION="mean"
    BOTTOMK="0.5"
    ;;
  *)
    echo "unknown dataset: $DATASET" >&2
    exit 2
    ;;
esac

for REGION in 1 2 3; do
  OUTPUT="$OUTPUT_DIR/${DATASET}_region${REGION}.npz"
  if [[ -f "$OUTPUT" ]]; then
    echo "reuse $OUTPUT"
    continue
  fi
  python "$ROOT/src/create_patch_params.py" \
    --csv "$INDEX" \
    --patch-emb-cache "$CACHE" \
    --output "$OUTPUT" \
    --duration 2 \
    --compact \
    --real-only \
    --max-patches-for-fit 300000 \
    --aggregation "$AGGREGATION" \
    --bottomk-ratio "$BOTTOMK" \
    --seed 42 \
    --patch-temp-mode same_grid_second_order \
    --patch-region-size "$REGION"
done
