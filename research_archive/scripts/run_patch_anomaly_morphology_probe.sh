#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/OneDay/STALL_project/STALL}"
PYTHON_BIN="${PYTHON_BIN:-conda run --no-capture-output -n stall python}"
OUT_DIR="${OUT_DIR:-$ROOT/results/patch_anomaly_morphology}"

mkdir -p "$OUT_DIR"

$PYTHON_BIN "$ROOT/tools/patch_anomaly_morphology_scores.py" \
  --csv "$ROOT/cache/indexes/comgenvid.csv" \
  --patch-emb-cache "$ROOT/cache/patch_embeddings/comgenvid" \
  --patch-params "$ROOT/precomputed/patch_params_comgenvid_real_second_order_region3_bottomk0p50_v2.npz" \
  --output-csv "$OUT_DIR/comgenvid_morphology_full.csv" \
  --duration 2 \
  --compact \
  --patch-temp-mode same_grid_second_order \
  --patch-region-size 3 \
  --score-device cuda \
  --score-batch-size 64

python3 "$ROOT/tools/three_score_fusion.py" \
  --dataset comgenvid \
  --tag morphology_frame_entropy_r0p2_rank \
  --score-a-csv "$ROOT/results/comgenvid_results.csv" \
  --score-b-csv "$ROOT/results/patch_region3_bottomk_sweep/comgenvid_patch_region3_second_order_bottomk0p50_spat10_temp90.csv" \
  --score-c-csv "$OUT_DIR/comgenvid_morphology_full.csv" \
  --score-c-col frame_entropy_r0p2_pct_low \
  --score-c-name morphology_frame_entropy_low \
  --weight-step 0.05 \
  --rank-normalize \
  --output-dir "$OUT_DIR/fusion_probe"
