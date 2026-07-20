#!/usr/bin/env bash
set -euo pipefail

# ComGenVid 1s representative duration/window experiment.
#
# This script mirrors the main ComGenVid patch-branch protocol in
# configs/alpha_stalled.yaml while changing only the sampled temporal window
# from 2s to 1s. It intentionally reports patch-only scores, because matching
# 1s global-branch scores are not part of the current release assets.
#
# Usage:
#   bash scripts/run_comgenvid_duration1_representative.sh
#   DEBUG_N=2 bash scripts/run_comgenvid_duration1_representative.sh
#
# Optional environment variables:
#   CONDA_ENV=stall
#   DEVICE=cuda
#   NUM_WORKERS=8
#   VIDEO_BATCH=4
#   FRAME_BATCH=32
#   SCORE_BATCH=16
#   DEBUG_N=2          # smoke mode; unset for full run
#   OUT_DIR=...

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="$(dirname "${ROOT_DIR}")"
REPO_NAME="$(basename "${ROOT_DIR}")"
cd "${PROJECT_DIR}"

CONDA_ENV="${CONDA_ENV:-stall}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VIDEO_BATCH="${VIDEO_BATCH:-4}"
FRAME_BATCH="${FRAME_BATCH:-32}"
SCORE_BATCH="${SCORE_BATCH:-16}"
OUT_DIR="${OUT_DIR:-${REPO_NAME}/results/journal_experiments/duration_window_representative}"

CSV="${REPO_NAME}/cache/indexes/comgenvid.csv"
PATCH_CACHE="${REPO_NAME}/cache/patch_embeddings/comgenvid"
DURATION=1
PATCH_REGION_SIZE=3
AGGREGATION="bottomk_mean"
BOTTOMK_RATIO="0.20"
PATCH_TEMP_MODE="same_grid_second_order"
PATCH_SPAT_WEIGHT="0.10"
PATCH_TEMP_WEIGHT="0.90"

mkdir -p "${OUT_DIR}"

DEBUG_ARGS=()
PARAM_DEBUG_ARGS=()
SUFFIX="full"
if [[ -n "${DEBUG_N:-}" ]]; then
  DEBUG_ARGS=(--debug-n "${DEBUG_N}")
  PARAM_DEBUG_ARGS=(--max-real-videos "${DEBUG_N}")
  SUFFIX="debug${DEBUG_N}"
fi

PARAMS="${REPO_NAME}/precomputed/patch_params_comgenvid_1s_real_second_order_region3_bottomk0p20_${SUFFIX}.npz"
PATCH_CSV="${OUT_DIR}/comgenvid_1s_patch_${SUFFIX}.csv"
METRICS_CSV="${OUT_DIR}/comgenvid_1s_metrics_${SUFFIX}.csv"
PREFILL_CSV="${OUT_DIR}/comgenvid_1s_prefill_${SUFFIX}.csv"

echo "[1/4] Prefill compact patch cache for ComGenVid ${DURATION}s (${SUFFIX})"
conda run --no-capture-output -n "${CONDA_ENV}" python "${REPO_NAME}/tools/prefill_patch_cache.py" \
  --csv "${CSV}" \
  --patch-emb-cache "${PATCH_CACHE}" \
  --duration "${DURATION}" \
  --compact \
  "${DEBUG_ARGS[@]}" \
  --num-workers "${NUM_WORKERS}" \
  --video-batch "${VIDEO_BATCH}" \
  --frame-batch "${FRAME_BATCH}" \
  --device "${DEVICE}" \
  --execute \
  --output-summary-csv "${PREFILL_CSV}"

echo "[2/4] Build real-video patch calibration parameters"
conda run --no-capture-output -n "${CONDA_ENV}" python "${REPO_NAME}/src/create_patch_params.py" \
  --csv "${CSV}" \
  --patch-emb-cache "${PATCH_CACHE}" \
  --output "${PARAMS}" \
  --duration "${DURATION}" \
  --compact \
  --real-only \
  "${PARAM_DEBUG_ARGS[@]}" \
  --patch-temp-mode "${PATCH_TEMP_MODE}" \
  --patch-region-size "${PATCH_REGION_SIZE}" \
  --aggregation "${AGGREGATION}" \
  --bottomk-ratio "${BOTTOMK_RATIO}"

echo "[3/4] Evaluate patch-only scores"
conda run --no-capture-output -n "${CONDA_ENV}" python "${REPO_NAME}/src/eval_patch_fast.py" \
  --csv "${CSV}" \
  --patch-emb-cache "${PATCH_CACHE}" \
  --patch-params "${PARAMS}" \
  --output-csv "${PATCH_CSV}" \
  --duration "${DURATION}" \
  --compact \
  "${DEBUG_ARGS[@]}" \
  --score-device "${DEVICE}" \
  --score-batch-size "${SCORE_BATCH}" \
  --patch-spat-weight "${PATCH_SPAT_WEIGHT}" \
  --patch-temp-weight "${PATCH_TEMP_WEIGHT}" \
  --patch-temp-mode "${PATCH_TEMP_MODE}" \
  --patch-region-size "${PATCH_REGION_SIZE}" \
  --aggregation "${AGGREGATION}" \
  --bottomk-ratio "${BOTTOMK_RATIO}"

echo "[4/4] Evaluate pairwise-balanced metrics"
conda run --no-capture-output -n "${CONDA_ENV}" python "${REPO_NAME}/tools/eval_score_csv.py" \
  --csv "${PATCH_CSV}" \
  --score-col patch_final_score \
  --output-csv "${METRICS_CSV}"

echo "Done:"
echo "  prefill: ${PREFILL_CSV}"
echo "  params:  ${PARAMS}"
echo "  scores:  ${PATCH_CSV}"
echo "  metrics: ${METRICS_CSV}"
