#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-STALL}"
PY="${PY:-/home/ubuntu/anaconda3/envs/stall/bin/python}"
DATASET="${DATASET:-comgenvid}"
MODE="${MODE:-same_grid_second_order}"
BOTTOMK="${BOTTOMK:-0.20}"
RUN_LENGTH="${RUN_LENGTH:-3}"
AGG_REGION="${AGG_REGION:-3}"
PATCH_REGION="${PATCH_REGION:-1}"
SPAT_WEIGHT="${SPAT_WEIGHT:-0.10}"
TEMP_WEIGHT="${TEMP_WEIGHT:-0.90}"
MAX_PATCHES="${MAX_PATCHES:-300000}"
SCORE_DEVICE="${SCORE_DEVICE:-cuda}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-128}"
BASE_PARAMS="${BASE_PARAMS:-${ROOT}/precomputed/patch_params_${DATASET}_real_second_order_v2.npz}"

ratio_tag="${BOTTOMK/./p}"
tag="${DATASET}_${MODE}_strun${RUN_LENGTH}x${AGG_REGION}_bottomk${ratio_tag}_patchregion${PATCH_REGION}_spat${SPAT_WEIGHT/./}_temp${TEMP_WEIGHT/./}"
params="${ROOT}/precomputed/patch_params_${DATASET}_real_${MODE}_strun${RUN_LENGTH}x${AGG_REGION}_bottomk${ratio_tag}_patchregion${PATCH_REGION}_v2.npz"
result="${ROOT}/results/patch_spatiotemporal_run/${tag}.csv"

export PYTHONNOUSERSITE=1
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"

mkdir -p "${ROOT}/results/patch_spatiotemporal_run"

if [[ -s "${params}" ]]; then
  echo "Skip existing params: ${params}"
else
  if [[ ! -s "${BASE_PARAMS}" ]]; then
    echo "Missing BASE_PARAMS=${BASE_PARAMS}" >&2
    echo "Set BASE_PARAMS to an existing same_grid_second_order params file." >&2
    exit 1
  fi
  echo "== Fast recalibrate params from base whitening: ${tag} =="
  "${PY}" "${ROOT}/tools/recalibrate_patch_params_fast.py" \
    --csv "${ROOT}/cache/indexes/${DATASET}.csv" \
    --patch-emb-cache "${ROOT}/cache/patch_embeddings/${DATASET}" \
    --base-params "${BASE_PARAMS}" \
    --output "${params}" \
    --duration 2 \
    --compact \
    --score-device "${SCORE_DEVICE}" \
    --score-batch-size "${SCORE_BATCH_SIZE}" \
    --patch-temp-mode "${MODE}" \
    --patch-region-size "${PATCH_REGION}" \
    --aggregation spatiotemporal_run_bottomk_mean \
    --bottomk-ratio "${BOTTOMK}" \
    --temporal-run-length "${RUN_LENGTH}" \
    --aggregation-region-size "${AGG_REGION}"
fi

if [[ -s "${result}" ]]; then
  echo "Skip existing result: ${result}"
else
  echo "== Eval: ${tag} =="
  "${PY}" "${ROOT}/src/eval_patch_fast.py" \
    --csv "${ROOT}/cache/indexes/${DATASET}.csv" \
    --patch-emb-cache "${ROOT}/cache/patch_embeddings/${DATASET}" \
    --patch-params "${params}" \
    --output-csv "${result}" \
    --duration 2 \
    --compact \
    --score-device "${SCORE_DEVICE}" \
    --score-batch-size "${SCORE_BATCH_SIZE}" \
    --patch-temp-mode "${MODE}" \
    --patch-spat-weight "${SPAT_WEIGHT}" \
    --patch-temp-weight "${TEMP_WEIGHT}"
fi

echo "Params: ${params}"
echo "Result: ${result}"
