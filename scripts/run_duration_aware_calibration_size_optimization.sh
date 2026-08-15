#!/usr/bin/env bash
set -euo pipefail

# Two-stage leakage-free calibration-size optimization.
#   MODE=select: real-only fitting/validation; run once before fake scoring.
#   MODE=confirm: score the frozen selected candidate for one deterministic shard.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-stall}"
MODE="${MODE:-select}"
DEVICE="${DEVICE:-cuda:0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
ANALYZE="${ANALYZE:-0}"
BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS:-1000}"

if [[ "${MODE}" == "select" ]]; then
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python tools/build_duration_aware_23source_protocol.py
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python tools/fit_duration_aware_local_params.py
  for dataset in comgenvid videofeedback genvideo; do
    conda run --no-capture-output -n "${CONDA_ENV}" \
      python tools/score_duration_aware_real_curve.py \
        --dataset "${dataset}" --device "${DEVICE}"
  done
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python tools/score_duration_aware_size_candidates.py \
      --dataset comgenvid --candidate-sizes 200 400 600 800 \
      --label real_curve_current --split calibration \
      --extract-device "${DEVICE}" --score-device "${DEVICE}"
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python tools/select_duration_aware_calibration_size.py \
      --bootstrap-iterations 1000
elif [[ "${MODE}" == "confirm" ]]; then
  for dataset in comgenvid videofeedback genvideo; do
    size="$(conda run -n "${CONDA_ENV}" python -c \
      "import json; print(json.load(open('results/duration_aware_23source/real_only_curve/selected_calibration_sizes.json'))['selected_sizes']['${dataset}'])" \
      | tail -n 1)"
    conda run --no-capture-output -n "${CONDA_ENV}" \
      python tools/score_duration_aware_size_candidates.py \
        --dataset "${dataset}" --candidate-sizes "${size}" \
        --label selected_realonly --split calibration evaluation \
        --num-shards "${NUM_SHARDS}" --shard-index "${SHARD_INDEX}" \
        --extract-device "${DEVICE}" --score-device "${DEVICE}"
  done
  if [[ "${ANALYZE}" == "1" ]]; then
    conda run --no-capture-output -n "${CONDA_ENV}" \
      python tools/analyze_duration_aware_23source.py \
        --bootstrap-iterations "${BOOTSTRAP_ITERATIONS}"
  fi
else
  echo "MODE must be select or confirm" >&2
  exit 2
fi
