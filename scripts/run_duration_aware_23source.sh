#!/usr/bin/env bash
set -euo pipefail

# Full duration-aware Alpha-STALLED confirmation run.
# Run from the repository root. Scoring is resumable by physical video ID.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-stall}"
NUM_SHARDS="${NUM_SHARDS:-2}"
SHARD_INDEX="${SHARD_INDEX:-0}"
DEVICE="${DEVICE:-cuda:0}"
BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS:-1000}"
FIT_PARAMS="${FIT_PARAMS:-1}"

conda run --no-capture-output -n "${CONDA_ENV}" \
  python tools/build_duration_aware_23source_protocol.py

# Requires the per-frame patch embedding caches named by the calibration
# indexes. Existing parameter files are reused unless the fitter receives
# --force explicitly.
if [[ "${FIT_PARAMS}" == "1" ]]; then
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python tools/fit_duration_aware_local_params.py
fi

for dataset in comgenvid videofeedback genvideo; do
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python tools/score_duration_aware_23source.py \
      --dataset "${dataset}" \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${SHARD_INDEX}" \
      --extract-device "${DEVICE}" \
      --score-device "${DEVICE}" \
      --video-batch-size 4 \
      --frame-batch-size 32 \
      --decode-workers 4
done

if [[ "${ANALYZE:-0}" == "1" ]]; then
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python tools/analyze_duration_aware_23source.py \
      --bootstrap-iterations "${BOOTSTRAP_ITERATIONS}"
fi
