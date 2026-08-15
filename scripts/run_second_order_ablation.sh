#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATASETS=(comgenvid videofeedback genvideo)
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
# Opt in explicitly: GPU1 may be occupied by an unrelated service.
GPU1_AVAILABLE="${GPU1_AVAILABLE:-0}"

conda run --no-capture-output -n stall \
  python tools/build_independent_remaining_real_manifest.py

for dataset in "${DATASETS[@]}"; do
  conda run --no-capture-output -n stall python tools/fit_u0_local_d1_params.py \
    --dataset "$dataset" --device "cuda:${GPU0}"
done

run_shard() {
  local dataset="$1"
  local shard="$2"
  local gpu="$3"
  local sampling="${4:-k3}"
  local num_shards="${5:-2}"
  CUDA_VISIBLE_DEVICES="$gpu" conda run --no-capture-output -n stall \
    python tools/score_u0_local_d1_windows.py \
      --dataset "$dataset" --sampling "$sampling" \
      --shard-index "$shard" --num-shards "$num_shards" \
      --extract-device cuda:0 --score-device cuda:0 \
      --frame-batch-size 32 --no-reuse-k1-cache \
      --output-dir results/second_order_independent_calibration/d1_windows_b32
}

run_pair() {
  local dataset="$1"
  local sampling="$2"
  local num_shards="${3:-2}"
  local second_gpu="$GPU1"
  if [[ "$GPU1_AVAILABLE" != "1" ]]; then
    second_gpu="$GPU0"
  fi
  run_shard "$dataset" 0 "$GPU0" "$sampling" "$num_shards" &
  pid0=$!
  run_shard "$dataset" 1 "$second_gpu" "$sampling" "$num_shards" &
  pid1=$!
  wait "$pid0"
  wait "$pid1"
}

for dataset in "${DATASETS[@]}"; do
  run_pair "$dataset" calibration_k1
done

for dataset in "${DATASETS[@]}"; do
  if [[ "$dataset" == "genvideo" ]]; then
    run_pair "$dataset" k3 4
    run_shard "$dataset" 2 "$GPU0" k3 4 &
    pid2=$!
    run_shard "$dataset" 3 "$GPU0" k3 4 &
    pid3=$!
    wait "$pid2"
    wait "$pid3"
  else
    run_pair "$dataset" k3 2
  fi
done

# The literal independent-real complement also evaluates the historical locked
# calibration identities as test real under each new reserve-based calibration.
run_candidate_shard() {
  local dataset="$1"
  local shard="$2"
  local gpu="$3"
  local num_shards="$4"
  CUDA_VISIBLE_DEVICES="$gpu" conda run --no-capture-output -n stall \
    python tools/score_u0_calibration_candidates.py \
      --split locked_calibration --dataset "$dataset" \
      --shard-index "$shard" --num-shards "$num_shards" \
      --extract-device cuda:0 --score-device cuda:0 \
      --frame-batch-size 32
}

for dataset in "${DATASETS[@]}"; do
  if [[ "$dataset" == "genvideo" ]]; then
    second_gpu="$GPU1"
    if [[ "$GPU1_AVAILABLE" != "1" ]]; then
      second_gpu="$GPU0"
    fi
    run_candidate_shard "$dataset" 0 "$GPU0" 2 &
    pid0=$!
    run_candidate_shard "$dataset" 1 "$second_gpu" 2 &
    pid1=$!
    wait "$pid0"
    wait "$pid1"
  else
    run_candidate_shard "$dataset" 0 "$GPU0" 1
  fi
done

for shard in 0 1 2 3; do
  gpu="$GPU0"
  if [[ "$GPU1_AVAILABLE" == "1" && "$shard" -eq 1 ]]; then
    gpu="$GPU1"
  fi
  CUDA_VISIBLE_DEVICES="$gpu" conda run --no-capture-output -n stall \
    python tools/score_u0_calibration_candidates.py \
      --split independent_remaining_real --dataset videofeedback \
      --shard-index "$shard" --num-shards 4 \
      --extract-device cuda:0 --score-device cuda:0 \
      --frame-batch-size 32 &
done
wait

conda run --no-capture-output -n stall \
  python tools/analyze_independent_real_complement.py

conda run --no-capture-output -n stall \
  python tools/analyze_second_order_independent_calibration.py \
    --iterations 1000 --workers 3
