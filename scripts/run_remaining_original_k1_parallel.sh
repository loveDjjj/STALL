#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOG_DIR="${LOG_DIR:-logs/full_coverage_paper_protocol}"

CACHE_SHARDS="${CACHE_SHARDS:-8}"
VIDEO_SHARDS="${VIDEO_SHARDS:-4}"

run_cache_shard() {
  local shard="$1"
  local gpu="$2"
  conda run -n stall python tools/score_duration_aware_original_k1.py \
    --input results/full_coverage_paper_protocol/original_k1_pending_cache_genvideo_remaining.csv \
    --input-mode cache \
    --dataset genvideo \
    --num-shards "$CACHE_SHARDS" \
    --shard-index "$shard" \
    --score-device "cuda:${gpu}" \
    --batch-size 4
}

run_video_shard() {
  local shard="$1"
  local gpu="$2"
  conda run -n stall python tools/score_duration_aware_original_k1.py \
    --input results/full_coverage_paper_protocol/original_k1_pending_dino.csv \
    --input-mode video \
    --dataset videofeedback \
    --num-shards "$VIDEO_SHARDS" \
    --shard-index "$shard" \
    --extract-device "cuda:${gpu}" \
    --score-device "cuda:${gpu}" \
    --batch-size 4 \
    --frame-batch-size 16
}

mkdir -p "$LOG_DIR"
pids=()
for ((shard=0; shard<CACHE_SHARDS; shard++)); do
  gpu=$((shard % 2))
  run_cache_shard "$shard" "$gpu" \
    > "$LOG_DIR/original_k1_cache_shard${shard}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

pids=()
for ((shard=0; shard<VIDEO_SHARDS; shard++)); do
  gpu=$((shard % 2))
  run_video_shard "$shard" "$gpu" \
    > "$LOG_DIR/original_k1_video_shard${shard}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
