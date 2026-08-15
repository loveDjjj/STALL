#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOG_DIR="${LOG_DIR:-logs/full_coverage_paper_protocol}"

run_shard() {
  local shard="$1"
  local gpu="$2"
  local dataset
  for dataset in comgenvid videofeedback genvideo; do
    conda run -n stall python tools/score_duration_aware_original_k1.py \
      --input results/full_coverage_paper_protocol/original_k1_pending_cache.csv \
      --input-mode cache \
      --dataset "$dataset" \
      --num-shards 2 \
      --shard-index "$shard" \
      --score-device "cuda:${gpu}" \
      --batch-size 4
  done
  for dataset in comgenvid videofeedback genvideo; do
    conda run -n stall python tools/score_duration_aware_original_k1.py \
      --input results/full_coverage_paper_protocol/original_k1_pending_dino.csv \
      --input-mode video \
      --dataset "$dataset" \
      --num-shards 2 \
      --shard-index "$shard" \
      --extract-device "cuda:${gpu}" \
      --score-device "cuda:${gpu}" \
      --batch-size 4 \
      --frame-batch-size 32
  done
}

mkdir -p "$LOG_DIR"
run_shard 0 0 > "$LOG_DIR/original_k1_shard0.log" 2>&1 &
pid0=$!
run_shard 1 1 > "$LOG_DIR/original_k1_shard1.log" 2>&1 &
pid1=$!

wait "$pid0"
wait "$pid1"
