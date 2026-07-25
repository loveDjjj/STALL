#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 GPU_INDEX SHARD_INDEX OUTPUT_DIR" >&2
  exit 2
fi

gpu_index="$1"
shard_index="$2"
output_dir="$3"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$repo_root"
for dataset in comgenvid videofeedback genvideo; do
  set +e
  CUDA_VISIBLE_DEVICES="$gpu_index" conda run --no-capture-output -n stall \
    python tools/score_u0_locked_windows.py \
    --dataset "$dataset" \
    --output-dir "$output_dir" \
    --num-shards 2 \
    --shard-index "$shard_index" \
    --video-batch-size 4 \
    --frame-batch-size 32 \
    --decode-workers 4 \
    --decode-attempts 3
  score_status=$?
  set -e
  conda run --no-capture-output -n stall \
    python tools/finalize_u0_locked_shard.py \
    --dataset "$dataset" \
    --output-dir "$output_dir" \
    --num-shards 2 \
    --shard-index "$shard_index"
  if [[ $score_status -ne 0 ]]; then
    echo "scorer exited $score_status after complete checkpoints; finalizer verified recovery" >&2
  fi
done
