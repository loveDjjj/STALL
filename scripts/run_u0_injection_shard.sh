#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 GPU_INDEX SHARD_INDEX" >&2
  exit 2
fi

gpu_index="$1"
shard_index="$2"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

for dataset in comgenvid videofeedback genvideo; do
  CUDA_VISIBLE_DEVICES="$gpu_index" PYTHONNOUSERSITE=1 \
    conda run --no-capture-output -n stall python tools/score_u0_injections.py \
    --dataset "$dataset" \
    --num-shards 4 \
    --shard-index "$shard_index" \
    --extract-device cuda:0 \
    --score-device cuda:0
done
