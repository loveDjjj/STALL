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
  raw_file="$output_dir/k1_raw/${dataset}_shard$(printf '%02d' "$shard_index")_of_02.csv"
  if [[ -f "$raw_file" ]]; then
    echo "reuse completed K1 cache shard: $raw_file"
    continue
  fi
  CUDA_VISIBLE_DEVICES="$gpu_index" conda run --no-capture-output -n stall \
    python tools/score_u0_locked_k1_cache.py \
    --dataset "$dataset" \
    --shard-index "$shard_index" \
    --num-shards 2 \
    --device cuda \
    --batch-size 16 \
    --output-dir "$output_dir"
done
