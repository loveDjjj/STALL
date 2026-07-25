#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 GPU_INDEX SHARD_INDEX" >&2
  exit 2
fi

gpu_index="$1"
shard_index="$2"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
params="$repo_root/release/u0_external_genvidbench/params/region1_mean.npz"
release="$repo_root/release/u0_external_genvidbench"
output="$repo_root/results/u0_external_genvidbench"
cd "$repo_root"

CUDA_VISIBLE_DEVICES="$gpu_index" PYTHONNOUSERSITE=1 \
  conda run --no-capture-output -n stall python tools/score_u0_external_genvidbench_k1.py \
  --num-shards 4 \
  --shard-index "$shard_index" \
  --device cuda:0 \
  --local-params "$params"

CUDA_VISIBLE_DEVICES="$gpu_index" PYTHONNOUSERSITE=1 \
  conda run --no-capture-output -n stall python tools/score_u0_locked_windows.py \
  --dataset genvidbench_pair1 \
  --release-dir "$release" \
  --output-dir "$output" \
  --local-params "$params" \
  --num-shards 4 \
  --shard-index "$shard_index" \
  --extract-device cuda:0 \
  --score-device cuda:0
