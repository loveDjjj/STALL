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

bash scripts/run_u0_injection_shard.sh "$gpu_index" "$shard_index"
bash scripts/run_u0_external_genvidbench_shard.sh "$gpu_index" "$shard_index"
bash scripts/run_u0_robustness_shard.sh "$gpu_index" "$shard_index"
