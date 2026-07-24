#!/usr/bin/env bash
set -euo pipefail

SHARD_INDEX="${1:?usage: run_local_residual_shard.sh SHARD_INDEX}"
NUM_SHARDS="${NUM_SHARDS:-2}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="$(dirname "$ROOT")"
RESULT_ROOT="$ROOT/results/local_residual/window_shards"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/tmp/alpha_stalled_local_residual}"

mkdir -p "$RESULT_ROOT"
cd "$PROJECT_ROOT"

for DATASET in comgenvid videofeedback genvideo; do
  PYTHONPATH="$ROOT/src:$ROOT/tools" conda run --no-capture-output -n stall \
    python "$ROOT/tools/score_local_residual_windows.py" \
    --dataset "$DATASET" \
    --residual-params "$ROOT/results/local_residual/params/${DATASET}_R1.npz" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$SHARD_INDEX" \
    --video-batch-size 4 \
    --frame-batch-size 32 \
    --score-batch-size 4 \
    --decode-workers 4 \
    --checkpoint-dir "$CHECKPOINT_ROOT" \
    --output "$RESULT_ROOT/${DATASET}_R1_shard${SHARD_INDEX}.csv"
done
