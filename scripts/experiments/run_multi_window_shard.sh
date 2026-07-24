#!/usr/bin/env bash
set -euo pipefail

SAMPLING="${1:?usage: run_multi_window_shard.sh SAMPLING SHARD_INDEX}"
SHARD_INDEX="${2:?usage: run_multi_window_shard.sh SAMPLING SHARD_INDEX}"
NUM_SHARDS="${NUM_SHARDS:-2}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="$(dirname "$ROOT")"
RESULT_ROOT="$ROOT/results/multi_window_joint_typicality/window_scores"

case "$SAMPLING" in
  K3_uniform)
    TOOL="score_multi_window.py"
    CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/tmp/alpha_stalled_multi_window_k3_full}"
    ;;
  K5_uniform|all_nonoverlap)
    TOOL="score_multi_window_incremental.py"
    CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/tmp/alpha_stalled_multi_window_incremental}"
    ;;
  *)
    echo "SAMPLING must be K3_uniform, K5_uniform, or all_nonoverlap" >&2
    exit 2
    ;;
esac

mkdir -p "$RESULT_ROOT"
cd "$PROJECT_ROOT"

for DATASET in comgenvid videofeedback genvideo; do
  COMMON_ARGS=(
    --dataset "$DATASET"
    --sampling "$SAMPLING"
    --num-shards "$NUM_SHARDS"
    --shard-index "$SHARD_INDEX"
    --checkpoint-dir "$CHECKPOINT_ROOT"
    --output "$RESULT_ROOT/${DATASET}_${SAMPLING}_shard${SHARD_INDEX}.csv"
  )
  if [[ "$SAMPLING" == "K3_uniform" ]]; then
    EXTRA_ARGS=(--video-batch-size 4 --frame-batch-size 32 --decode-workers 4)
  else
    EXTRA_ARGS=(
      --video-batch-size 8
      --frame-batch-size 64
      --decode-workers 8
      --decode-attempts 3
      --reuse-score "$RESULT_ROOT/${DATASET}_K3_uniform_shard${SHARD_INDEX}.csv"
    )
    if [[ "$SAMPLING" == "all_nonoverlap" ]]; then
      EXTRA_ARGS+=(--reuse-score "$RESULT_ROOT/${DATASET}_K5_uniform_shard${SHARD_INDEX}.csv")
    fi
  fi
  PYTHONPATH="$ROOT/src:$ROOT/tools" conda run --no-capture-output -n stall \
    python "$ROOT/tools/$TOOL" "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
done
