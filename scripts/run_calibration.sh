#!/usr/bin/env bash
set -euo pipefail

# 校准可靠性实验：仅改变真实校准视频数量和抽样种子，方法与评测身份保持固定。
# 可透传：--dry-run、--overwrite、--set runtime.device=cuda:1、
# --set 'runtime.devices=[cuda:0,cuda:1]'、--set runtime.score_batch_size=16、--set runtime.cache_io_workers=1。
# 用法示例：bash scripts/run_calibration.sh 17 200 --dry-run
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEED="${1:?usage: run_calibration.sh <seed> <real-video-count> [runner args]}"
COUNT="${2:?usage: run_calibration.sh <seed> <real-video-count> [runner args]}"
shift 2
for argument in "$@"; do
  if [[ "$argument" == "--run-name" || "$argument" == --run-name=* ]]; then
    echo "run_calibration.sh 根据 seed 和样本量固定运行名，请不要传入 --run-name" >&2
    exit 2
  fi
done
conda run --no-capture-output -n "${STALL_ENV:-stall}" python "$ROOT/scripts/run_experiment.py" \
  --run-name "alpha_stall_seed${SEED}_n${COUNT}" \
  --set "calibration.seed=${SEED}" \
  --set "calibration.real_videos_per_dataset=${COUNT}" \
  --set method.local.parameter_source=fit_real_only \
  "$@"
