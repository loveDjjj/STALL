#!/usr/bin/env bash
set -euo pipefail

# 外部泛化实验：保持检测器配置冻结，仅切换指定外部数据集及其真实校准数据。
# 用法示例：bash scripts/run_external.sh genvidbench --dry-run
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:?usage: run_external.sh <dataset-id> [runner args]}"
shift
for argument in "$@"; do
  if [[ "$argument" == "--run-name" || "$argument" == --run-name=* ]]; then
    echo "run_external.sh 根据数据集固定运行名，请不要传入 --run-name" >&2
    exit 2
  fi
done
SETS=(--set "data.datasets=[${DATASET}]")
# 当前 GenVidBench 外部真实校准集固定为 199 条；开发集仍使用基础配置的 200 条。
if [[ "$DATASET" == "genvidbench" ]]; then
  SETS+=(--set calibration.real_videos_per_dataset=199)
fi
conda run --no-capture-output -n "${STALL_ENV:-stall}" python "$ROOT/scripts/run_experiment.py" \
  --run-name "alpha_stall_external_${DATASET}" \
  "${SETS[@]}" \
  "$@"
