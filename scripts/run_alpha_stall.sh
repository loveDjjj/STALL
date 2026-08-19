#!/usr/bin/env bash
set -euo pipefail

# Alpha STALL 默认主方法：使用 benchmark.yaml 中的 Global + Local D2 + K=3 配置。
# 用法示例：bash scripts/run_alpha_stall.sh --dry-run
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for argument in "$@"; do
  if [[ "$argument" == "--run-name" || "$argument" == --run-name=* ]]; then
    echo "run_alpha_stall.sh 固定使用运行名 alpha_stall，请不要传入 --run-name" >&2
    exit 2
  fi
done
conda run --no-capture-output -n "${STALL_ENV:-stall}" python "$ROOT/scripts/run_experiment.py" --run-name alpha_stall "$@"
