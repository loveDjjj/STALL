#!/usr/bin/env bash
set -euo pipefail

# Alpha STALL 默认主方法：使用 benchmark.yaml 中的 Global + Local D2 + K=3 配置。
# 可透传 runner 参数：--dry-run（只检查计划）、--overwrite（明确覆盖同名未冻结结果）、
# --set KEY=VALUE（覆盖运行资源或受控实验参数）、--scores-csv PATH（导入外部分数）。
# 单卡示例：bash scripts/run_alpha_stall.sh --set runtime.device=cuda:1
# 双卡示例：bash scripts/run_alpha_stall.sh --set 'runtime.devices=[cuda:0,cuda:1]'
# 吞吐示例：bash scripts/run_alpha_stall.sh --set runtime.score_batch_size=16 --set runtime.cache_io_workers=4
# 请勿用此入口改变 sampling、method 或 data；对应对比实验应使用 run_ablation.sh 等专用脚本。
# 用法示例：bash scripts/run_alpha_stall.sh --dry-run
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for argument in "$@"; do
  if [[ "$argument" == "--run-name" || "$argument" == --run-name=* ]]; then
    echo "run_alpha_stall.sh 固定使用运行名 alpha_stall，请不要传入 --run-name" >&2
    exit 2
  fi
done
conda run --no-capture-output -n "${STALL_ENV:-stall}" python "$ROOT/scripts/run_experiment.py" --run-name alpha_stall "$@"
