#!/usr/bin/env bash
set -euo pipefail

# Stage 3 条件真实动力学矩阵：一次读取 Patch cache，同时运行
# D1 p(normalized vector D2 | current-speed bin) 与
# D3 p(4-D geometry | current-speed bin)。D0 无条件 D2 复用正式 C0，
# D2 无条件 geometry 复用 Stage 2 T5；slow/medium/fast 边界仅由
# 每个数据集 200 条互斥 calibration real 的 1/3、2/3 分位确定。
#
# 用法：
#   bash scripts/run_conditional.sh --dry-run
#   bash scripts/run_conditional.sh
#   bash scripts/run_conditional.sh --overwrite --set runtime.score_batch_size=32
#
# 只允许传 --dry-run、--overwrite 和 runtime 资源覆盖；不得按数据集或 fake
# 结果修改 bin 数、边界、DINO、采样、Global 或融合权重。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda run --no-capture-output -n "${STALL_ENV:-stall}" \
  python "$ROOT/scripts/run_trajectory_matrix.py" --matrix conditional "$@"
