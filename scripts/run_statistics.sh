#!/usr/bin/env bash
set -euo pipefail

# Stage 4 高维 D2 统计模型矩阵：一次读取 Patch cache，同时评估
# S1 Ledoit-Wolf 与 S2 OAS shrinkage covariance；S0 empirical 复用正式 C0。
# 两种 shrinkage 系数都仅由每个数据集 200 条互斥 calibration real 闭式估计，
# 不使用 fake 调节。S3 Student-t 与 S4 kNN 依赖的低维 geometry 已在 Stage 2
# 大幅失败，按预注册 gate 剪枝，不再做无意义全量运行。
#
# 用法：
#   bash scripts/run_statistics.sh --dry-run
#   bash scripts/run_statistics.sh
#   bash scripts/run_statistics.sh --overwrite --set runtime.score_batch_size=16
#
# 只允许传 --dry-run、--overwrite 和 runtime 资源覆盖。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda run --no-capture-output -n "${STALL_ENV:-stall}" \
  python "$ROOT/scripts/run_trajectory_matrix.py" --matrix statistics "$@"
