#!/usr/bin/env bash
set -euo pipefail

# 开发集核心实验矩阵：依次运行 Global、Local 时序和 K=1 参考，不并发抢占两张评分卡。
# 完整锁定 K=3 主实验复用 results/runs/alpha_stall_locked_u0，不在此重复运行。
# D1 与 K=1 的运行名含 refit，表示 Local 参数仅由各数据集互斥 calibration real 重新拟合。
# 默认双卡示例：bash scripts/run_development_matrix.sh
# 指定单卡示例：bash scripts/run_development_matrix.sh --set 'runtime.devices=[cuda:0]'
# 预检示例：bash scripts/run_development_matrix.sh --dry-run

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANTS=(global_only local_d2_locked local_d1_refit full_d1_refit full_k1_refit)

for variant in "${VARIANTS[@]}"; do
  echo "[矩阵] 开始 ${variant}" >&2
  bash "$ROOT/scripts/run_ablation.sh" "$variant" "$@"
  echo "[矩阵] 完成 ${variant}" >&2
done
