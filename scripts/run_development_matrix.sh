#!/usr/bin/env bash
set -euo pipefail

# 开发集核心实验矩阵：依次运行最终无 Spatial 方法的 Global、Local、D1 和 K=1 对照。
# 完整 D2 K=3 主方法由 run_alpha_stall.sh 单独运行，不在此重复；全部对照使用 real-only refit。
# 默认双卡示例：bash scripts/run_development_matrix.sh
# 指定单卡示例：bash scripts/run_development_matrix.sh --set 'runtime.devices=[cuda:0]'
# 预检示例：bash scripts/run_development_matrix.sh --dry-run

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANTS=(
  global_only_k3_refit
  local_d1_refit
  local_d2_refit
  full_d1_k3_no_spatial_refit
  full_d2_k1_no_spatial_refit
)

for variant in "${VARIANTS[@]}"; do
  echo "[矩阵] 开始 ${variant}" >&2
  bash "$ROOT/scripts/run_ablation.sh" "$variant" "$@"
  echo "[矩阵] 完成 ${variant}" >&2
done
