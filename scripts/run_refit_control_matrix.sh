#!/usr/bin/env bash
set -euo pipefail

# 历史 Spatial 方法的重拟合控制矩阵，仅用于复核旧 D1/D2、K1/K3 表。
# 最终无 Spatial 方法请使用 run_final_no_spatial_matrix.sh；不要混合两套协议。
# 单卡示例：bash scripts/run_refit_control_matrix.sh --set 'runtime.devices=[cuda:1]'
# 预检示例：bash scripts/run_refit_control_matrix.sh --dry-run

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANTS=(local_d2_refit full_d2_k3_refit)

for variant in "${VARIANTS[@]}"; do
  echo "[控制矩阵] 开始 ${variant}" >&2
  bash "$ROOT/scripts/run_ablation.sh" "$variant" "$@"
  echo "[控制矩阵] 完成 ${variant}" >&2
done
