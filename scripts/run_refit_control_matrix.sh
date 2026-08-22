#!/usr/bin/env bash
set -euo pipefail

# 严格重拟合控制矩阵：补齐 D1/D2 与 K1/K3 的单因素比较。
# 两项均使用通用 K=3 packed cache、目标数据集互斥真实 calibration 和同一评测集；
# 因而只能分别改变 Local 时序阶数或窗口数，不能与 locked U0 历史主表混合。
# 单卡示例：bash scripts/run_refit_control_matrix.sh --set 'runtime.devices=[cuda:1]'
# 预检示例：bash scripts/run_refit_control_matrix.sh --dry-run

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANTS=(local_d2_refit full_d2_k3_refit)

for variant in "${VARIANTS[@]}"; do
  echo "[控制矩阵] 开始 ${variant}" >&2
  bash "$ROOT/scripts/run_ablation.sh" "$variant" "$@"
  echo "[控制矩阵] 完成 ${variant}" >&2
done
