#!/usr/bin/env bash
set -euo pipefail

# 最终无 Spatial 方法的严格控制矩阵：补齐 Full D1 与 K=1 对照。
# 两项都使用通用缓存、real-only refit、Global+D2-only 结构；分别只改变 D1/D2 或 K1/K3。
# 单卡示例：bash scripts/run_final_no_spatial_matrix.sh --set 'runtime.devices=[cuda:1]'
# 预检示例：bash scripts/run_final_no_spatial_matrix.sh --dry-run

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANTS=(full_d1_k3_no_spatial_refit full_d2_k1_no_spatial_refit)

for variant in "${VARIANTS[@]}"; do
  echo "[最终无 Spatial 矩阵] 开始 ${variant}" >&2
  bash "$ROOT/scripts/run_ablation.sh" "$variant" "$@"
  echo "[最终无 Spatial 矩阵] 完成 ${variant}" >&2
done
