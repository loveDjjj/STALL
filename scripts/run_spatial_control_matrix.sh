#!/usr/bin/env bash
set -euo pipefail

# Local Spatial 必要性控制矩阵：所有变体使用通用 K=3 窗口和 real-only refit。
# 依次回答最终模型能否移除 Spatial、Spatial 单独是否有效、Spatial 是否改善 Local，
# 并补齐同协议 Global-only。各项串行执行，避免并发读取同一机械盘缓存。
# 单卡示例：bash scripts/run_spatial_control_matrix.sh --set 'runtime.devices=[cuda:1]'
# 预检示例：bash scripts/run_spatial_control_matrix.sh --dry-run

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANTS=(
  full_d2_k3_no_spatial_refit
  local_spatial_only_refit
  local_spatial_d2_refit
  global_only_k3_refit
)

for variant in "${VARIANTS[@]}"; do
  echo "[Spatial 控制矩阵] 开始 ${variant}" >&2
  bash "$ROOT/scripts/run_ablation.sh" "$variant" "$@"
  echo "[Spatial 控制矩阵] 完成 ${variant}" >&2
done
