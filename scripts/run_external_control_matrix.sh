#!/usr/bin/env bash
set -euo pipefail

# GenVidBench 外部控制矩阵：冻结开发集确定的无 Spatial 方法结构，仅切换外部数据身份。
# 三项共享 199 条 VRiPT real calibration 与 600 条 evaluation；外部 fake 不参与参数拟合。
# final_k3 是正式外部结果，final_k1 检验窗口覆盖，global_k3 检验 Local D2 外部贡献。
# 单卡示例：bash scripts/run_external_control_matrix.sh --set 'runtime.devices=[cuda:1]'
# 预检示例：bash scripts/run_external_control_matrix.sh --dry-run

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMON=(
  --set 'data.datasets=[genvidbench]'
  --set calibration.real_videos_per_dataset=199
  --set method.local.parameter_source=fit_real_only
)

echo "[外部矩阵] 开始 final_k3" >&2
conda run --no-capture-output -n "${STALL_ENV:-stall}" python "$ROOT/scripts/run_experiment.py" \
  --run-name alpha_stall_external_genvidbench \
  "${COMMON[@]}" \
  --set method.local.spatial_enabled=false \
  --set method.local.temporal_order=2 \
  --set sampling.num_windows=3 \
  "$@"
echo "[外部矩阵] 完成 final_k3" >&2

echo "[外部矩阵] 开始 final_k1" >&2
conda run --no-capture-output -n "${STALL_ENV:-stall}" python "$ROOT/scripts/run_experiment.py" \
  --run-name alpha_stall_external_genvidbench_k1 \
  "${COMMON[@]}" \
  --set method.local.spatial_enabled=false \
  --set method.local.temporal_order=2 \
  --set sampling.num_windows=1 \
  "$@"
echo "[外部矩阵] 完成 final_k1" >&2

echo "[外部矩阵] 开始 global_k3" >&2
conda run --no-capture-output -n "${STALL_ENV:-stall}" python "$ROOT/scripts/run_experiment.py" \
  --run-name alpha_stall_external_genvidbench_global_k3 \
  "${COMMON[@]}" \
  --set method.local.enabled=false \
  --set sampling.num_windows=3 \
  "$@"
echo "[外部矩阵] 完成 global_k3" >&2
