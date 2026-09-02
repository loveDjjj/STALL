#!/usr/bin/env bash
set -euo pipefail

# Stage 2 轨迹几何受控实验。T0 直接复用当前 C0 same-grid D2 主结果，不重复评分；
# 本脚本运行 T1 curvature、T2 speed ratio、T3 path/chord、T4 D2+curvature、
# T5 四维 geometry descriptor。全部实验固定 same-grid、K=3、real-only Gaussian
# likelihood、Global STALL、CDF 和 0.60/0.40 融合，只改变 Local dynamics 表示。
#
# 用法：
#   bash scripts/run_trajectory.sh
#   bash scripts/run_trajectory.sh --dry-run
#   bash scripts/run_trajectory.sh --variant t1
#   bash scripts/run_trajectory.sh --variant t5 --overwrite
#   bash scripts/run_trajectory.sh --variant t4 --set runtime.score_batch_size=8
#
# 可传参数：
#   --variant {t1|t2|t3|t4|t5|all}  只跑指定实验，默认 all。
#   其他参数原样传给 run_experiment.py，仅用于 --dry-run、--overwrite 和 runtime 资源。
# 禁止在本入口覆盖 correspondence、采样、数据或融合定义。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="all"
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant)
      VARIANT="$2"
      shift 2
      ;;
    *)
      PASSTHROUGH+=("$1")
      shift
      ;;
  esac
done

if [[ ! "$VARIANT" =~ ^(t1|t2|t3|t4|t5|all)$ ]]; then
  echo "--variant 只能是 t1、t2、t3、t4、t5 或 all" >&2
  exit 2
fi

run_variant() {
  local variant="$1"
  local dynamics=""
  case "$variant" in
    t1) dynamics="curvature" ;;
    t2) dynamics="speed_ratio" ;;
    t3) dynamics="path_chord" ;;
    t4) dynamics="d2_curvature" ;;
    t5) dynamics="geometry" ;;
  esac
  local run_name="stage2_${variant}_${dynamics}_k3"
  echo "[Stage 2] 开始 ${run_name}" >&2
  conda run --no-capture-output -n "${STALL_ENV:-stall}" \
    python "$ROOT/scripts/run_experiment.py" \
      --run-name "$run_name" \
      --set runtime.device=cuda:1 \
      --set 'runtime.devices=[cuda:1]' \
      --set method.local.correspondence.type=same_grid \
      --set method.local.correspondence.confidence=none \
      --set "method.local.dynamics=${dynamics}" \
      --set runtime.reuse_global_run=alpha_stall_full_d2_k3_no_spatial_refit \
      "${PASSTHROUGH[@]}"
  echo "[Stage 2] 完成 ${run_name}" >&2
}

if [[ "$VARIANT" == "all" ]]; then
  for item in t1 t2 t3 t4 t5; do
    run_variant "$item"
  done
else
  run_variant "$VARIANT"
fi
