#!/usr/bin/env bash
set -euo pipefail

# Stage 1 局部对应受控实验：依次运行 C0 same-grid D2、C1 hard local D2、
# C2 soft local D2、C3 soft local D2 + entropy confidence likelihood 聚合。
# 四组实验共享 DINO 缓存、数据划分、seed、K=3、Global、Local 白化、CDF 和融合权重，
# 只覆盖 method.local.correspondence 字段；默认使用 GPU 1，避免占用当前 GPU 0。
#
# 用法：
#   bash scripts/run_correspondence.sh
#   bash scripts/run_correspondence.sh --dry-run
#   bash scripts/run_correspondence.sh --variant c2 --overwrite
#   bash scripts/run_correspondence.sh --variant c3 --set runtime.score_batch_size=8
#
# 可传参数：
#   --variant {c0|c1|c2|c3|all}  只跑指定实验，默认 all。
#   --dry-run                       只验证配置与运行计划。
#   --overwrite                     覆盖同名未冻结结果。
#   --set KEY=VALUE                 透传运行资源覆盖；不要覆盖方法、采样或数据定义。
#   --radius {1|2}                  覆盖 hard/soft 搜索半径，默认 1；正式 C0-C3 固定 1。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="all"
RADIUS="1"
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant)
      VARIANT="$2"
      shift 2
      ;;
    --radius)
      RADIUS="$2"
      shift 2
      ;;
    *)
      PASSTHROUGH+=("$1")
      shift
      ;;
  esac
done

if [[ ! "$VARIANT" =~ ^(c0|c1|c2|c3|all)$ ]]; then
  echo "--variant 只能是 c0、c1、c2、c3 或 all" >&2
  exit 2
fi
if [[ ! "$RADIUS" =~ ^(1|2)$ ]]; then
  echo "--radius 只能是 1 或 2" >&2
  exit 2
fi

run_variant() {
  local variant="$1"
  local mode="same_grid"
  local confidence="none"
  case "$variant" in
    c0) mode="same_grid" ;;
    c1) mode="hard_local" ;;
    c2) mode="soft_local" ;;
    c3) mode="soft_local"; confidence="aggregation" ;;
  esac
  local run_name="stage1_${variant}_${mode}_d2_k3"
  echo "[Stage 1] 开始 ${run_name}" >&2
  conda run --no-capture-output -n "${STALL_ENV:-stall}" \
    python "$ROOT/scripts/run_experiment.py" \
      --run-name "$run_name" \
      --set runtime.device=cuda:1 \
      --set 'runtime.devices=[cuda:1]' \
      --set "method.local.correspondence.type=${mode}" \
      --set "method.local.correspondence.radius=${RADIUS}" \
      --set "method.local.correspondence.confidence=${confidence}" \
      "${PASSTHROUGH[@]}"
  echo "[Stage 1] 完成 ${run_name}" >&2
}

if [[ "$VARIANT" == "all" ]]; then
  for item in c0 c1 c2 c3; do
    run_variant "$item"
  done
else
  run_variant "$VARIANT"
fi
