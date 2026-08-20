#!/usr/bin/env bash
set -euo pipefail

# 我们方法的结构与时间覆盖消融：每个 variant 仅改变下方明确列出的基础配置字段。
# 可透传：--dry-run、--overwrite、--set runtime.device=cuda:1、
# --set 'runtime.devices=[cuda:0,cuda:1]'、--set runtime.score_batch_size=16、--set runtime.cache_io_workers=4。
# 用法示例：bash scripts/run_ablation.sh local_d1 --dry-run
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 ]]; then
  echo "用法：run_ablation.sh <global_only|local_d1|local_d2|combined_local_d2|global_k1|full_k1|full_k3> [runner 参数]" >&2
  exit 2
fi
VARIANT="$1"
shift
for argument in "$@"; do
  if [[ "$argument" == "--run-name" || "$argument" == --run-name=* ]]; then
    echo "run_ablation.sh 根据 variant 固定运行名，请不要传入 --run-name" >&2
    exit 2
  fi
done

case "$VARIANT" in
  global_only)
    SETS=(--set method.local.enabled=false)
    RUN_NAME=alpha_stall_global_only
    ;;
  local_d1)
    SETS=(--set method.global.enabled=false --set method.local.temporal_order=1 --set method.local.spatial_enabled=false)
    RUN_NAME=alpha_stall_local_d1
    ;;
  local_d2)
    SETS=(--set method.global.enabled=false --set method.local.temporal_order=2 --set method.local.spatial_enabled=false)
    RUN_NAME=alpha_stall_local_d2
    ;;
  combined_local_d2)
    SETS=(--set method.local.temporal_order=2 --set method.local.spatial_enabled=true)
    RUN_NAME=alpha_stall_combined_local_d2
    ;;
  global_k1)
    SETS=(--set method.local.enabled=false --set sampling.num_windows=1)
    RUN_NAME=alpha_stall_global_k1
    ;;
  full_k1)
    SETS=(--set sampling.num_windows=1)
    RUN_NAME=alpha_stall_full_k1
    ;;
  full_k3)
    SETS=(--set sampling.num_windows=3)
    RUN_NAME=alpha_stall_full_k3
    ;;
  *)
    echo "未知消融名称：$VARIANT" >&2
    exit 2
    ;;
esac

conda run --no-capture-output -n "${STALL_ENV:-stall}" python "$ROOT/scripts/run_experiment.py" --run-name "$RUN_NAME" "${SETS[@]}" "$@"
