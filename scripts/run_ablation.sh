#!/usr/bin/env bash
set -euo pipefail

# 我们方法的结构与时间覆盖消融。锁定 D2 变体复用主实验的固定 Local 参数与独立 K=1 CDF；
# D1 和 K=1 没有对应锁定资产，必须使用互斥真实 calibration 重拟合，并在运行名中显式标明。
# 可透传：--dry-run、--overwrite、--set runtime.device=cuda:1、
# --set 'runtime.devices=[cuda:0,cuda:1]'、--set runtime.score_batch_size=16、--set runtime.cache_io_workers=1。
# 用法示例：bash scripts/run_ablation.sh local_d1_refit --dry-run
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 ]]; then
  echo "用法：run_ablation.sh <global_only|global_only_k3_refit|local_spatial_only_refit|local_d1_refit|local_d2_locked|local_d2_refit|local_spatial_d2_refit|full_d1_refit|full_d1_k3_no_spatial_refit|full_d2_k3_refit|full_d2_k3_no_spatial_refit|full_k1_refit|full_d2_k1_no_spatial_refit|full_k3_locked> [runner 参数]" >&2
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
    SETS=(--set method.local.enabled=false --set method.local.parameter_source=locked_u0)
    RUN_NAME=alpha_stall_global_only
    ;;
  global_only_k3_refit)
    # 与全部 refit 消融共用通用 K=3 窗口；Local 关闭，因此不会拟合或评分局部参数。
    SETS=(--set method.local.enabled=false --set method.local.parameter_source=fit_real_only --set sampling.num_windows=3)
    RUN_NAME=alpha_stall_global_only_k3_refit
    ;;
  local_spatial_only_refit)
    # 仅保留 Local Spatial，用于测量单帧 patch 外观证据的独立检测能力。
    SETS=(--set method.global.enabled=false --set method.local.parameter_source=fit_real_only --set method.local.spatial_enabled=true --set method.local.temporal_enabled=false --set method.local.spatial_weight=1.0 --set method.local.temporal_weight=0.0 --set sampling.num_windows=3)
    RUN_NAME=alpha_stall_local_spatial_only_refit
    ;;
  local_d1_refit)
    SETS=(--set method.global.enabled=false --set method.local.parameter_source=fit_real_only --set method.local.temporal_order=1 --set method.local.spatial_enabled=false)
    RUN_NAME=alpha_stall_local_d1_refit
    ;;
  local_d2_locked)
    # 只移除 Global；D2 参数、K=3 锁定帧索引与 Local K=1 CDF 均与主实验一致。
    SETS=(--set method.global.enabled=false --set method.local.parameter_source=locked_u0 --set method.local.temporal_order=2 --set method.local.spatial_enabled=false)
    RUN_NAME=alpha_stall_local_d2_locked
    ;;
  local_d2_refit)
    # 与 local_d1_refit 共用通用 K=3 窗口和 real-only 参数拟合，只将时序阶数改为 D2。
    SETS=(--set method.global.enabled=false --set method.local.parameter_source=fit_real_only --set method.local.temporal_order=2 --set method.local.spatial_enabled=false)
    RUN_NAME=alpha_stall_local_d2_refit
    ;;
  local_spatial_d2_refit)
    # 与 local_d2_refit 相比只启用 Local Spatial，检验 0.1/0.9 局部融合是否必要。
    SETS=(--set method.global.enabled=false --set method.local.parameter_source=fit_real_only --set method.local.temporal_order=2 --set method.local.spatial_enabled=true --set method.local.temporal_enabled=true --set method.local.spatial_weight=0.1 --set method.local.temporal_weight=0.9 --set sampling.num_windows=3)
    RUN_NAME=alpha_stall_local_spatial_d2_refit
    ;;
  full_d1_refit)
    # 与完整方法相比只改用 D1；因无锁定 D1 参数，Local 分支由 calibration real 重拟合。
    SETS=(--set method.local.parameter_source=fit_real_only --set method.local.temporal_order=1 --set method.local.spatial_enabled=true --set method.local.spatial_weight=0.1 --set method.local.temporal_weight=0.9)
    RUN_NAME=alpha_stall_full_d1_refit
    ;;
  full_d1_k3_no_spatial_refit)
    # 新正式方法的 D1 对照：与无 Spatial D2 主方法相比只改变时序差分阶数。
    SETS=(--set method.local.parameter_source=fit_real_only --set method.local.temporal_order=1 --set method.local.spatial_enabled=false --set method.local.temporal_enabled=true --set method.local.spatial_weight=0.0 --set method.local.temporal_weight=1.0 --set sampling.num_windows=3)
    RUN_NAME=alpha_stall_full_d1_k3_no_spatial_refit
    ;;
  full_d2_k3_refit)
    # 与 full_d1_refit、full_k1_refit 共用 real-only 拟合协议；此项固定 D2 和 K=3。
    # 它分别是 D2-vs-D1 与 K=3-vs-K1 的受控参照，不能与 locked U0 主表混称同一协议。
    SETS=(--set method.local.parameter_source=fit_real_only --set method.local.temporal_order=2 --set method.local.spatial_enabled=true --set method.local.spatial_weight=0.1 --set method.local.temporal_weight=0.9 --set sampling.num_windows=3)
    RUN_NAME=alpha_stall_full_d2_k3_refit
    ;;
  full_d2_k3_no_spatial_refit)
    # 与 full_d2_k3_refit 相比只关闭 Local Spatial；这是决定最终方法是否保留该分支的主对照。
    SETS=(--set method.local.parameter_source=fit_real_only --set method.local.temporal_order=2 --set method.local.spatial_enabled=false --set method.local.temporal_enabled=true --set sampling.num_windows=3)
    RUN_NAME=alpha_stall_full_d2_k3_no_spatial_refit
    ;;
  full_k1_refit)
    # 当前仓库没有历史锁定 K=1 evaluation 帧索引，故这是可复现的 K=1 重拟合参考，
    # 不可标注为“仅改变 K”的严格锁定因子对照。
    SETS=(--set method.local.parameter_source=fit_real_only --set method.local.spatial_enabled=true --set method.local.spatial_weight=0.1 --set method.local.temporal_weight=0.9 --set sampling.num_windows=1)
    RUN_NAME=alpha_stall_full_k1_refit
    ;;
  full_d2_k1_no_spatial_refit)
    # 新正式方法的 K=1 对照：与无 Spatial K=3 主方法相比只改变窗口数量。
    SETS=(--set method.local.parameter_source=fit_real_only --set method.local.temporal_order=2 --set method.local.spatial_enabled=false --set method.local.temporal_enabled=true --set method.local.spatial_weight=0.0 --set method.local.temporal_weight=1.0 --set sampling.num_windows=1)
    RUN_NAME=alpha_stall_full_d2_k1_no_spatial_refit
    ;;
  full_k3_locked)
    # 已完成的 alpha_stall_locked_u0 就是该配置；仅在明确需要独立重复时才运行。
    SETS=(--set method.local.parameter_source=locked_u0 --set method.local.spatial_enabled=true --set method.local.spatial_weight=0.1 --set method.local.temporal_weight=0.9 --set sampling.num_windows=3)
    RUN_NAME=alpha_stall_full_k3_locked
    ;;
  *)
    echo "未知消融名称：$VARIANT" >&2
    exit 2
    ;;
esac

conda run --no-capture-output -n "${STALL_ENV:-stall}" python "$ROOT/scripts/run_experiment.py" --run-name "$RUN_NAME" "${SETS[@]}" "$@"
