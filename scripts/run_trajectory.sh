#!/usr/bin/env bash
set -euo pipefail

# Stage 2 轨迹几何受控矩阵：一次读取 Patch cache，同时运行 T1 curvature、
# T2 speed ratio、T3 path/chord、T4 D2+curvature 与 T5 四维 geometry likelihood。
# T0 直接复用当前 C0 same-grid D2 主结果；Global 窗口分数从该 completed run
# 按唯一窗口身份严格复用，避免五次重复读取约 630GB 缓存和重复高维评分。
#
# 用法：
#   bash scripts/run_trajectory.sh
#   bash scripts/run_trajectory.sh --dry-run
#   bash scripts/run_trajectory.sh --overwrite
#   bash scripts/run_trajectory.sh --set runtime.score_batch_size=64
#
# 可传参数：--dry-run、--overwrite，以及 runtime 下的资源型 --set 覆盖。
# 方法、采样、数据、融合和 correspondence 由矩阵入口锁定，不能逐候选调整。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda run --no-capture-output -n "${STALL_ENV:-stall}" \
  python "$ROOT/scripts/run_trajectory_matrix.py" "$@"
