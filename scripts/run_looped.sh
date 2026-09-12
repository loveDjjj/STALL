#!/usr/bin/env bash
# 十模型共享批次全矩阵：后台启动、检查状态、停止或汇总，所有大资产留在/data。
# 示例：bash scripts/run_looped.sh start；再次start会检查现有父/子进程，不重复启动。
# 停止后同命令start从各自优化器更新检查点恢复，不重提Patch、不减少epoch。
set -euo pipefail
LOOPED_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LOOPED_ROOT"
if [ "$#" -eq 0 ]; then set -- status; fi
exec /home/ubuntu/anaconda3/envs/stall/bin/python scripts/looped_control.py "$@"
