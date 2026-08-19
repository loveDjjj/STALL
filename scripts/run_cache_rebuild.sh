#!/usr/bin/env bash
set -euo pipefail

# 按当前全部 manifest 重建严格的完整 8 FPS DINOv3 Global+patch 缓存。
# 默认要求目标 GPU 至少空闲 12 GiB；显存不足会安全退出，不会抢占正在运行的任务。
# 用法：bash scripts/run_cache_rebuild.sh [rebuild_patch_cache.py 参数]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda run --no-capture-output -n "${STALL_ENV:-stall}" python "$ROOT/scripts/rebuild_patch_cache.py" \
  --allow-short-videos \
  "$@"
