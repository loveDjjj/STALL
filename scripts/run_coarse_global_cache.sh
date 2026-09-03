#!/usr/bin/env bash
set -euo pipefail

# CAES 1 FPS Global-only缓存构建：从三个development benchmark完整视频提取
# DINOv3最终Global token，以float16保存到独立cache/coarse_global_1fps。
# 不读取/修改现有630GB Patch cache，也不保存Patch token。
#
# 用法：
#   bash scripts/run_coarse_global_cache.sh --audit-only
#   bash scripts/run_coarse_global_cache.sh
#   bash scripts/run_coarse_global_cache.sh --limit 20
#   bash scripts/run_coarse_global_cache.sh --scope all
#   bash scripts/run_coarse_global_cache.sh --shard-index 0 --shard-count 2
#
# 可传参数：--scope、--cache-dir、--device、--frame-batch-size、
# --video-batch-size、--workers、--shard-index/count、--limit、--audit-only。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda run --no-capture-output -n "${STALL_ENV:-stall}" \
  python "$ROOT/scripts/build_coarse_global_cache.py" "$@"
