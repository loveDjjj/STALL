#!/usr/bin/env bash
set -euo pipefail

# 将严格 K=3 单视频缓存迁移为 32 视频顺序 shard，改善机械盘随机读取。
# 每个 shard 均先写入、读回校验并更新 index，之后才删除该 shard 对应的旧 .pt 与 metadata；可安全续跑。
# 用法：bash scripts/run_cache_pack.sh --dry-run
# 实际迁移：bash scripts/run_cache_pack.sh
# 小规模验证：bash scripts/run_cache_pack.sh --limit-shards 1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda run --no-capture-output -n "${STALL_ENV:-stall}" python "$ROOT/scripts/pack_patch_cache.py" --shard-size 32 "$@"
