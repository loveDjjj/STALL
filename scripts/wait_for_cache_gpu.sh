#!/usr/bin/env bash
set -euo pipefail

# 等待 GPU 0 具备安全空闲显存后启动严格缓存重建。
# 此脚本不会终止、暂停或抢占其他用户任务；未达到阈值时仅每分钟重试一次。
# 用法：bash scripts/wait_for_cache_gpu.sh [最低空闲 GiB，默认 12]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MINIMUM_FREE_GIB="${1:-12}"
MINIMUM_FREE_MIB="$(awk -v gib="$MINIMUM_FREE_GIB" 'BEGIN { printf "%d", gib * 1024 }')"

while true; do
  FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d '[:space:]')"
  if [[ "$FREE_MIB" =~ ^[0-9]+$ ]] && (( FREE_MIB >= MINIMUM_FREE_MIB )); then
    echo "$(date '+%F %T') GPU 0 空闲 ${FREE_MIB}MiB，开始缓存重建。"
    exec bash "$ROOT/scripts/run_cache_rebuild.sh" \
      --device cuda:0 \
      --minimum-free-gib "$MINIMUM_FREE_GIB"
  fi
  echo "$(date '+%F %T') GPU 0 空闲 ${FREE_MIB:-未知}MiB，等待至少 ${MINIMUM_FREE_MIB}MiB。"
  sleep 60
done
