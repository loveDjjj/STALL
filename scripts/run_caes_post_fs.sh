#!/usr/bin/env bash
set -euo pipefail

# CAES Stage FS 后处理守护脚本。
# 等待指定 dense runner PID 结束；只有主任务 progress.json 明确写入 completed，
# 才使用释放后的 GPU 执行 FS0 on-demand/strict-cache 等价审计。
# 主任务若中断或失败，本脚本返回非零状态，绝不自动重跑或跳过失败。
#
# 用法：
#   bash scripts/run_caes_post_fs.sh <dense-runner-pid> [cuda-device]

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "用法：bash scripts/run_caes_post_fs.sh <dense-runner-pid> [cuda-device]" >&2
  exit 2
fi

RUNNER_PID="$1"
DEVICE="${2:-cuda:1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROGRESS="$ROOT/results/runs/caes_stage_fs/progress.json"

echo "[等待] CAES dense runner PID=$RUNNER_PID"
while kill -0 "$RUNNER_PID" 2>/dev/null; do
  sleep 30
done

if ! rg -q '"status": "completed"' "$PROGRESS"; then
  echo "[拒绝] Stage FS没有以completed状态结束：$PROGRESS" >&2
  exit 3
fi

echo "[开始] Stage FS已完成，执行FS0数值等价审计"
conda run --no-capture-output -n "${STALL_ENV:-stall}" \
  python "$ROOT/scripts/verify_caes_fs0_equivalence.py" \
  --device "$DEVICE" \
  --overwrite
