#!/usr/bin/env bash
set -euo pipefail

# 等待Tail likelihood field构建成功后，自动运行三selector的Tail与crossfit评测。
# 主任务若未以completed结束，本脚本直接返回非零状态，不自动重跑。
#
# 用法：
#   bash scripts/run_caes_tail_post.sh <tail-field-runner-pid>

if [[ $# -ne 1 ]]; then
  echo "用法：bash scripts/run_caes_tail_post.sh <tail-field-runner-pid>" >&2
  exit 2
fi

RUNNER_PID="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROGRESS="$ROOT/results/caes/tail_fields_v1/progress.json"

echo "[等待] Tail field runner PID=$RUNNER_PID"
while kill -0 "$RUNNER_PID" 2>/dev/null; do
  sleep 30
done

if ! rg -q '"status": "completed"' "$PROGRESS"; then
  echo "[拒绝] Tail field构建未以completed结束：$PROGRESS" >&2
  exit 3
fi

echo "[开始] Tail field已完成，评估三selector×四聚合×两种calibration"
bash "$ROOT/scripts/run_caes_tail.sh" --overwrite
echo "[开始] 计算Tail与crossfit主对照的配对bootstrap"
conda run --no-capture-output -n "${STALL_ENV:-stall}" \
  python "$ROOT/scripts/analyze_caes_tail.py"
