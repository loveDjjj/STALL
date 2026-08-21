#!/usr/bin/env bash
set -euo pipefail

# Alpha STALL 默认主方法：使用 benchmark.yaml 中的 Global + Local D2 + K=3 配置。
# 可透传 runner 参数：--dry-run（只检查计划）、--overwrite（明确覆盖同名未冻结结果）、
# --set KEY=VALUE（覆盖运行资源或受控实验参数）、--scores-csv PATH（导入外部分数）。
# 单卡示例：bash scripts/run_alpha_stall.sh --set runtime.device=cuda:1
# 双卡示例：bash scripts/run_alpha_stall.sh --set 'runtime.devices=[cuda:0,cuda:1]'
# 吞吐示例：bash scripts/run_alpha_stall.sh --set runtime.score_batch_size=16 --set runtime.cache_io_workers=1
# 汇总恢复：bash scripts/run_alpha_stall.sh --resume-bootstrap（仅补跑中断运行的 bootstrap，不重读缓存）
# 请勿用此入口改变 sampling、method 或 data；对应对比实验应使用 run_ablation.sh 等专用脚本。
# 用法示例：bash scripts/run_alpha_stall.sh --dry-run
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT/results/runs/alpha_stall"
CHILD_PID=""
RECEIVED_SIGNAL=""
DRY_RUN=false

# 运行器自身会记录 Python 异常；此处补齐 shell/conda 层的 PID、退出码和可捕获信号。
# SIGKILL 无法被任何进程捕获，但其余异常退出均可据此与 runner 日志交叉定位。
write_launcher_status() {
  local exit_code="$1"
  if [[ "$DRY_RUN" == true ]]; then
    return
  fi
  local status="completed"
  if [[ "$exit_code" -ne 0 ]]; then
    status="failed"
  fi
  if [[ -n "$RECEIVED_SIGNAL" ]]; then
    status="terminated"
  fi
  mkdir -p "$RUN_DIR"
  printf '{\n  "status": "%s",\n  "exit_code": %s,\n  "signal": "%s",\n  "launcher_pid": %s,\n  "child_pid": %s,\n  "finished_at_utc": "%s"\n}\n' \
    "$status" "$exit_code" "$RECEIVED_SIGNAL" "$$" "${CHILD_PID:-null}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "$RUN_DIR/launcher_status.json"
}

on_signal() {
  RECEIVED_SIGNAL="$1"
  if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
    kill -"$1" "$CHILD_PID" 2>/dev/null || true
    wait "$CHILD_PID" || true
  fi
  exit 128
}

on_exit() {
  local exit_code="$?"
  trap - EXIT
  write_launcher_status "$exit_code"
  exit "$exit_code"
}

trap 'on_signal HUP' HUP
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap on_exit EXIT
for argument in "$@"; do
  if [[ "$argument" == "--dry-run" ]]; then
    DRY_RUN=true
  fi
  if [[ "$argument" == "--run-name" || "$argument" == --run-name=* ]]; then
    echo "run_alpha_stall.sh 固定使用运行名 alpha_stall，请不要传入 --run-name" >&2
    exit 2
  fi
done
conda run --no-capture-output -n "${STALL_ENV:-stall}" python "$ROOT/scripts/run_experiment.py" --run-name alpha_stall "$@" &
CHILD_PID="$!"
wait "$CHILD_PID"
