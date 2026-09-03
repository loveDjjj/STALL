#!/usr/bin/env bash
set -euo pipefail

# CAES固定检测参考：从C0完全相同的200 calibration IDs和FS0 K=3窗口
# 重拟合empirical Local D2，逐窗口核对source raw score后保存mu/W/CDF。
# FS1-FS5必须共享该参数，确保第一轮只改变窗口位置。
#
# 用法：bash scripts/freeze_caes_detector_reference.sh
# 可传：--device、--output-dir、--datasets、--overwrite。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda run --no-capture-output -n "${STALL_ENV:-stall}" \
  python "$ROOT/scripts/freeze_caes_detector_reference.py" "$@"
