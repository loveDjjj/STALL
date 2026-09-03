#!/usr/bin/env bash
set -euo pipefail

# CAES FS0-FS5 WindowManifest生成：一次读取1 FPS Global缓存，只用每数据集
# 200条互斥calibration real拟合selector reference，并对calibration/test执行
# 完全相同的候选生成与selector。只写signals和窗口清单，不运行dense detector。
#
# 用法：
#   bash scripts/run_window_selection.sh --datasets comgenvid --limit-evaluation 20
#   bash scripts/run_window_selection.sh
#   bash scripts/run_window_selection.sh --overwrite
#
# 可传：--config、--cache-dir、--output-dir、--device、--datasets、
# --limit-evaluation、--overwrite。不得传fake-guided阈值或selector权重。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda run --no-capture-output -n "${STALL_ENV:-stall}" \
  python "$ROOT/scripts/run_window_selection.py" "$@"
