#!/usr/bin/env bash
set -euo pipefail

# 为Uniform、Feature-change和Real-anomaly生成5-fold OOF calibration选窗清单。
# Uniform与Feature-change不依赖真实reference，必须逐位复现standard，作为控制；
# Real-anomaly每折只使用另外160条真实视频拟合selector reference。
#
# 用法：
#   bash scripts/run_caes_crossfit.sh
#   bash scripts/run_caes_crossfit.sh --datasets comgenvid --overwrite
#
# 可传：--config、--standard-window-dir、--output-dir、--cache-dir、
# --device、--datasets、--overwrite。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda run --no-capture-output -n "${STALL_ENV:-stall}" \
  python "$ROOT/scripts/run_caes_crossfit_manifests.py" "$@"
