#!/usr/bin/env bash
set -euo pipefail

# 为Uniform、Feature-change、Real-anomaly及Real-anomaly OOF calibration构建
# Local D2位置级likelihood field。只保存约14×14×14的float32似然，不保存Patch token。
#
# 用法：
#   bash scripts/run_caes_tail_fields.sh
#   bash scripts/run_caes_tail_fields.sh --resume
#   bash scripts/run_caes_tail_fields.sh --datasets comgenvid --limit-evaluation-per-source 5 --output-dir results/caes/tail_fields_smoke
#
# 可传：--config、--standard-window-dir、--crossfit-window-dir、
# --detector-reference-dir、--output-dir、--device、--datasets、--chunk-videos、
# --decode-workers、--limit-evaluation-per-source、--resume、--overwrite。
# 数值合同固定frame batch=8、score window batch=48。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda run --no-capture-output -n "${STALL_ENV:-stall}" \
  python "$ROOT/scripts/build_caes_tail_fields.py" "$@"
