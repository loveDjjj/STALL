#!/usr/bin/env bash
set -euo pipefail

# CAES Stage FS dense评分：读取FS1-FS5 WindowManifest，对每视频selected窗口
# 并集按需提取DINO Global+Patch，使用冻结C0 Local D2参数评分，并按selector
# 分别执行matched real calibration。FS0直接复用C0正式分数。
#
# 用法：
#   bash scripts/run_caes_dense.sh --datasets comgenvid --output-dir results/runs/caes_fs_smoke
#   bash scripts/run_caes_dense.sh
#   bash scripts/run_caes_dense.sh --resume
#   bash scripts/run_caes_dense.sh --overwrite
#
# 可传：--config、--window-dir、--output-dir、--detector-reference-dir、
# --device、--datasets、--chunk-videos、--decode-workers、--resume、--overwrite。
# 第一轮为保证与C0数值合同一致，--frame-batch-size固定8，
# --score-window-batch-size固定48，不接受其他值。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda run --no-capture-output -n "${STALL_ENV:-stall}" \
  python "$ROOT/scripts/run_caes_dense.py" "$@"
