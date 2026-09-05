#!/usr/bin/env bash
set -euo pipefail

# 四域跨真实校准矩阵：冻结Feature-change K3与0.5/0.5最终融合，分别使用
# ComGenVid、VideoFeedback、GenVideo、GenVidBench real bank，以及等域贡献的
# Universal-3/4 bank，交叉评测四域并报告实际real FPR与fake recall。
# 矩阵完成后继续运行不含ViF real的Universal-4 -> ViF-Bench严格迁移。
#
# 参数：--device cuda:1、--resume、--overwrite。
# 示例：bash scripts/run_cross_domain_calibration.sh --device cuda:1 --resume

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="cuda:1"
MODE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="${2:?--device需要设备名}"; shift 2 ;;
    --resume) MODE="--resume"; shift ;;
    --overwrite) MODE="--overwrite"; shift ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done

ENVIRONMENT="${STALL_ENV:-stall}"
RUN_DIR="$ROOT/results/runs/cross_domain_feature_k3_equal_fusion"
VIF_DIR="$ROOT/results/runs/universal4_to_vifbench_feature_k3"

if [[ -f "$RUN_DIR/run_manifest.json" ]] \
  && grep -q '"status": "completed"' "$RUN_DIR/run_manifest.json" \
  && [[ "$MODE" != "--overwrite" ]]; then
  echo "[完成/复用] $RUN_DIR"
else
  conda run --no-capture-output -n "$ENVIRONMENT" \
    python "$ROOT/scripts/run_cross_domain_calibration.py" \
    --output-dir "$RUN_DIR" --device "$DEVICE" --chunk-videos 64 \
    --decode-workers 8 ${MODE:---resume}
fi

conda run --no-capture-output -n "$ENVIRONMENT" \
  python "$ROOT/scripts/finalize_cross_domain_calibration.py" \
  --run-dir "$RUN_DIR"

VIF_MODE="$MODE"
if [[ -z "$VIF_MODE" ]]; then
  VIF_MODE="--resume"
fi
if [[ -f "$VIF_DIR/run_manifest.json" ]] \
  && grep -q '"status": "completed"' "$VIF_DIR/run_manifest.json" \
  && [[ "$MODE" != "--overwrite" ]]; then
  echo "[完成/复用] $VIF_DIR"
else
  conda run --no-capture-output -n "$ENVIRONMENT" \
    python "$ROOT/scripts/run_universal_vifbench_transfer.py" \
    --cross-domain-dir "$RUN_DIR" --output-dir "$VIF_DIR" \
    --device "$DEVICE" --chunk-videos 64 --decode-workers 8 "$VIF_MODE"
fi

conda run --no-capture-output -n "$ENVIRONMENT" \
  python "$ROOT/scripts/analyze_cross_domain_results.py" --overwrite
conda run --no-capture-output -n "$ENVIRONMENT" \
  python "$ROOT/scripts/analyze_universal_vifbench_transfer.py" --overwrite
