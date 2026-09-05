#!/usr/bin/env bash
set -euo pipefail

# GenVidBench Pair-1 外部确认实验：对比 Uniform K=3 与指定窗口selector。
# Feature-change冻结开发阶段确定的规则；Random使用固定seed 17作为随机对照。
# 全程不使用生成视频拟合参数，
# Local D2 仅由 199 条独立 VRiPT 真实视频重建，评测为 300 real + 300 fake。
#
# 可传参数：
#   --device cuda:1       指定特征提取和评分 GPU，默认 cuda:1。
#   --selector NAME       feature_change或random，默认feature_change。
#   --overwrite           删除并重建窗口清单、Local 参考和 dense 结果。
#   --resume              从已有 dense raw shard 继续；默认发现未完成结果时使用。
#
# 示例：
#   bash scripts/run_caes_external_feature_change.sh
#   bash scripts/run_caes_external_feature_change.sh --selector random
#   bash scripts/run_caes_external_feature_change.sh --device cuda:0 --overwrite

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="cuda:1"
SELECTOR="feature_change"
OVERWRITE=0
RESUME=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      DEVICE="${2:?--device 需要设备名，例如 cuda:1}"
      shift 2
      ;;
    --selector)
      SELECTOR="${2:?--selector 需要 feature_change 或 random}"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    *)
      echo "未知参数：$1" >&2
      exit 2
      ;;
  esac
done
if [[ "$SELECTOR" != "feature_change" && "$SELECTOR" != "random" ]]; then
  echo "--selector 仅支持 feature_change 或 random" >&2
  exit 2
fi

ENVIRONMENT="${STALL_ENV:-stall}"
SOURCE_RUN="alpha_stall_external_genvidbench"
SOURCE_CONFIG="$ROOT/results/runs/$SOURCE_RUN/resolved_config.yaml"
WINDOW_DIR="$ROOT/results/caes/external_genvidbench_${SELECTOR}_seed17"
REFERENCE_DIR="$ROOT/precomputed/caes_detector_external"
OUTPUT_DIR="$ROOT/results/runs/caes_external_genvidbench_${SELECTOR}"

echo "[1/4] 审计并补齐 GenVidBench 1 FPS Global coarse cache"
conda run --no-capture-output -n "$ENVIRONMENT" \
  python "$ROOT/scripts/build_coarse_global_cache.py" \
  --scope external --device "$DEVICE" --frame-batch-size 64 \
  --video-batch-size 16 --workers 8

WINDOW_ARGS=()
REFERENCE_ARGS=()
DENSE_ARGS=()
if [[ "$OVERWRITE" -eq 1 ]]; then
  WINDOW_ARGS+=(--overwrite)
  REFERENCE_ARGS+=(--overwrite)
  DENSE_ARGS+=(--overwrite)
elif [[ "$RESUME" -eq 1 || -d "$OUTPUT_DIR/raw_shards" ]]; then
  DENSE_ARGS+=(--resume)
fi

echo "[2/4] 生成冻结 Uniform/${SELECTOR} K=3 窗口清单"
if [[ ! -f "$WINDOW_DIR/run_manifest.json" || "$OVERWRITE" -eq 1 ]]; then
  conda run --no-capture-output -n "$ENVIRONMENT" \
    python "$ROOT/scripts/run_window_selection.py" \
    --config "$SOURCE_CONFIG" --manifest-scope external \
    --datasets genvidbench --selectors uniform "$SELECTOR" \
    --output-dir "$WINDOW_DIR" --device "$DEVICE" "${WINDOW_ARGS[@]}"
else
  echo "[复用] $WINDOW_DIR"
fi

echo "[3/4] 冻结 GenVidBench real-only Local D2 参考"
if [[ ! -f "$REFERENCE_DIR/genvidbench_local_d2.npz" || "$OVERWRITE" -eq 1 ]]; then
  conda run --no-capture-output -n "$ENVIRONMENT" \
    python "$ROOT/scripts/freeze_caes_detector_reference.py" \
    --source-run "$SOURCE_RUN" --manifest-scope external \
    --datasets genvidbench --output-dir "$REFERENCE_DIR" \
    --device "$DEVICE" "${REFERENCE_ARGS[@]}"
else
  echo "[复用] $REFERENCE_DIR/genvidbench_local_d2.npz"
fi

echo "[4/4] 执行严格视频身份配对的外部 dense 评分"
if [[ -f "$OUTPUT_DIR/progress.json" ]] \
  && grep -q '"status": "completed"' "$OUTPUT_DIR/progress.json" \
  && [[ "$OVERWRITE" -eq 0 ]]; then
  echo "[完成/复用] $OUTPUT_DIR"
else
  conda run --no-capture-output -n "$ENVIRONMENT" \
    python "$ROOT/scripts/run_caes_dense.py" \
    --config "$SOURCE_CONFIG" --source-run "$SOURCE_RUN" \
    --manifest-scope external --datasets genvidbench \
    --selectors uniform "$SELECTOR" --window-dir "$WINDOW_DIR" \
    --detector-reference-dir "$REFERENCE_DIR" --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" --chunk-videos 64 --decode-workers 8 \
    "${DENSE_ARGS[@]}"
fi

echo "[结果] $OUTPUT_DIR/dataset_metrics.csv"
