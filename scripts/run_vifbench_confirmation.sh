#!/usr/bin/env bash
set -euo pipefail

# ViF-Bench 未见生成器冻结确认：准备官方 source_videos.zip，固定80条real校准，
# 对85条real与其1,531条语义配对fake运行Uniform/Feature-change/Random K3。
# 最终检测器固定为Global内部0.5/0.5、Global/Local最终0.5/0.5、Local D2。
# 本脚本不会下载8.23GB parsed_frames.zip，只使用约608MB官方MP4压缩包。
#
# 参数：
#   --device cuda:1    指定GPU，默认cuda:1。
#   --resume           从已有dense raw shard恢复。
#   --overwrite        重建manifest、Local参考和dense结果。
#
# 示例：bash scripts/run_vifbench_confirmation.sh --device cuda:1 --resume

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
DOWNLOAD="$ROOT/datasets/vifbench_download"
ARCHIVE="$DOWNLOAD/source_videos.zip"
SOURCE="$DOWNLOAD/source_videos"
MANIFEST_ROOT="$ROOT/data/manifests/confirmation"
WINDOW_DIR="$ROOT/results/caes/confirmation_vifbench_seed17"
REFERENCE_DIR="$ROOT/precomputed/caes_detector_confirmation"
OUTPUT_DIR="$ROOT/results/runs/caes_confirmation_vifbench"
RANDOM_OUTPUT_DIR="$ROOT/results/runs/caes_confirmation_vifbench_random"

mkdir -p "$DOWNLOAD"
if [[ ! -f "$ARCHIVE" ]]; then
  echo "[1/7] 下载ViF-Bench官方MP4压缩包"
  curl -L --retry 20 --retry-delay 5 --continue-at - \
    'https://hf-mirror.com/datasets/JoeLeelyf/ViF-Bench/resolve/main/source_videos.zip?download=true' \
    -o "$ARCHIVE"
fi
if [[ ! -d "$SOURCE" ]]; then
  echo "[2/7] 解包官方MP4"
  unzip -q "$ARCHIVE" -d "$DOWNLOAD"
fi

PREPARE_ARGS=()
WINDOW_ARGS=()
REFERENCE_ARGS=()
DENSE_ARGS=()
if [[ "$MODE" == "--overwrite" ]]; then
  PREPARE_ARGS+=(--overwrite)
  WINDOW_ARGS+=(--overwrite)
  REFERENCE_ARGS+=(--overwrite)
  DENSE_ARGS+=(--overwrite)
elif [[ "$MODE" == "--resume" ]]; then
  DENSE_ARGS+=(--resume)
elif [[ -d "$OUTPUT_DIR/raw_shards" ]]; then
  DENSE_ARGS+=(--resume)
fi

echo "[3/7] 审计数据并建立无配对泄漏manifest"
if [[ ! -f "$MANIFEST_ROOT/vifbench_protocol.json" || "$MODE" == "--overwrite" ]]; then
  conda run --no-capture-output -n "$ENVIRONMENT" \
    python "$ROOT/scripts/prepare_vifbench.py" "${PREPARE_ARGS[@]}"
else
  echo "[复用] $MANIFEST_ROOT/vifbench_protocol.json"
fi

echo "[4/7] 构建1 FPS Global coarse cache"
conda run --no-capture-output -n "$ENVIRONMENT" \
  python "$ROOT/scripts/build_coarse_global_cache.py" \
  --scope confirmation --device "$DEVICE" --frame-batch-size 64 \
  --video-batch-size 16 --workers 8

echo "[5/7] 冻结Uniform/Feature-change窗口清单"
if [[ ! -f "$WINDOW_DIR/run_manifest.json" || "$MODE" == "--overwrite" ]]; then
  conda run --no-capture-output -n "$ENVIRONMENT" \
    python "$ROOT/scripts/run_window_selection.py" \
    --config "$ROOT/configs/benchmark.yaml" --manifest-scope confirmation \
    --datasets vifbench --selectors uniform feature_change \
    --calibration-real-count 80 --output-dir "$WINDOW_DIR" \
    --device "$DEVICE" "${WINDOW_ARGS[@]}"
else
  echo "[复用] $WINDOW_DIR"
fi

echo "[6/7] 构建80-real Uniform K3 cache并冻结Local D2参考"
conda run --no-capture-output -n "$ENVIRONMENT" \
  python "$ROOT/scripts/rebuild_patch_cache.py" \
  --manifest "$MANIFEST_ROOT/vifbench_calibration.csv" \
  --cache-dir "$ROOT/cache/patch_embeddings_confirmation_vifbench" \
  --device "$DEVICE" --frame-batch-size 8 --video-batch-size 1 \
  --decode-workers 4 --skip-audit-report
if [[ ! -f "$REFERENCE_DIR/vifbench_local_d2.npz" || "$MODE" == "--overwrite" ]]; then
  conda run --no-capture-output -n "$ENVIRONMENT" \
    python "$ROOT/scripts/fit_confirmation_local_reference.py" \
    --cache-dir "$ROOT/cache/patch_embeddings_confirmation_vifbench" \
    --device "$DEVICE" "${REFERENCE_ARGS[@]}"
else
  echo "[复用] $REFERENCE_DIR/vifbench_local_d2.npz"
fi

echo "[7/7] 运行全量冻结确认并生成报告数据"
if [[ -f "$OUTPUT_DIR/progress.json" ]] \
  && grep -q '"status": "completed"' "$OUTPUT_DIR/progress.json" \
  && [[ "$MODE" != "--overwrite" ]]; then
  echo "[完成/复用] $OUTPUT_DIR"
else
  conda run --no-capture-output -n "$ENVIRONMENT" \
    python "$ROOT/scripts/run_caes_dense.py" \
    --config "$ROOT/configs/benchmark.yaml" --manifest-scope confirmation \
    --datasets vifbench --selectors uniform feature_change \
    --calibration-real-count 80 --fusion-global-weight 0.5 \
    --score-uniform-on-demand --window-dir "$WINDOW_DIR" \
    --detector-reference-dir "$REFERENCE_DIR" --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" --chunk-videos 64 --decode-workers 8 \
    "${DENSE_ARGS[@]}"
fi

echo "[附加对照] 运行固定seed 17的Random K3"
if [[ -f "$RANDOM_OUTPUT_DIR/progress.json" ]] \
  && grep -q '"status": "completed"' "$RANDOM_OUTPUT_DIR/progress.json" \
  && [[ "$MODE" != "--overwrite" ]]; then
  echo "[完成/复用] $RANDOM_OUTPUT_DIR"
else
  RANDOM_ARGS=()
  if [[ "$MODE" == "--overwrite" ]]; then
    RANDOM_ARGS+=(--overwrite)
  elif [[ -d "$RANDOM_OUTPUT_DIR/raw_shards" ]]; then
    RANDOM_ARGS+=(--resume)
  fi
  conda run --no-capture-output -n "$ENVIRONMENT" \
    python "$ROOT/scripts/run_caes_dense.py" \
    --config "$ROOT/configs/benchmark.yaml" --manifest-scope confirmation \
    --datasets vifbench --selectors uniform random \
    --calibration-real-count 80 --fusion-global-weight 0.5 \
    --score-uniform-on-demand --window-dir "$WINDOW_DIR" \
    --detector-reference-dir "$REFERENCE_DIR" --output-dir "$RANDOM_OUTPUT_DIR" \
    --device "$DEVICE" --chunk-videos 64 --decode-workers 8 \
    "${RANDOM_ARGS[@]}"
fi
conda run --no-capture-output -n "$ENVIRONMENT" \
  python "$ROOT/scripts/analyze_vifbench_confirmation.py" --overwrite
