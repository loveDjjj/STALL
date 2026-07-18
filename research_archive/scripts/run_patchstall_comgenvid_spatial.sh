#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/OneDay/STALL_project"
STALL_DIR="$ROOT/STALL"

CSV_PATH="$STALL_DIR/cache/indexes/comgenvid.csv"
PATCH_CACHE_DIR="$STALL_DIR/cache/patch_embeddings/comgenvid"
PATCH_PARAMS_PATH="$STALL_DIR/precomputed/patch_params_comgenvid_real.npz"
OUTPUT_CSV="$STALL_DIR/results/comgenvid_patch_spatial.csv"

NUM_WORKERS="${NUM_WORKERS:-8}"
VIDEO_BATCH="${VIDEO_BATCH:-4}"
FRAME_BATCH="${FRAME_BATCH:-32}"
MAX_PATCHES_FOR_FIT="${MAX_PATCHES_FOR_FIT:-300000}"

export TORCH_HOME="${TORCH_HOME:-/tmp/torch_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

echo "ROOT=$ROOT"
echo "STALL_DIR=$STALL_DIR"
echo "CSV_PATH=$CSV_PATH"
echo "PATCH_CACHE_DIR=$PATCH_CACHE_DIR"
echo "PATCH_PARAMS_PATH=$PATCH_PARAMS_PATH"
echo "OUTPUT_CSV=$OUTPUT_CSV"
echo "NUM_WORKERS=$NUM_WORKERS"
echo "VIDEO_BATCH=$VIDEO_BATCH"
echo "FRAME_BATCH=$FRAME_BATCH"
echo "MAX_PATCHES_FOR_FIT=$MAX_PATCHES_FOR_FIT"
echo "TORCH_HOME=$TORCH_HOME"
echo "XDG_CACHE_HOME=$XDG_CACHE_HOME"
echo

cd "$ROOT"
source /home/ubuntu/anaconda3/bin/activate stall

mkdir -p "$PATCH_CACHE_DIR" "$STALL_DIR/precomputed" "$STALL_DIR/results"

echo "== Step 1/3: Build patch cache =="
python - <<'PY'
import os
import sys
sys.path.insert(0, 'STALL/src')

from tqdm import tqdm

from stall_patch import PatchSTALL
from dataset_utils_patch import count_patch_cache_misses, prefill_patch_emb_cache

csv_path = 'STALL/cache/indexes/comgenvid.csv'
cache_dir = 'STALL/cache/patch_embeddings/comgenvid'
num_workers = int(os.environ.get('NUM_WORKERS', '8'))
video_batch = int(os.environ.get('VIDEO_BATCH', '4'))
frame_batch = int(os.environ.get('FRAME_BATCH', '32'))

model = PatchSTALL(device='cuda', data_dict=None, load_dino=True)
misses = count_patch_cache_misses(csv_path, cache_dir, duration_sec=2, compact=True)
print(f'Patch cache build: {misses} misses', flush=True)

for _ in tqdm(
    prefill_patch_emb_cache(
        csv_path,
        cache_dir,
        model=model,
        duration_sec=2,
        compact=True,
        num_workers=num_workers,
        video_batch=video_batch,
        batch_size=frame_batch,
    ),
    total=misses,
    desc='Patch cache',
    unit=' video',
    dynamic_ncols=True,
):
    pass
PY

echo
echo "== Step 2/3: Fit patch spatial params =="
python "$STALL_DIR/src/create_patch_params.py" \
  --csv "$CSV_PATH" \
  --patch-emb-cache "$PATCH_CACHE_DIR" \
  --output "$PATCH_PARAMS_PATH" \
  --duration 2 \
  --compact \
  --real-only \
  --max-patches-for-fit "$MAX_PATCHES_FOR_FIT"

echo
echo "== Step 3/3: Run patch spatial eval =="
python "$STALL_DIR/src/eval_patch.py" \
  --csv "$CSV_PATH" \
  --patch-emb-cache "$PATCH_CACHE_DIR" \
  --patch-params "$PATCH_PARAMS_PATH" \
  --output-csv "$OUTPUT_CSV" \
  --duration 2 \
  --compact \
  --workers "$NUM_WORKERS" \
  --video-batch "$VIDEO_BATCH"

echo
echo "Done."
echo "Patch params: $PATCH_PARAMS_PATH"
echo "Result CSV:   $OUTPUT_CSV"
