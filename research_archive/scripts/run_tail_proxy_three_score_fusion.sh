#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/OneDay/STALL_project/STALL}"
ENS_DIR="${ENS_DIR:-$ROOT/results/patch_score_ensemble}"
OUT_DIR="${OUT_DIR:-$ROOT/results/tail_proxy_three_score_fusion}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$OUT_DIR"

run_one() {
  local dataset="$1"
  local tag="$2"
  local global_csv="$3"
  local best_patch_csv="$4"
  local ensemble_scores="$5"
  local tail_col="$6"

  "$PYTHON_BIN" "$ROOT/tools/three_score_fusion.py" \
    --dataset "$dataset" \
    --tag "$tag" \
    --score-a-csv "$global_csv" \
    --score-b-csv "$best_patch_csv" \
    --score-c-csv "$ensemble_scores" \
    --score-c-col "$tail_col" \
    --score-a-name global \
    --score-b-name best_patch \
    --score-c-name "$tail_col" \
    --weight-step 0.05 \
    --rank-normalize \
    --output-dir "$OUT_DIR"
}

run_one \
  "comgenvid" \
  "tail_gap_neg_rank" \
  "$ROOT/results/comgenvid_results.csv" \
  "$ROOT/results/patch_region3_bottomk_sweep/comgenvid_patch_region3_second_order_bottomk0p50_spat10_temp90.csv" \
  "$ENS_DIR/comgenvid_region3_second_order_tail_sweep_patch_ensemble_scores.csv" \
  "patch_ensemble_tail_gap_neg"

run_one \
  "videofeedback" \
  "tail_gap_neg_rank" \
  "$ROOT/results/videofeedback_results.csv" \
  "$ROOT/results/patch_cross_dataset_ablation/videofeedback_patch_same_grid_second_order_region1_mean_spat10_temp90.csv" \
  "$ENS_DIR/videofeedback_region1_second_order_multiscale_patch_ensemble_scores.csv" \
  "patch_ensemble_tail_gap_neg"

run_one \
  "genvideo" \
  "tail_gap_neg_rank" \
  "$ROOT/results/genvideo_results.csv" \
  "$ROOT/results/patch_cross_dataset_ablation/genvideo_patch_same_grid_second_order_region2_mean_spat10_temp90.csv" \
  "$ENS_DIR/genvideo_region2_second_order_multiscale_patch_ensemble_scores.csv" \
  "patch_ensemble_tail_gap_neg"

"$PYTHON_BIN" - "$OUT_DIR" <<'PY'
from pathlib import Path
import sys
import pandas as pd

out_dir = Path(sys.argv[1])
summary = pd.concat([pd.read_csv(p) for p in sorted(out_dir.glob("*_three_score_summary.csv"))], ignore_index=True)
summary.to_csv(out_dir / "all_three_score_summary.csv", index=False)
best = summary.sort_values(["dataset", "avg_auc", "avg_ap"], ascending=[True, False, False]).groupby("dataset").head(5)
best.to_csv(out_dir / "top5_by_dataset.csv", index=False)
universal = (
    summary.groupby(["wa", "wb", "wc"], as_index=False)
    .agg(mean_auc=("avg_auc", "mean"), mean_ap=("avg_ap", "mean"), min_auc=("avg_auc", "min"), min_ap=("avg_ap", "min"))
    .sort_values(["mean_auc", "mean_ap"], ascending=False)
)
universal.to_csv(out_dir / "universal_three_score_summary.csv", index=False)
print(universal.head(12).to_string(index=False))
PY

echo "Done."
