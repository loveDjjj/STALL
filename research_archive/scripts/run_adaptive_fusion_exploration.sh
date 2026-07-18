#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/OneDay/STALL_project/STALL}"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/anaconda3/envs/stall/bin/python}"
OUT_DIR="${OUT_DIR:-$ROOT/results/adaptive_fusion_exploration}"

mkdir -p "$OUT_DIR"

run_one() {
  local dataset="$1"
  local tag="$2"
  local global_csv="$3"
  local patch_csv="$4"

  echo "============================================================"
  echo "Adaptive fusion: dataset=$dataset tag=$tag"
  echo "Global: $global_csv"
  echo "Patch:  $patch_csv"
  echo "============================================================"

  "$PYTHON_BIN" "$ROOT/tools/adaptive_fusion.py" \
    --dataset "$dataset" \
    --tag "$tag" \
    --global-csv "$global_csv" \
    --patch-csv "$patch_csv" \
    --output-dir "$OUT_DIR" \
    --save-best-fused-csv
}

echo "Adaptive fusion outputs: $OUT_DIR"

run_one \
  "comgenvid" \
  "second_order_region3_bottomk0p20_spat10_temp90" \
  "$ROOT/results/comgenvid_results.csv" \
  "$ROOT/results/comgenvid_patch_second_order_region3_bottomk0p20_spat10_temp90.csv"

run_one \
  "comgenvid" \
  "second_order_bottomk0p20_region1_spat10_temp90" \
  "$ROOT/results/comgenvid_results.csv" \
  "$ROOT/results/patch_second_order_bottomk_sweep/comgenvid_patch_second_order_bottomk0p20_spat10_temp90.csv"

run_one \
  "comgenvid" \
  "lag1_v2_spat70_temp30" \
  "$ROOT/results/comgenvid_results.csv" \
  "$ROOT/results/comgenvid_patch_samegrid_lag1_v2.csv"

run_one \
  "videofeedback" \
  "second_order_region1_mean_spat10_temp90" \
  "$ROOT/results/videofeedback_results.csv" \
  "$ROOT/results/patch_cross_dataset_ablation/videofeedback_patch_same_grid_second_order_region1_mean_spat10_temp90.csv"

run_one \
  "videofeedback" \
  "lag1_region1_mean_spat10_temp90" \
  "$ROOT/results/videofeedback_results.csv" \
  "$ROOT/results/patch_cross_dataset_ablation/videofeedback_patch_same_grid_lag1_region1_mean_spat10_temp90.csv"

run_one \
  "videofeedback" \
  "second_order_region1_bottomk0p5_spat10_temp90" \
  "$ROOT/results/videofeedback_results.csv" \
  "$ROOT/results/patch_cross_dataset_ablation/videofeedback_patch_same_grid_second_order_region1_bottomk0p5_spat10_temp90.csv"

run_one \
  "genvideo" \
  "second_order_region2_mean_spat10_temp90" \
  "$ROOT/results/genvideo_results.csv" \
  "$ROOT/results/patch_cross_dataset_ablation/genvideo_patch_same_grid_second_order_region2_mean_spat10_temp90.csv"

run_one \
  "genvideo" \
  "lag1_region2_mean_spat10_temp90" \
  "$ROOT/results/genvideo_results.csv" \
  "$ROOT/results/patch_cross_dataset_ablation/genvideo_patch_same_grid_lag1_region2_mean_spat10_temp90.csv"

run_one \
  "genvideo" \
  "second_order_region1_mean_spat10_temp90" \
  "$ROOT/results/genvideo_results.csv" \
  "$ROOT/results/patch_cross_dataset_ablation/genvideo_patch_same_grid_second_order_region1_mean_spat10_temp90.csv"

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("/data/OneDay/STALL_project/STALL")
out = root / "results/adaptive_fusion_exploration"
frames = []
for path in sorted(out.glob("*_adaptive_summary.csv")):
    frames.append(pd.read_csv(path))
summary = pd.concat(frames, ignore_index=True)
summary.to_csv(out / "all_adaptive_fusion_results.csv", index=False)
best_by_config = summary.sort_values(["dataset", "tag", "avg_auc", "avg_ap"], ascending=[True, True, False, False]).groupby(["dataset", "tag"]).head(1)
best_by_dataset = summary.sort_values(["dataset", "avg_auc", "avg_ap"], ascending=[True, False, False]).groupby("dataset").head(1)
best_by_config.to_csv(out / "best_adaptive_by_config.csv", index=False)
best_by_dataset.to_csv(out / "best_adaptive_by_dataset.csv", index=False)

print("Best adaptive by dataset:")
print(best_by_dataset[["dataset", "tag", "strategy", "base_alpha", "avg_auc", "avg_ap", "alpha_mean"]].to_string(index=False))
PY

echo "Done."
