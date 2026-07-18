#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/OneDay/STALL_project/STALL}"
OUT_DIR="${OUT_DIR:-$ROOT/results/patch_score_ensemble}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$OUT_DIR"

run_one() {
  local dataset="$1"
  local tag="$2"
  local global_csv="$3"
  local specs="$4"

  "$PYTHON_BIN" "$ROOT/tools/patch_score_ensemble.py" \
    --dataset "$dataset" \
    --tag "$tag" \
    --global-csv "$global_csv" \
    --patch-specs "$specs" \
    --output-dir "$OUT_DIR" \
    --alphas "0:1:0.025" \
    --save-scores
}

run_one \
  "comgenvid" \
  "region3_second_order_tail_sweep" \
  "$ROOT/results/comgenvid_results.csv" \
  "b10=$ROOT/results/patch_region3_bottomk_sweep/comgenvid_patch_region3_second_order_bottomk0p10_spat10_temp90.csv,b15=$ROOT/results/patch_region3_bottomk_sweep/comgenvid_patch_region3_second_order_bottomk0p15_spat10_temp90.csv,b25=$ROOT/results/patch_region3_bottomk_sweep/comgenvid_patch_region3_second_order_bottomk0p25_spat10_temp90.csv,b35=$ROOT/results/patch_region3_bottomk_sweep/comgenvid_patch_region3_second_order_bottomk0p35_spat10_temp90.csv,b50=$ROOT/results/patch_region3_bottomk_sweep/comgenvid_patch_region3_second_order_bottomk0p50_spat10_temp90.csv"

run_one \
  "videofeedback" \
  "region1_second_order_multiscale" \
  "$ROOT/results/videofeedback_results.csv" \
  "b20=$ROOT/results/patch_cross_dataset_ablation/videofeedback_patch_same_grid_second_order_region1_bottomk0p2_spat10_temp90.csv,b50=$ROOT/results/patch_cross_dataset_ablation/videofeedback_patch_same_grid_second_order_region1_bottomk0p5_spat10_temp90.csv,mean=$ROOT/results/patch_cross_dataset_ablation/videofeedback_patch_same_grid_second_order_region1_mean_spat10_temp90.csv"

run_one \
  "genvideo" \
  "region2_second_order_multiscale" \
  "$ROOT/results/genvideo_results.csv" \
  "b20=$ROOT/results/patch_cross_dataset_ablation/genvideo_patch_same_grid_second_order_region2_bottomk0p2_spat10_temp90.csv,b50=$ROOT/results/patch_cross_dataset_ablation/genvideo_patch_same_grid_second_order_region2_bottomk0p5_spat10_temp90.csv,mean=$ROOT/results/patch_cross_dataset_ablation/genvideo_patch_same_grid_second_order_region2_mean_spat10_temp90.csv"

"$PYTHON_BIN" - "$OUT_DIR" <<'PY'
from pathlib import Path
import sys
import pandas as pd

out_dir = Path(sys.argv[1])
patch = pd.concat([pd.read_csv(p) for p in sorted(out_dir.glob("*_patch_ensemble_summary.csv"))], ignore_index=True)
fusion = pd.concat([pd.read_csv(p) for p in sorted(out_dir.glob("*_fusion_summary.csv"))], ignore_index=True)
patch.to_csv(out_dir / "all_patch_ensemble_summary.csv", index=False)
fusion.to_csv(out_dir / "all_patch_ensemble_fusion_summary.csv", index=False)

best_patch = patch.sort_values(["dataset", "avg_auc", "avg_ap"], ascending=[True, False, False]).groupby("dataset").head(5)
best_fusion = fusion.sort_values(["dataset", "avg_auc", "avg_ap"], ascending=[True, False, False]).groupby("dataset").head(5)
universal = (
    fusion.groupby(["method", "alpha"], as_index=False)
    .agg(mean_auc=("avg_auc", "mean"), mean_ap=("avg_ap", "mean"), min_auc=("avg_auc", "min"), min_ap=("avg_ap", "min"))
    .sort_values(["mean_auc", "mean_ap"], ascending=False)
)
best_patch.to_csv(out_dir / "top5_patch_ensemble_by_dataset.csv", index=False)
best_fusion.to_csv(out_dir / "top5_patch_ensemble_fusion_by_dataset.csv", index=False)
universal.to_csv(out_dir / "universal_patch_ensemble_fusion_summary.csv", index=False)

lines = [
    "# Patch Score Ensemble Summary",
    "",
    "Patch ensembles are built from existing per-video patch CSVs; no DINO or likelihood recomputation.",
    "",
    "## Universal Fusion Ranking",
    "",
    "| Method | Alpha | Mean AUC | Mean AP | Min AUC | Min AP |",
    "|---|---:|---:|---:|---:|---:|",
]
for row in universal.head(15).itertuples(index=False):
    lines.append(f"| {row.method} | {row.alpha:.3f} | {row.mean_auc:.4f} | {row.mean_ap:.4f} | {row.min_auc:.4f} | {row.min_ap:.4f} |")
lines += [
    "",
    "## Top Patch Ensemble Per Dataset",
    "",
    "| Dataset | Method | AUC | AP |",
    "|---|---|---:|---:|",
]
for row in best_patch.itertuples(index=False):
    lines.append(f"| {row.dataset} | {row.method} | {row.avg_auc:.4f} | {row.avg_ap:.4f} |")
lines += [
    "",
    "## Top Fusion Per Dataset",
    "",
    "| Dataset | Method | Alpha | AUC | AP |",
    "|---|---|---:|---:|---:|",
]
for row in best_fusion.itertuples(index=False):
    lines.append(f"| {row.dataset} | {row.method} | {row.alpha:.3f} | {row.avg_auc:.4f} | {row.avg_ap:.4f} |")
(out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(universal.head(12).to_string(index=False))
print(f"Wrote {out_dir / 'README.md'}")
PY

echo "Done."
