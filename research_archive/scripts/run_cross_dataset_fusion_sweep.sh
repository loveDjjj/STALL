#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/OneDay/STALL_project/STALL}"
OUT_DIR="${OUT_DIR:-$ROOT/results/fusion_sweep_second_order}"
ALPHAS="${ALPHAS:-0:1:0.05}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$OUT_DIR"

run_one() {
  local dataset="$1"
  local tag="$2"
  local global_csv="$3"
  local patch_csv="$4"

  echo "============================================================"
  echo "Dataset=$dataset Tag=$tag"
  echo "Global: $global_csv"
  echo "Patch:  $patch_csv"
  echo "============================================================"

  "$PYTHON_BIN" "$ROOT/tools/fuse_scores.py" \
    --dataset "$dataset" \
    --tag "$tag" \
    --global-csv "$global_csv" \
    --patch-csv "$patch_csv" \
    --alphas "$ALPHAS" \
    --output-dir "$OUT_DIR"
}

echo "Writing fusion sweep outputs to: $OUT_DIR"
echo "Alpha sweep: $ALPHAS"

# ComGenVid: 3 representative patch configs.
run_one \
  "comgenvid" \
  "second_order_bottomk0p20_region1_spat10_temp90" \
  "$ROOT/results/comgenvid_results.csv" \
  "$ROOT/results/patch_second_order_bottomk_sweep/comgenvid_patch_second_order_bottomk0p20_spat10_temp90.csv"

run_one \
  "comgenvid" \
  "second_order_region3_bottomk0p20_spat10_temp90" \
  "$ROOT/results/comgenvid_results.csv" \
  "$ROOT/results/comgenvid_patch_second_order_region3_bottomk0p20_spat10_temp90.csv"

run_one \
  "comgenvid" \
  "lag1_v2_spat70_temp30" \
  "$ROOT/results/comgenvid_results.csv" \
  "$ROOT/results/comgenvid_patch_samegrid_lag1_v2.csv"

# VideoFeedback: 3 representative patch configs from the cross-dataset grid.
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

# GenVideo: 3 representative patch configs from the cross-dataset grid.
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
out_dir = Path(root / "results/fusion_sweep_second_order")
frames = []
for path in sorted(out_dir.glob("*_summary.csv")):
    df = pd.read_csv(path)
    frames.append(df)

summary = pd.concat(frames, ignore_index=True)
best = summary.sort_values(["dataset", "avg_auc", "avg_ap"], ascending=[True, False, False]).groupby("dataset").head(5)
best_all = summary.sort_values(["avg_auc", "avg_ap"], ascending=False).groupby(["dataset", "tag"]).head(1)
summary.to_csv(out_dir / "all_fusion_sweeps.csv", index=False)
best_all.to_csv(out_dir / "best_alpha_by_config.csv", index=False)

lines = [
    "# Cross-Dataset Fusion Sweep Summary",
    "",
    "`final_score = alpha * global_score + (1 - alpha) * patch_score`",
    "",
    "## Best Alpha Per Config",
    "",
    "| Dataset | Config | Best alpha | AUC | AP |",
    "|---|---|---:|---:|---:|",
]
for row in best_all.sort_values(["dataset", "avg_auc"], ascending=[True, False]).itertuples(index=False):
    lines.append(f"| {row.dataset} | {row.tag} | {row.alpha:.2f} | {row.avg_auc:.4f} | {row.avg_ap:.4f} |")

lines += [
    "",
    "## Top 5 Per Dataset",
    "",
    "| Dataset | Config | Alpha | AUC | AP |",
    "|---|---|---:|---:|---:|",
]
for row in best.itertuples(index=False):
    lines.append(f"| {row.dataset} | {row.tag} | {row.alpha:.2f} | {row.avg_auc:.4f} | {row.avg_ap:.4f} |")

(out_dir / "README.md").write_text("\\n".join(lines) + "\\n", encoding="utf-8")
print(f"Combined summary: {out_dir / 'README.md'}")
print(best_all.sort_values(["dataset", "avg_auc"], ascending=[True, False])[["dataset", "tag", "alpha", "avg_auc", "avg_ap"]].to_string(index=False))
PY

echo "Done."
