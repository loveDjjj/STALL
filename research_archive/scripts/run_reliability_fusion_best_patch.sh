#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/OneDay/STALL_project/STALL}"
OUT_DIR="${OUT_DIR:-$ROOT/results/reliability_fusion}"
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

  "$PYTHON_BIN" "$ROOT/tools/reliability_fusion.py" \
    --dataset "$dataset" \
    --tag "$tag" \
    --global-csv "$global_csv" \
    --patch-csv "$patch_csv" \
    --output-dir "$OUT_DIR" \
    --base-alphas "0.50,0.55,0.60,0.65,0.70,0.75,0.80" \
    --save-best-fused-csv
}

run_one \
  "comgenvid" \
  "best_patch_second_order_r3_bottomk0p50" \
  "$ROOT/results/comgenvid_results.csv" \
  "$ROOT/results/patch_region3_bottomk_sweep/comgenvid_patch_region3_second_order_bottomk0p50_spat10_temp90.csv"

run_one \
  "videofeedback" \
  "best_patch_second_order_r1_mean" \
  "$ROOT/results/videofeedback_results.csv" \
  "$ROOT/results/patch_cross_dataset_ablation/videofeedback_patch_same_grid_second_order_region1_mean_spat10_temp90.csv"

run_one \
  "genvideo" \
  "best_patch_second_order_r2_mean" \
  "$ROOT/results/genvideo_results.csv" \
  "$ROOT/results/patch_cross_dataset_ablation/genvideo_patch_same_grid_second_order_region2_mean_spat10_temp90.csv"

"$PYTHON_BIN" - "$OUT_DIR" <<'PY'
from pathlib import Path
import sys
import pandas as pd

out_dir = Path(sys.argv[1])
frames = [pd.read_csv(p) for p in sorted(out_dir.glob("*_reliability_summary.csv"))]
summary = pd.concat(frames, ignore_index=True)
summary.to_csv(out_dir / "all_reliability_summary.csv", index=False)

per_dataset_best = (
    summary.sort_values(["dataset", "avg_auc", "avg_ap"], ascending=[True, False, False])
    .groupby("dataset", as_index=False)
    .head(5)
)
per_dataset_best.to_csv(out_dir / "top5_by_dataset.csv", index=False)

universal = (
    summary.groupby(["method", "base_alpha"], as_index=False)
    .agg(
        mean_auc=("avg_auc", "mean"),
        mean_ap=("avg_ap", "mean"),
        min_auc=("avg_auc", "min"),
        min_ap=("avg_ap", "min"),
        alpha_mean=("alpha_mean", "mean"),
    )
    .sort_values(["mean_auc", "mean_ap"], ascending=False)
)
universal.to_csv(out_dir / "universal_reliability_summary.csv", index=False)

lines = [
    "# Reliability-Aware Fusion Summary",
    "",
    "Score definition:",
    "",
    "`final_score = alpha(x) * global_score + (1 - alpha(x)) * patch_score`",
    "",
    "Reliability uses only the real-score empirical tail distance of each branch.",
    "",
    "## Universal Method Ranking",
    "",
    "| Method | Base alpha | Mean AUC | Mean AP | Min AUC | Min AP | Mean alpha |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for row in universal.head(15).itertuples(index=False):
    lines.append(
        f"| {row.method} | {row.base_alpha:.2f} | {row.mean_auc:.4f} | {row.mean_ap:.4f} | "
        f"{row.min_auc:.4f} | {row.min_ap:.4f} | {row.alpha_mean:.3f} |"
    )

lines += [
    "",
    "## Top 5 Per Dataset",
    "",
    "| Dataset | Method | Base alpha | AUC | AP | Mean alpha |",
    "|---|---|---:|---:|---:|---:|",
]
for row in per_dataset_best.itertuples(index=False):
    lines.append(
        f"| {row.dataset} | {row.method} | {row.base_alpha:.2f} | "
        f"{row.avg_auc:.4f} | {row.avg_ap:.4f} | {row.alpha_mean:.3f} |"
    )

(out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(universal.head(12).to_string(index=False))
print(f"Wrote {out_dir / 'README.md'}")
PY

echo "Done."
