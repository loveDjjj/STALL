#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/OneDay/STALL_project"
STALL_DIR="$ROOT/STALL"

CSV_PATH="$STALL_DIR/cache/indexes/comgenvid.csv"
GLOBAL_CACHE_DIR="$STALL_DIR/cache/embeddings/comgenvid"
PATCH_CACHE_DIR="$STALL_DIR/cache/patch_embeddings/comgenvid"
GLOBAL_PARAMS="$STALL_DIR/precomputed/stall_params_vatex_dino_v3.npz"
PATCH_PARAMS="$STALL_DIR/precomputed/patch_params_comgenvid_real.npz"
RESULT_DIR="$STALL_DIR/results/comgenvid_fusion_grid"
SUMMARY_CSV="$STALL_DIR/results/comgenvid_fusion_grid_summary.csv"
SUMMARY_MD="$STALL_DIR/results/comgenvid_fusion_grid_summary.md"

export TORCH_HOME="${TORCH_HOME:-/tmp/torch_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

GLOBAL_WEIGHTS=(${GLOBAL_WEIGHTS:-0.5 0.6 0.7 0.8})
PATCH_SPAT_WEIGHTS=(${PATCH_SPAT_WEIGHTS:-0.5 0.3 0.7})
PATCH_TEMP_WEIGHTS=(${PATCH_TEMP_WEIGHTS:-0.5 0.7 0.3})

echo "ROOT=$ROOT"
echo "STALL_DIR=$STALL_DIR"
echo "RESULT_DIR=$RESULT_DIR"
echo "SUMMARY_CSV=$SUMMARY_CSV"
echo "SUMMARY_MD=$SUMMARY_MD"
echo "GLOBAL_WEIGHTS=${GLOBAL_WEIGHTS[*]}"
echo "PATCH_SPAT_WEIGHTS=${PATCH_SPAT_WEIGHTS[*]}"
echo "PATCH_TEMP_WEIGHTS=${PATCH_TEMP_WEIGHTS[*]}"
echo "TORCH_HOME=$TORCH_HOME"
echo "XDG_CACHE_HOME=$XDG_CACHE_HOME"
echo

cd "$ROOT"
source /home/ubuntu/anaconda3/bin/activate stall

mkdir -p "$RESULT_DIR"
rm -f "$SUMMARY_CSV" "$SUMMARY_MD"

python - <<'PY'
import csv
from pathlib import Path

summary_path = Path("/data/OneDay/STALL_project/STALL/results/comgenvid_fusion_grid_summary.csv")
summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "run_name",
        "global_weight",
        "patch_spat_weight",
        "patch_temp_weight",
        "sora_auc",
        "sora_ap",
        "veo3_auc",
        "veo3_ap",
        "avg_auc",
        "avg_ap",
        "output_csv",
    ])
PY

run_counter=0
for global_weight in "${GLOBAL_WEIGHTS[@]}"; do
  for i in "${!PATCH_SPAT_WEIGHTS[@]}"; do
    patch_spat_weight="${PATCH_SPAT_WEIGHTS[$i]}"
    patch_temp_weight="${PATCH_TEMP_WEIGHTS[$i]}"
    run_name="gw_${global_weight}_ps_${patch_spat_weight}_pt_${patch_temp_weight}"
    output_csv="$RESULT_DIR/${run_name}.csv"

    run_counter=$((run_counter + 1))
    echo "== Run ${run_counter}: ${run_name} =="

    python "$STALL_DIR/src/eval_patch.py" \
      --csv "$CSV_PATH" \
      --emb-cache "$GLOBAL_CACHE_DIR" \
      --patch-emb-cache "$PATCH_CACHE_DIR" \
      --global-params "$GLOBAL_PARAMS" \
      --patch-params "$PATCH_PARAMS" \
      --output-csv "$output_csv" \
      --duration 2 \
      --compact \
      --workers 8 \
      --video-batch 4 \
      --patch-temp-mode same_grid \
      --fusion avg \
      --global-weight "$global_weight" \
      --patch-spat-weight "$patch_spat_weight" \
      --patch-temp-weight "$patch_temp_weight"

    python - "$output_csv" "$run_name" "$global_weight" "$patch_spat_weight" "$patch_temp_weight" <<'PY'
import csv
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, '/data/OneDay/STALL_project/STALL/src')
from metrics import Score, ScoreDirection, get_results_df

output_csv, run_name, global_weight, patch_spat_weight, patch_temp_weight = sys.argv[1:]
df = pd.read_csv(output_csv)
res = get_results_df(
    df[["subset", "source_model"]].copy(),
    {"final_score": Score(value=df["final_score"].to_numpy(), direction=ScoreDirection.HIGHER_IS_REAL)},
)
res = res.set_index("Generative Model")

row = [
    run_name,
    float(global_weight),
    float(patch_spat_weight),
    float(patch_temp_weight),
    float(res.loc["Sora", "final_score AUC"]),
    float(res.loc["Sora", "final_score AP"]),
    float(res.loc["VEO3", "final_score AUC"]),
    float(res.loc["VEO3", "final_score AP"]),
    float(res.loc["Average", "final_score AUC"]),
    float(res.loc["Average", "final_score AP"]),
    output_csv,
]

summary_path = Path("/data/OneDay/STALL_project/STALL/results/comgenvid_fusion_grid_summary.csv")
with summary_path.open("a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(row)
print("summary_row", row)
PY

    echo
  done
done

python - <<'PY'
from pathlib import Path
import pandas as pd

summary_csv = Path("/data/OneDay/STALL_project/STALL/results/comgenvid_fusion_grid_summary.csv")
summary_md = Path("/data/OneDay/STALL_project/STALL/results/comgenvid_fusion_grid_summary.md")

df = pd.read_csv(summary_csv)
df = df.sort_values(["avg_auc", "avg_ap"], ascending=False).reset_index(drop=True)
df.to_csv(summary_csv, index=False)

best = df.iloc[0]

lines = []
lines.append("# ComGenVid Fusion Grid Search")
lines.append("")
lines.append("按 `avg_auc` 与 `avg_ap` 排序的 same-grid fusion 权重搜索结果。")
lines.append("")
lines.append("## 最优配置")
lines.append("")
lines.append(f"- run_name: `{best['run_name']}`")
lines.append(f"- global_weight: `{best['global_weight']}`")
lines.append(f"- patch_spat_weight: `{best['patch_spat_weight']}`")
lines.append(f"- patch_temp_weight: `{best['patch_temp_weight']}`")
lines.append(f"- Sora: `AUC {best['sora_auc']:.3f} / AP {best['sora_ap']:.3f}`")
lines.append(f"- VEO3: `AUC {best['veo3_auc']:.3f} / AP {best['veo3_ap']:.3f}`")
lines.append(f"- Average: `AUC {best['avg_auc']:.3f} / AP {best['avg_ap']:.3f}`")
lines.append("")
lines.append("## 全部结果")
lines.append("")
lines.append(df.to_markdown(index=False))
lines.append("")

summary_md.write_text("\n".join(lines), encoding="utf-8")
print(f"Saved summary markdown -> {summary_md}")
PY

echo "Done."
echo "Summary CSV: $SUMMARY_CSV"
echo "Summary MD:  $SUMMARY_MD"
