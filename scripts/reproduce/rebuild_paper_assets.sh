#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p results/paper_scores results/paper_tables results/paper_sweeps
run_py() {
  conda run --no-capture-output -n stall python "$@"
}

echo "[1/5] 重建 Alpha-STALLED 融合分数和主指标"
for dataset in comgenvid videofeedback genvideo; do
  run_py tools/eval_alpha_stalled.py \
    --global-csv "results/paper_scores/${dataset}_global.csv" \
    --patch-csv "results/paper_scores/${dataset}_patch_second_order.csv" \
    --patch-score-col patch_final_score \
    --alpha 0.60 \
    --output-csv "results/paper_scores/${dataset}_alpha_stalled.csv" \
    --metrics-csv "results/paper_tables/${dataset}_alpha_stalled_metrics.csv"
done

echo "[2/5] 重建组件消融指标"
for dataset in comgenvid videofeedback genvideo; do
  run_py tools/eval_score_csv.py \
    --csv "results/paper_scores/${dataset}_global.csv" \
    --score-col final_score \
    --output-csv "results/paper_tables/${dataset}_global_only_metrics.csv"
  run_py tools/eval_score_csv.py \
    --csv "results/paper_scores/${dataset}_patch_second_order.csv" \
    --score-col patch_final_score \
    --output-csv "results/paper_tables/${dataset}_patch_only_metrics.csv"
done

echo "[3/5] 重建 ComGenVid 局部时序消融指标"
run_py tools/eval_score_csv.py \
  --csv results/paper_scores/comgenvid_patch_spatial.csv \
  --score-col patch_final_score \
  --output-csv results/paper_tables/comgenvid_patch_spatial_metrics.csv
run_py tools/eval_score_csv.py \
  --csv results/paper_scores/comgenvid_patch_lag1.csv \
  --score-col patch_final_score \
  --output-csv results/paper_tables/comgenvid_patch_lag1_metrics.csv
run_py tools/eval_score_csv.py \
  --csv results/paper_scores/comgenvid_patch_multilag.csv \
  --score-col patch_final_score \
  --output-csv results/paper_tables/comgenvid_patch_multilag_metrics.csv
run_py tools/eval_score_csv.py \
  --csv results/paper_scores/comgenvid_patch_motionhard.csv \
  --score-col patch_final_score \
  --output-csv results/paper_tables/comgenvid_patch_motionhard_metrics.csv
run_py tools/eval_score_csv.py \
  --csv results/paper_scores/comgenvid_patch_motionsoft.csv \
  --score-col patch_final_score \
  --output-csv results/paper_tables/comgenvid_patch_motionsoft_metrics.csv
run_py tools/eval_score_csv.py \
  --csv results/paper_scores/comgenvid_patch_second_order_ablation.csv \
  --score-col patch_final_score \
  --output-csv results/paper_tables/comgenvid_patch_second_order_metrics.csv

echo "[4/5] 重建 alpha sweep"
for dataset in comgenvid videofeedback genvideo; do
  run_py tools/fuse_scores.py \
    --dataset "$dataset" \
    --tag same_grid_second_order \
    --global-csv "results/paper_scores/${dataset}_global.csv" \
    --patch-csv "results/paper_scores/${dataset}_patch_second_order.csv" \
    --patch-score-col patch_final_score \
    --alphas 0:1:0.05 \
    --output-dir "results/paper_sweeps/${dataset}_alpha"
done

echo "[5/5] 重建 markdown 汇总文件"
conda run --no-capture-output -n stall python - <<'PY'
from pathlib import Path
import pandas as pd

DATASETS = ["comgenvid", "videofeedback", "genvideo"]
DISPLAY = {"comgenvid": "ComGenVid", "videofeedback": "VideoFeedback", "genvideo": "GenVideo"}

def avg_metrics(path: str, auc_col: str, ap_col: str) -> tuple[float, float]:
    df = pd.read_csv(path)
    row = df[df["Generative Model"] == "Average"].iloc[0]
    return float(row[auc_col]), float(row[ap_col])

main_lines = [
    "# Alpha-STALLED 主实验汇总",
    "",
    "由 refactor 分支上的以下命令生成：",
    "",
    "```text",
    "tools/eval_alpha_stalled.py",
    "alpha = 0.60",
    "metrics = src/metrics.py pairwise balanced AUC/AP",
    "```",
    "",
    "| 数据集 | 融合 score CSV | Metrics CSV | 平均 AUC | 平均 AP |",
    "|---|---|---|---:|---:|",
]
for dataset in DATASETS:
    metrics_csv = f"results/paper_tables/{dataset}_alpha_stalled_metrics.csv"
    auc, ap = avg_metrics(metrics_csv, "final_score AUC", "final_score AP")
    main_lines.append(
        f"| {DISPLAY[dataset]} | "
        f"`results/paper_scores/{dataset}_alpha_stalled.csv` | "
        f"`{metrics_csv}` | {auc:.4f} | {ap:.4f} |"
    )
main_lines += [
    "",
    "这些数值是当前已验证的 release baseline。若更新手稿表格，",
    "应使用同一组命令重新生成本文件和 `docs/restructure/reproducibility_audit.md`。",
]
Path("results/paper_tables/alpha_stalled_main_summary.md").write_text(
    "\n".join(main_lines) + "\n", encoding="utf-8"
)

component_rows = []
for dataset in DATASETS:
    for label, suffix, col in [
        ("仅 Global", "global_only", "final_score"),
        ("仅 Patch", "patch_only", "patch_final_score"),
        ("Alpha-STALLED", "alpha_stalled", "final_score"),
    ]:
        metrics_csv = f"results/paper_tables/{dataset}_{suffix}_metrics.csv"
        auc, ap = avg_metrics(metrics_csv, f"{col} AUC", f"{col} AP")
        score_csv = (
            f"results/paper_scores/{dataset}_alpha_stalled.csv"
            if suffix == "alpha_stalled"
            else f"results/paper_scores/{dataset}_{'patch_second_order' if suffix == 'patch_only' else 'global'}.csv"
        )
        component_rows.append((dataset, label, score_csv, col, auc, ap))

local_specs = [
    ("仅 patch 空间", "results/paper_scores/comgenvid_patch_spatial.csv", "comgenvid_patch_spatial_metrics.csv"),
    ("同网格 lag-1", "results/paper_scores/comgenvid_patch_lag1.csv", "comgenvid_patch_lag1_metrics.csv"),
    ("Multi-lag", "results/paper_scores/comgenvid_patch_multilag.csv", "comgenvid_patch_multilag_metrics.csv"),
    ("Motion-hard", "results/paper_scores/comgenvid_patch_motionhard.csv", "comgenvid_patch_motionhard_metrics.csv"),
    ("Motion-soft", "results/paper_scores/comgenvid_patch_motionsoft.csv", "comgenvid_patch_motionsoft_metrics.csv"),
    (
        "同网格二阶",
        "results/paper_scores/comgenvid_patch_second_order_ablation.csv",
        "comgenvid_patch_second_order_metrics.csv",
    ),
]

ablation_lines = [
    "# Alpha-STALLED 消融汇总",
    "",
    "本文件所有行均由 `tools/eval_score_csv.py` 重新计算。该工具使用",
    "`src/metrics.py` 的 pairwise balanced real-vs-generator 评测。结果报告为",
    "`AUC / AP`，且分数越高表示越接近真实视频。",
    "",
    "## Global/Patch 组件消融",
    "",
    "| Benchmark | 组件 | Score CSV | 分数列 | 平均 AUC / AP |",
    "|---|---|---|---|---:|",
]
for dataset, label, score_csv, col, auc, ap in component_rows:
    ablation_lines.append(
        f"| {DISPLAY[dataset]} | {label} | "
        f"`{score_csv}` | `{col}` | {auc:.4f} / {ap:.4f} |"
    )
ablation_lines += [
    "",
    "## ComGenVid 局部时序定义消融",
    "",
    "| 局部证据 | Score CSV | 分数列 | 平均 AUC / AP |",
    "|---|---|---|---:|",
]
for label, score_csv, metrics_name in local_specs:
    auc, ap = avg_metrics(f"results/paper_tables/{metrics_name}", "patch_final_score AUC", "patch_final_score AP")
    ablation_lines.append(f"| {label} | `{score_csv}` | `patch_final_score` | {auc:.4f} / {ap:.4f} |")
ablation_lines += [
    "",
    "## 说明",
    "",
    "- 这些数值取代早期一次性脚本生成的表格；早期脚本没有始终使用统一的",
    "  real/generated 平衡口径。",
    "- Persistence、fallback 和 source/rank selector 仍为诊断上限资产，",
    "  不纳入本消融汇总。",
]
Path("results/paper_tables/ablation_summary.md").write_text(
    "\n".join(ablation_lines) + "\n", encoding="utf-8"
)

sweep_rows = []
for dataset in DATASETS:
    path = Path(f"results/paper_sweeps/{dataset}_alpha/{dataset}_same_grid_second_order_summary.csv")
    df = pd.read_csv(path)
    best_auc = df.sort_values(["avg_auc", "avg_ap"], ascending=False).iloc[0]
    best_ap = df.sort_values(["avg_ap", "avg_auc"], ascending=False).iloc[0]
    alpha06 = df[df["alpha"].round(10) == 0.6].iloc[0]
    sweep_rows.append((dataset, path, best_auc, best_ap, alpha06))

sweep_lines = [
    "# Alpha-STALLED 固定 alpha sweep 汇总",
    "",
    "由 `tools/fuse_scores.py` 和 `src/metrics.py` pairwise balanced 指标生成。",
    "",
    "| 数据集 | Best AUC alpha | Best AUC / AP | Best AP alpha | Best AP AUC / AP | 冻结 alpha=0.60 AUC / AP | Summary CSV |",
    "|---|---:|---:|---:|---:|---:|---|",
]
for dataset, path, best_auc, best_ap, alpha06 in sweep_rows:
    sweep_lines.append(
        f"| {dataset} | {best_auc['alpha']:.2f} | {best_auc['avg_auc']:.4f} / {best_auc['avg_ap']:.4f} | "
        f"{best_ap['alpha']:.2f} | {best_ap['avg_auc']:.4f} / {best_ap['avg_ap']:.4f} | "
        f"{alpha06['avg_auc']:.4f} / {alpha06['avg_ap']:.4f} | `{path}` |"
    )
sweep_lines += [
    "",
    "best-alpha 列是对已有 score 文件的诊断性 oracle sweep。除非另行定义验证协议，"
    "论文主线 release baseline 仍使用 `configs/alpha_stalled.yaml` 中记录的冻结 alpha。",
]
Path("results/paper_sweeps/alpha_sweep_summary.md").write_text(
    "\n".join(sweep_lines) + "\n", encoding="utf-8"
)
PY

run_py tools/verify_alpha_stalled_release.py
echo "论文资产已重建并通过验证。"
