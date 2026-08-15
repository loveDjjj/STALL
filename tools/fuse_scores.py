#!/usr/bin/env python3
"""扫描全局 STALL 与 patch STALL 之间的固定权重融合。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.score_csv import (
    KEY_COLUMNS,
    metric_rows as _metric_rows,
    read_keyed_scores as _read_scores,
)


def _parse_alphas(value: str) -> list[float]:
    if ":" in value:
        start, stop, step = [float(x) for x in value.split(":")]
        count = int(round((stop - start) / step)) + 1
        return [round(start + i * step, 10) for i in range(count)]
    return [float(x) for x in value.split(",")]


def _score_summary(df: pd.DataFrame) -> dict[str, float]:
    real = df[df["subset"].str.lower() == "real"]
    fake = df[df["subset"].str.lower() != "real"]
    return {
        "global_real_mean": real["global_score"].mean(),
        "global_fake_mean": fake["global_score"].mean(),
        "patch_real_mean": real["patch_score"].mean(),
        "patch_fake_mean": fake["patch_score"].mean(),
        "corr_global_patch": df[["global_score", "patch_score"]].corr().iloc[0, 1],
    }


def _write_markdown(path: Path, summary: pd.DataFrame, dataset: str) -> None:
    best_auc = summary.sort_values(["avg_auc", "avg_ap"], ascending=False).iloc[0]
    best_ap = summary.sort_values(["avg_ap", "avg_auc"], ascending=False).iloc[0]

    lines = [
        f"# {dataset} 融合 sweep",
        "",
        "分数定义：",
        "",
        "`final_score = alpha * global_score + (1 - alpha) * patch_score`",
        "",
        f"- Best AUC: alpha={best_auc.alpha:.2f}, AUC={best_auc.avg_auc:.4f}, AP={best_auc.avg_ap:.4f}",
        f"- Best AP: alpha={best_ap.alpha:.2f}, AUC={best_ap.avg_auc:.4f}, AP={best_ap.avg_ap:.4f}",
        "",
        "| alpha | 平均 AUC | 平均 AP |",
        "|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.alpha:.2f} | {row.avg_auc:.4f} | {row.avg_ap:.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--global-csv", type=Path, required=True)
    parser.add_argument("--patch-csv", type=Path, required=True)
    parser.add_argument("--global-score-col", default="final_score")
    parser.add_argument("--patch-score-col", default="final_score")
    parser.add_argument("--alphas", default="0:1:0.05")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-fused-csv", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    global_df = _read_scores(args.global_csv, args.global_score_col, "global")
    patch_df = _read_scores(args.patch_csv, args.patch_score_col, "patch")
    merged = global_df.merge(
        patch_df,
        on=list(KEY_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(global_df) or len(merged) != len(patch_df):
        print(
            f"警告：合并后 {len(merged)} 行；global={len(global_df)} patch={len(patch_df)}。"
            "指标只使用交集。"
        )

    stats = _score_summary(merged)
    rows = []
    per_model_tables = []
    for alpha in _parse_alphas(args.alphas):
        scored = merged.copy()
        scored["final_score"] = alpha * scored["global_score"] + (1.0 - alpha) * scored["patch_score"]
        metrics = _metric_rows(scored, "final_score", seed=args.seed)
        metrics.insert(0, "alpha", alpha)
        per_model_tables.append(metrics)
        avg = metrics[metrics["source_model"] == "Average"].iloc[0]
        rows.append(
            {
                "dataset": args.dataset,
                "tag": args.tag,
                "alpha": alpha,
                "avg_auc": avg["auc"],
                "avg_ap": avg["ap"],
                "n_rows": len(scored),
                **stats,
            }
        )
        if args.save_fused_csv:
            scored.to_csv(args.output_dir / f"{args.dataset}_{args.tag}_alpha{alpha:.2f}.csv", index=False)

    summary = pd.DataFrame(rows)
    per_model = pd.concat(per_model_tables, ignore_index=True)
    summary_path = args.output_dir / f"{args.dataset}_{args.tag}_summary.csv"
    per_model_path = args.output_dir / f"{args.dataset}_{args.tag}_per_model.csv"
    md_path = args.output_dir / f"{args.dataset}_{args.tag}_summary.md"
    summary.to_csv(summary_path, index=False)
    per_model.to_csv(per_model_path, index=False)
    _write_markdown(md_path, summary, f"{args.dataset} / {args.tag}")

    best = summary.sort_values(["avg_auc", "avg_ap"], ascending=False).iloc[0]
    print(
        f"{args.dataset}/{args.tag}: rows={len(merged)} "
        f"best_alpha={best.alpha:.2f} AUC={best.avg_auc:.4f} AP={best.avg_ap:.4f}"
    )
    print(f"  汇总: {summary_path}")
    print(f"  逐模型: {per_model_path}")


if __name__ == "__main__":
    main()
