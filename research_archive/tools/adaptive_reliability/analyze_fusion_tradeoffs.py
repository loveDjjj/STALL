#!/usr/bin/env python3
"""Analyze universal fusion tradeoffs and per-generator effects."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--per-model-glob", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.summary_csv)
    for beta_ap in [0.0, 0.25, 0.5, 1.0]:
        for beta_min in [0.0, 0.25, 0.5, 1.0]:
            summary[f"objective_ap{beta_ap}_min{beta_min}"] = (
                summary["mean_auc"]
                + beta_ap * summary["mean_ap"]
                + beta_min * summary["min_auc"]
            )
    objective_cols = [c for c in summary.columns if c.startswith("objective_")]
    rows = []
    for col in objective_cols:
        best = summary.sort_values([col, "mean_auc", "mean_ap"], ascending=False).iloc[0]
        rows.append({"objective": col, **best.to_dict()})
    pd.DataFrame(rows).to_csv(args.output_dir / "objective_best_weights.csv", index=False)

    per_model_frames = []
    for path in sorted(Path().glob(args.per_model_glob)):
        df = pd.read_csv(path)
        per_model_frames.append(df)
    per_model = pd.concat(per_model_frames, ignore_index=True)
    per_model.to_csv(args.output_dir / "all_per_model.csv", index=False)

    # Use the top mean-AUC universal weights from the summary.
    top = summary.sort_values(["mean_auc", "mean_ap"], ascending=False).iloc[0]
    key_cols = [c for c in ["wa", "wb", "wc", "alpha", "method"] if c in per_model.columns and c in top.index]
    selected = per_model.copy()
    for col in key_cols:
        selected = selected[selected[col] == top[col]]
    selected.to_csv(args.output_dir / "top_universal_per_model.csv", index=False)

    print(pd.DataFrame(rows).head(12).to_string(index=False), flush=True)
    print(f"Top universal weights: {top.to_dict()}", flush=True)


if __name__ == "__main__":
    main()
