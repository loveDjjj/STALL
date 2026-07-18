#!/usr/bin/env python3
"""Combine per-dataset persistence scale sweeps into a global source audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-source-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-worst-csv", type=Path, required=True)
    args = parser.parse_args()

    rows = pd.concat([pd.read_csv(path) for path in args.per_source_csv], ignore_index=True)
    summaries = []
    worst_rows = []
    for scale, group in rows.groupby("scale", sort=True):
        summaries.append(
            {
                "scale": float(scale),
                "n_sources": int(len(group)),
                "n_negative_auc": int((group["delta_auc"] < 0).sum()),
                "n_negative_ap": int((group["delta_ap"] < 0).sum()),
                "mean_delta_auc": float(group["delta_auc"].mean()),
                "min_delta_auc": float(group["delta_auc"].min()),
                "mean_delta_ap": float(group["delta_ap"].mean()),
                "min_delta_ap": float(group["delta_ap"].min()),
                "weighted_mean_delta_auc": float((group["delta_auc"] * group["n_fake"]).sum() / group["n_fake"].sum()),
                "weighted_mean_delta_ap": float((group["delta_ap"] * group["n_fake"]).sum() / group["n_fake"].sum()),
                "worst_auc_dataset": str(group.sort_values(["delta_auc", "delta_ap"]).iloc[0]["dataset"]),
                "worst_auc_source": str(group.sort_values(["delta_auc", "delta_ap"]).iloc[0]["source_model"]),
                "worst_ap_dataset": str(group.sort_values(["delta_ap", "delta_auc"]).iloc[0]["dataset"]),
                "worst_ap_source": str(group.sort_values(["delta_ap", "delta_auc"]).iloc[0]["source_model"]),
            }
        )
        worst = group.sort_values(["delta_ap", "delta_auc", "dataset", "source_model"]).head(5).copy()
        worst_rows.append(worst)

    summary = pd.DataFrame(summaries).sort_values(
        ["n_negative_auc", "n_negative_ap", "min_delta_ap", "min_delta_auc", "mean_delta_auc"],
        ascending=[True, True, False, False, False],
    )
    worst_out = pd.concat(worst_rows, ignore_index=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_worst_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    worst_out.to_csv(args.output_worst_csv, index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved combined scale summary -> {args.output_csv}")
    print(f"Saved worst-source rows -> {args.output_worst_csv}")


if __name__ == "__main__":
    main()
