#!/usr/bin/env python3
"""Summarize per-source fallback-vs-default audit files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--summary-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    parser.add_argument("--output-worst-csv", type=Path, required=True)
    parser.add_argument("--n-worst", type=int, default=10)
    args = parser.parse_args()

    summaries = pd.concat([pd.read_csv(path) for path in args.summary_csv], ignore_index=True)
    audits = pd.concat([pd.read_csv(path) for path in args.audit_csv], ignore_index=True)

    total = {
        "dataset": "ALL",
        "n_sources": int(summaries["n_sources"].sum()),
        "n_negative_auc": int(summaries["n_negative_auc"].sum()),
        "n_negative_ap": int(summaries["n_negative_ap"].sum()),
        "min_delta_auc": float(audits["delta_auc"].min()),
        "min_delta_ap": float(audits["delta_ap"].min()),
        "mean_delta_auc": float(audits["delta_auc"].mean()),
        "mean_delta_ap": float(audits["delta_ap"].mean()),
        "weighted_mean_delta_auc": float((audits["delta_auc"] * audits["n_fake"]).sum() / audits["n_fake"].sum()),
        "weighted_mean_delta_ap": float((audits["delta_ap"] * audits["n_fake"]).sum() / audits["n_fake"].sum()),
        "worst_auc_source": str(audits.sort_values(["delta_auc", "delta_ap"]).iloc[0]["source_model"]),
        "worst_ap_source": str(audits.sort_values(["delta_ap", "delta_auc"]).iloc[0]["source_model"]),
    }
    out_summary = pd.concat([summaries, pd.DataFrame([total])], ignore_index=True)
    worst = audits.sort_values(["delta_auc", "delta_ap", "dataset", "source_model"]).head(args.n_worst)

    args.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_worst_csv.parent.mkdir(parents=True, exist_ok=True)
    out_summary.to_csv(args.output_summary_csv, index=False)
    worst.to_csv(args.output_worst_csv, index=False)
    print(out_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved summary -> {args.output_summary_csv}")
    print(f"Saved worst sources -> {args.output_worst_csv}")


if __name__ == "__main__":
    main()
