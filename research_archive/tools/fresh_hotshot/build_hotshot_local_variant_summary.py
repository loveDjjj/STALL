#!/usr/bin/env python3
"""Summarize the completed VideoFeedback Hotshot local-variant run."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _metrics(path: Path, score_col: str = "final_score") -> pd.DataFrame:
    df = pd.read_csv(path)
    real = df[df["subset"].astype(str).str.lower() == "real"]
    fake = df[df["subset"].astype(str).str.lower() != "real"]
    rows = []
    real_scores = real[score_col].to_numpy(float)
    for source, group in fake.groupby("source_model", sort=True):
        y = np.concatenate([np.ones(len(real_scores)), np.zeros(len(group))])
        s = np.concatenate([real_scores, group[score_col].to_numpy(float)])
        rows.append(
            {
                "source_model": source,
                "auc": float(roc_auc_score(y, s)),
                "ap": float(average_precision_score(y, s)),
                "n_fake": int(len(group)),
            }
        )
    out = pd.DataFrame(rows)
    out.loc[len(out)] = {
        "source_model": "Average",
        "auc": float(out["auc"].mean()),
        "ap": float(out["ap"].mean()),
        "n_fake": int(out["n_fake"].sum()),
    }
    return out


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = _metrics(args.global_csv, args.global_score_col)
    final = _metrics(args.final_scores_csv, args.final_score_col)
    merged = baseline.merge(final, on="source_model", suffixes=("_global", "_final"))
    merged["delta_auc"] = merged["auc_final"] - merged["auc_global"]
    merged["delta_ap"] = merged["ap_final"] - merged["ap_global"]
    avg = merged[merged["source_model"] == "Average"].iloc[0]
    hotshot = merged[merged["source_model"] == "Hotshot-XL"].iloc[0]
    summary = pd.DataFrame(
        [
            {
                "candidate": "videofeedback_hotshot_local_variant",
                "decision": "COMPLETED_LOCAL_VARIANT_WITH_DURATION_CAVEAT",
                "target_rows": int(pd.read_csv(args.final_scores_csv).shape[0]),
                "source_count": int(len(final) - 1),
                "global_avg_auc": float(avg["auc_global"]),
                "global_avg_ap": float(avg["ap_global"]),
                "final_avg_auc": float(avg["auc_final"]),
                "final_avg_ap": float(avg["ap_final"]),
                "delta_avg_auc": float(avg["delta_auc"]),
                "delta_avg_ap": float(avg["delta_ap"]),
                "hotshot_xl_auc": float(hotshot["auc_final"]),
                "hotshot_xl_ap": float(hotshot["ap_final"]),
                "hotshot_xl_delta_auc": float(hotshot["delta_auc"]),
                "hotshot_xl_delta_ap": float(hotshot["delta_ap"]),
                "required_label": "local variant with duration=1 target and duration=2 real-calibration caveat",
            }
        ]
    )
    return summary, merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-csv", type=Path, required=True)
    parser.add_argument("--final-scores-csv", type=Path, required=True)
    parser.add_argument("--global-score-col", default="final_score")
    parser.add_argument("--final-score-col", default="final_score")
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    parser.add_argument("--output-per-source-csv", type=Path, required=True)
    args = parser.parse_args()

    summary, per_source = run(args)
    args.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_per_source_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary_csv, index=False)
    per_source.to_csv(args.output_per_source_csv, index=False)
    print(summary.to_string(index=False))
    print(f"Saved summary -> {args.output_summary_csv}")
    print(f"Saved per-source -> {args.output_per_source_csv}")


if __name__ == "__main__":
    main()
