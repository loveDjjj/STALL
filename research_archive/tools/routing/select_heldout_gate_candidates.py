#!/usr/bin/env python3
"""Select held-out gate candidates from a precomputed candidate table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _select(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (_, mode), group in candidates.groupby(["heldout_source", "apply_mode"], sort=True):
        eligible = group[group["train_passes_nonregression"]].copy()
        if eligible.empty:
            selected = group.sort_values(
                ["selection_score", "train_mean_delta_auc", "train_mean_delta_ap"],
                ascending=False,
            ).iloc[0].copy()
            selected["selected_from"] = "best_training_score_no_nonregression_pass"
        else:
            selected = eligible.sort_values(
                ["train_mean_delta_auc", "train_mean_delta_ap", "train_min_delta_ap"],
                ascending=False,
            ).iloc[0].copy()
            selected["selected_from"] = "best_training_nonregression_pass"
        rows.append(selected)
    return pd.DataFrame(rows)


def _summary(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, feature_family, apply_mode), group in selected.groupby(["dataset", "feature_family", "apply_mode"], dropna=False):
        rows.append(
            {
                "dataset": dataset,
                "feature_family": feature_family,
                "apply_mode": apply_mode,
                "n_heldout_sources": int(len(group)),
                "mean_heldout_delta_auc": float(group["heldout_delta_auc"].mean()),
                "mean_heldout_delta_ap": float(group["heldout_delta_ap"].mean()),
                "min_heldout_delta_auc": float(group["heldout_delta_auc"].min()),
                "min_heldout_delta_ap": float(group["heldout_delta_ap"].min()),
                "n_negative_heldout_auc": int((group["heldout_delta_auc"] < 0).sum()),
                "n_negative_heldout_ap": int((group["heldout_delta_ap"] < 0).sum()),
                "mean_train_delta_auc": float(group["train_mean_delta_auc"].mean()),
                "mean_train_delta_ap": float(group["train_mean_delta_ap"].mean()),
                "total_triggered_real": int(group["n_triggered_real"].sum()),
                "total_triggered_train_fake": int(group["n_triggered_train_fake"].sum()),
                "total_triggered_heldout_fake": int(group["n_triggered_heldout_fake"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["apply_mode", "mean_heldout_delta_auc", "mean_heldout_delta_ap"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--output-selected-csv", type=Path, required=True)
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    args = parser.parse_args()

    candidates = pd.read_csv(args.candidates_csv)
    selected = _select(candidates)
    summary = _summary(selected)
    args.output_selected_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_selected_csv, index=False)
    summary.to_csv(args.output_summary_csv, index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved selected -> {args.output_selected_csv}")
    print(f"Saved summary -> {args.output_summary_csv}")


if __name__ == "__main__":
    main()
