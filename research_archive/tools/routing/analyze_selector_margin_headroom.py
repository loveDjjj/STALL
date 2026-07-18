#!/usr/bin/env python3
"""Analyze selector margin headroom from per-source fallback sweep outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RULE_COLUMNS = ["feature", "threshold", "direction", "real_policy"]


def _rule_id(row: pd.Series) -> str:
    threshold = row["threshold"]
    if pd.isna(threshold):
        threshold_text = ""
    else:
        threshold_text = f"{float(threshold):g}"
    return f"{row['feature']}/{threshold_text}/{row['direction']}/{row['real_policy']}"


def _select_rule(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    ids = df.apply(_rule_id, axis=1)
    out = df[ids == rule].copy()
    if out.empty:
        raise ValueError(f"Rule not found: {rule}")
    return out


def _summarize(rows: pd.DataFrame, label: str) -> dict[str, object]:
    return {
        "label": label,
        "rule": _rule_id(rows.iloc[0]),
        "n_sources": int(len(rows)),
        "mean_delta_vs_base_auc": float(rows["delta_vs_base_auc"].mean()),
        "min_delta_vs_base_auc": float(rows["delta_vs_base_auc"].min()),
        "mean_delta_vs_base_ap": float(rows["delta_vs_base_ap"].mean()),
        "min_delta_vs_base_ap": float(rows["delta_vs_base_ap"].min()),
        "worst_auc_source": str(rows.sort_values(["delta_vs_base_auc", "delta_vs_base_ap"]).iloc[0]["source_model"]),
        "worst_ap_source": str(rows.sort_values(["delta_vs_base_ap", "delta_vs_base_auc"]).iloc[0]["source_model"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-source-csv", type=Path, required=True)
    parser.add_argument("--current-rule", default="split_minus_universal/0/lt/split")
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    parser.add_argument("--output-source-csv", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.per_source_csv)
    missing = [
        c
        for c in [
            *RULE_COLUMNS,
            "source_model",
            "base_auc",
            "base_ap",
            "split_auc",
            "split_ap",
            "fallback_auc",
            "fallback_ap",
            "delta_vs_base_auc",
            "delta_vs_base_ap",
        ]
        if c not in df.columns
    ]
    if missing:
        raise ValueError(f"{args.per_source_csv} missing columns: {missing}")

    current = _select_rule(df, args.current_rule)
    always_split = _select_rule(df, "always//ge/split")

    grouped = []
    for _, rows in df.groupby(RULE_COLUMNS, dropna=False, sort=False):
        grouped.append(_summarize(rows, "candidate"))
    candidates = pd.DataFrame(grouped)
    nonnegative = candidates[
        (candidates["min_delta_vs_base_auc"] >= 0.0) & (candidates["min_delta_vs_base_ap"] >= 0.0)
    ].copy()
    best_ap_margin_rule = nonnegative.sort_values(
        ["min_delta_vs_base_ap", "min_delta_vs_base_auc", "mean_delta_vs_base_ap"],
        ascending=False,
    ).iloc[0]["rule"]
    best_auc_margin_rule = nonnegative.sort_values(
        ["min_delta_vs_base_auc", "min_delta_vs_base_ap", "mean_delta_vs_base_auc"],
        ascending=False,
    ).iloc[0]["rule"]
    best_ap_margin = _select_rule(df, str(best_ap_margin_rule))
    best_auc_margin = _select_rule(df, str(best_auc_margin_rule))

    summary = pd.DataFrame(
        [
            _summarize(current, "current"),
            _summarize(always_split, "always_split_upper_bound"),
            _summarize(best_ap_margin, "best_nonnegative_min_ap"),
            _summarize(best_auc_margin, "best_nonnegative_min_auc"),
        ]
    )
    current_sources = current[
        [
            "source_model",
            "base_auc",
            "base_ap",
            "universal_auc",
            "universal_ap",
            "split_auc",
            "split_ap",
            "fallback_auc",
            "fallback_ap",
            "delta_vs_base_auc",
            "delta_vs_base_ap",
        ]
    ].copy()
    current_sources["split_headroom_auc"] = current_sources["split_auc"] - current_sources["fallback_auc"]
    current_sources["split_headroom_ap"] = current_sources["split_ap"] - current_sources["fallback_ap"]
    current_sources = current_sources.sort_values(["delta_vs_base_ap", "delta_vs_base_auc", "source_model"])

    args.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_source_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary_csv, index=False)
    current_sources.to_csv(args.output_source_csv, index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved summary -> {args.output_summary_csv}")
    print(f"Saved source headroom -> {args.output_source_csv}")


if __name__ == "__main__":
    main()
