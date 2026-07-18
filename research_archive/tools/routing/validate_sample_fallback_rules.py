#!/usr/bin/env python3
"""Held-out validation over sample-level fallback candidate rules.

This consumes the per-source CSV files produced by
sample_level_fallback_sweep.py. It does not recompute scores; it validates
whether a fallback rule selected on training sources/datasets transfers to
held-out sources/datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RULE_COLUMNS = ["feature", "threshold", "direction", "real_policy"]
METRIC_COLUMNS = [
    "delta_vs_base_auc",
    "delta_vs_base_ap",
    "delta_vs_universal_auc",
    "delta_vs_universal_ap",
    "delta_vs_split_auc",
    "delta_vs_split_ap",
]


def _read_inputs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        missing = [c for c in ["dataset", "source_model", *RULE_COLUMNS, *METRIC_COLUMNS] if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["threshold_key"] = out["threshold"].fillna("__nan__").astype(str)
    return out


def _rule_columns() -> list[str]:
    return ["feature", "threshold_key", "direction", "real_policy"]


def _rule_label(row: pd.Series) -> str:
    threshold = row["threshold_key"]
    if threshold == "__nan__":
        return f"{row['feature']}/*/{row['direction']}/{row['real_policy']}"
    return f"{row['feature']}/{threshold}/{row['direction']}/{row['real_policy']}"


def _aggregate_rules(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(_rule_columns(), dropna=False)
    summary = grouped.agg(
        mean_delta_vs_base_auc=("delta_vs_base_auc", "mean"),
        min_delta_vs_base_auc=("delta_vs_base_auc", "min"),
        mean_delta_vs_base_ap=("delta_vs_base_ap", "mean"),
        min_delta_vs_base_ap=("delta_vs_base_ap", "min"),
        mean_delta_vs_universal_auc=("delta_vs_universal_auc", "mean"),
        min_delta_vs_universal_auc=("delta_vs_universal_auc", "min"),
        mean_delta_vs_universal_ap=("delta_vs_universal_ap", "mean"),
        min_delta_vs_universal_ap=("delta_vs_universal_ap", "min"),
        mean_delta_vs_split_auc=("delta_vs_split_auc", "mean"),
        min_delta_vs_split_auc=("delta_vs_split_auc", "min"),
        mean_delta_vs_split_ap=("delta_vs_split_ap", "mean"),
        min_delta_vs_split_ap=("delta_vs_split_ap", "min"),
        n=("source_model", "count"),
    ).reset_index()
    return summary


def _objective_tuple(row: pd.Series, objective: str) -> tuple[float, ...]:
    if objective == "robust_universal_then_base":
        return (
            row["min_delta_vs_universal_auc"],
            row["min_delta_vs_universal_ap"],
            row["min_delta_vs_base_auc"],
            row["min_delta_vs_base_ap"],
            row["mean_delta_vs_universal_auc"],
            row["mean_delta_vs_base_auc"],
        )
    if objective == "robust_base_then_universal":
        return (
            row["min_delta_vs_base_auc"],
            row["min_delta_vs_base_ap"],
            row["min_delta_vs_universal_auc"],
            row["min_delta_vs_universal_ap"],
            row["mean_delta_vs_base_auc"],
            row["mean_delta_vs_universal_auc"],
        )
    if objective == "mean_universal_then_base":
        return (
            row["mean_delta_vs_universal_auc"],
            row["mean_delta_vs_universal_ap"],
            row["min_delta_vs_universal_auc"],
            row["min_delta_vs_base_auc"],
            row["mean_delta_vs_base_auc"],
        )
    raise ValueError(f"Unknown objective: {objective}")


def _select_rule(train_rows: pd.DataFrame, objective: str) -> pd.Series:
    summary = _aggregate_rules(train_rows)
    best_idx = max(summary.index, key=lambda i: _objective_tuple(summary.loc[i], objective))
    return summary.loc[best_idx]


def _evaluate_rule(rows: pd.DataFrame, rule: pd.Series) -> dict[str, float]:
    mask = np.ones(len(rows), dtype=bool)
    for col in _rule_columns():
        mask &= rows[col].astype(str).to_numpy() == str(rule[col])
    selected = rows.loc[mask]
    if selected.empty:
        raise ValueError(f"No rows found for selected rule: {_rule_label(rule)}")
    agg = _aggregate_rules(selected).iloc[0]
    return {c: float(agg[c]) for c in agg.index if c not in _rule_columns()}


def _validate_leave_one_source(rows: pd.DataFrame, objective: str) -> pd.DataFrame:
    out = []
    for dataset, source in sorted(rows[["dataset", "source_model"]].drop_duplicates().itertuples(index=False)):
        heldout = (rows["dataset"] == dataset) & (rows["source_model"] == source)
        rule = _select_rule(rows.loc[~heldout], objective)
        metrics = _evaluate_rule(rows.loc[heldout], rule)
        out.append(
            {
                "validation_mode": "leave_one_source",
                "objective": objective,
                "heldout_dataset": dataset,
                "heldout_source": source,
                "selected_rule": _rule_label(rule),
                **metrics,
            }
        )
    return pd.DataFrame(out)


def _validate_leave_one_dataset(rows: pd.DataFrame, objective: str) -> pd.DataFrame:
    out = []
    for dataset in sorted(rows["dataset"].unique()):
        heldout = rows["dataset"] == dataset
        rule = _select_rule(rows.loc[~heldout], objective)
        metrics = _evaluate_rule(rows.loc[heldout], rule)
        out.append(
            {
                "validation_mode": "leave_one_dataset",
                "objective": objective,
                "heldout_dataset": dataset,
                "heldout_source": "__all__",
                "selected_rule": _rule_label(rule),
                **metrics,
            }
        )
    return pd.DataFrame(out)


def _summarize(validation: pd.DataFrame) -> pd.DataFrame:
    return (
        validation.groupby(["validation_mode", "objective", "selected_rule"], dropna=False)
        .agg(
            mean_delta_vs_base_auc=("mean_delta_vs_base_auc", "mean"),
            min_delta_vs_base_auc=("min_delta_vs_base_auc", "min"),
            mean_delta_vs_base_ap=("mean_delta_vs_base_ap", "mean"),
            min_delta_vs_base_ap=("min_delta_vs_base_ap", "min"),
            mean_delta_vs_universal_auc=("mean_delta_vs_universal_auc", "mean"),
            min_delta_vs_universal_auc=("min_delta_vs_universal_auc", "min"),
            mean_delta_vs_universal_ap=("mean_delta_vs_universal_ap", "mean"),
            min_delta_vs_universal_ap=("min_delta_vs_universal_ap", "min"),
            mean_delta_vs_split_auc=("mean_delta_vs_split_auc", "mean"),
            min_delta_vs_split_auc=("min_delta_vs_split_auc", "min"),
            mean_delta_vs_split_ap=("mean_delta_vs_split_ap", "mean"),
            min_delta_vs_split_ap=("min_delta_vs_split_ap", "min"),
            n=("heldout_source", "count"),
        )
        .reset_index()
        .sort_values(
            ["validation_mode", "min_delta_vs_universal_auc", "mean_delta_vs_universal_auc"],
            ascending=[True, False, False],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-source-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument(
        "--objectives",
        default="robust_universal_then_base,robust_base_then_universal,mean_universal_then_base",
    )
    args = parser.parse_args()

    rows = _read_inputs(args.per_source_csv)
    validations = []
    for objective in [x.strip() for x in args.objectives.split(",") if x.strip()]:
        validations.append(_validate_leave_one_source(rows, objective))
        validations.append(_validate_leave_one_dataset(rows, objective))
    validation = pd.concat(validations, ignore_index=True)
    summary = _summarize(validation)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    validation.to_csv(args.output_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved sample fallback rule validation -> {args.output_csv}")
    print(f"Saved sample fallback rule summary -> {args.summary_csv}")


if __name__ == "__main__":
    main()
