#!/usr/bin/env python3
"""Promotion gate for the fixed sample-level fallback rule.

This script turns the research decision into an explicit PASS/FAIL check. It
expects held-out validation summaries from validate_sample_fallback_rules.py and
optionally final-score summaries from apply_sample_fallback_rule.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_VALIDATION_COLUMNS = [
    "validation_mode",
    "objective",
    "selected_rule",
    "mean_delta_vs_base_auc",
    "min_delta_vs_base_auc",
    "mean_delta_vs_base_ap",
    "min_delta_vs_base_ap",
    "mean_delta_vs_universal_auc",
    "min_delta_vs_universal_auc",
    "mean_delta_vs_universal_ap",
    "min_delta_vs_universal_ap",
]

REQUIRED_SOURCE_AUDIT_COLUMNS = [
    "dataset",
    "n_sources",
    "n_negative_auc",
    "n_negative_ap",
    "min_delta_auc",
    "min_delta_ap",
]


def _check_columns(df: pd.DataFrame, path: Path, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")


def _load_validation(path: Path, rule: str, objective: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    _check_columns(df, path, REQUIRED_VALIDATION_COLUMNS)
    selected = df[(df["selected_rule"] == rule) & (df["objective"] == objective)].copy()
    if selected.empty:
        raise ValueError(f"No validation rows found for rule={rule!r}, objective={objective!r}")
    return selected


def _metric(rows: pd.DataFrame, validation_mode: str, metric: str) -> float:
    mode_rows = rows[rows["validation_mode"] == validation_mode]
    if mode_rows.empty:
        raise ValueError(f"Missing validation mode: {validation_mode}")
    return float(mode_rows[metric].min())


def _add_check(checks: list[dict[str, object]], name: str, actual: float, threshold: float) -> None:
    checks.append(
        {
            "check": name,
            "actual": actual,
            "threshold": threshold,
            "passed": bool(actual >= threshold),
        }
    )


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = _load_validation(args.validation_summary_csv, args.rule, args.objective)
    checks: list[dict[str, object]] = []
    for mode in ["leave_one_source", "leave_one_dataset"]:
        _add_check(
            checks,
            f"{mode}: min_delta_vs_base_auc",
            _metric(rows, mode, "min_delta_vs_base_auc"),
            args.min_delta_vs_base_auc,
        )
        _add_check(
            checks,
            f"{mode}: min_delta_vs_base_ap",
            _metric(rows, mode, "min_delta_vs_base_ap"),
            args.min_delta_vs_base_ap,
        )
        _add_check(
            checks,
            f"{mode}: min_delta_vs_universal_auc",
            _metric(rows, mode, "min_delta_vs_universal_auc"),
            args.min_delta_vs_universal_auc,
        )
        _add_check(
            checks,
            f"{mode}: min_delta_vs_universal_ap",
            _metric(rows, mode, "min_delta_vs_universal_ap"),
            args.min_delta_vs_universal_ap,
        )
        _add_check(
            checks,
            f"{mode}: mean_delta_vs_universal_auc",
            _metric(rows, mode, "mean_delta_vs_universal_auc"),
            args.min_mean_delta_vs_universal_auc,
        )
        _add_check(
            checks,
            f"{mode}: mean_delta_vs_universal_ap",
            _metric(rows, mode, "mean_delta_vs_universal_ap"),
            args.min_mean_delta_vs_universal_ap,
        )

    if args.final_score_summary_csv is not None:
        final_summary = pd.read_csv(args.final_score_summary_csv)
        _check_columns(
            final_summary,
            args.final_score_summary_csv,
            ["dataset", "avg_auc", "avg_ap", "min_source_auc", "min_source_ap", "n_sources"],
        )
        real_datasets = final_summary[final_summary["dataset"] != "Mean"]
        _add_check(
            checks,
            "final_score_summary: n_datasets",
            float(len(real_datasets)),
            float(args.min_datasets),
        )
        _add_check(
            checks,
            "final_score_summary: n_sources",
            float(real_datasets["n_sources"].sum()),
            float(args.min_sources),
        )

    if args.source_audit_summary_csv is not None:
        source_summary = pd.read_csv(args.source_audit_summary_csv)
        _check_columns(source_summary, args.source_audit_summary_csv, REQUIRED_SOURCE_AUDIT_COLUMNS)
        all_rows = source_summary[source_summary["dataset"] == "ALL"]
        if all_rows.empty:
            raise ValueError(f"{args.source_audit_summary_csv} missing dataset='ALL' aggregate row")
        all_row = all_rows.iloc[0]
        _add_check(
            checks,
            "source_audit: n_sources",
            float(all_row["n_sources"]),
            float(args.min_sources),
        )
        _add_check(
            checks,
            "source_audit: n_negative_auc",
            -float(all_row["n_negative_auc"]),
            -float(args.max_negative_source_auc),
        )
        _add_check(
            checks,
            "source_audit: n_negative_ap",
            -float(all_row["n_negative_ap"]),
            -float(args.max_negative_source_ap),
        )
        _add_check(
            checks,
            "source_audit: min_delta_auc",
            float(all_row["min_delta_auc"]),
            args.min_source_delta_auc,
        )
        _add_check(
            checks,
            "source_audit: min_delta_ap",
            float(all_row["min_delta_ap"]),
            args.min_source_delta_ap,
        )

    checks_df = pd.DataFrame(checks)
    decision = pd.DataFrame(
        [
            {
                "rule": args.rule,
                "objective": args.objective,
                "decision": "PASS" if bool(checks_df["passed"].all()) else "FAIL",
                "n_checks": len(checks_df),
                "n_passed": int(checks_df["passed"].sum()),
            }
        ]
    )
    return decision, checks_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-summary-csv", type=Path, required=True)
    parser.add_argument("--final-score-summary-csv", type=Path, default=None)
    parser.add_argument("--source-audit-summary-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--checks-csv", type=Path, required=True)
    parser.add_argument("--rule", default="persistence_minus_base/0.0/ge/rule")
    parser.add_argument("--objective", default="robust_universal_then_base")
    parser.add_argument("--min-delta-vs-base-auc", type=float, default=0.0)
    parser.add_argument("--min-delta-vs-base-ap", type=float, default=0.0)
    parser.add_argument("--min-delta-vs-universal-auc", type=float, default=0.0)
    parser.add_argument("--min-delta-vs-universal-ap", type=float, default=0.0)
    parser.add_argument("--min-mean-delta-vs-universal-auc", type=float, default=0.001)
    parser.add_argument("--min-mean-delta-vs-universal-ap", type=float, default=0.001)
    parser.add_argument("--min-datasets", type=int, default=3)
    parser.add_argument("--min-sources", type=int, default=20)
    parser.add_argument("--max-negative-source-auc", type=int, default=0)
    parser.add_argument("--max-negative-source-ap", type=int, default=0)
    parser.add_argument("--min-source-delta-auc", type=float, default=0.0)
    parser.add_argument("--min-source-delta-ap", type=float, default=0.0)
    args = parser.parse_args()

    decision, checks = run(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.checks_csv.parent.mkdir(parents=True, exist_ok=True)
    decision.to_csv(args.output_csv, index=False)
    checks.to_csv(args.checks_csv, index=False)
    print(decision.to_string(index=False))
    print(checks.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved promotion decision -> {args.output_csv}")
    print(f"Saved promotion checks -> {args.checks_csv}")


if __name__ == "__main__":
    main()
