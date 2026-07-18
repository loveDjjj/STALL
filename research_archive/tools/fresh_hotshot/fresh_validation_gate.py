#!/usr/bin/env python3
"""Gate a single fresh validation run for the frozen sample-fallback scorer."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _check_columns(df: pd.DataFrame, path: Path, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")


def _add_check(rows: list[dict[str, object]], check: str, actual: float, threshold: float) -> None:
    rows.append({"check": check, "actual": actual, "threshold": threshold, "passed": bool(actual >= threshold)})


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(args.metrics_csv)
    audit_summary = pd.read_csv(args.source_audit_summary_csv)
    _check_columns(metrics, args.metrics_csv, ["source_model", "auc", "ap", "n_fake"])
    _check_columns(
        audit_summary,
        args.source_audit_summary_csv,
        ["n_sources", "n_negative_auc", "n_negative_ap", "min_delta_auc", "min_delta_ap", "mean_delta_auc", "mean_delta_ap"],
    )
    avg = metrics[metrics["source_model"] == "Average"]
    if len(avg) != 1:
        raise ValueError(f"{args.metrics_csv} expected exactly one Average row, found {len(avg)}")
    audit = audit_summary.iloc[0]

    checks: list[dict[str, object]] = []
    _add_check(checks, "metrics: avg_auc", float(avg.iloc[0]["auc"]), args.min_avg_auc)
    _add_check(checks, "metrics: avg_ap", float(avg.iloc[0]["ap"]), args.min_avg_ap)
    _add_check(checks, "metrics: n_sources", float(len(metrics[metrics["source_model"] != "Average"])), args.min_sources)
    _add_check(checks, "source_audit: n_sources", float(audit["n_sources"]), args.min_sources)
    _add_check(checks, "source_audit: n_negative_auc", -float(audit["n_negative_auc"]), -float(args.max_negative_source_auc))
    _add_check(checks, "source_audit: n_negative_ap", -float(audit["n_negative_ap"]), -float(args.max_negative_source_ap))
    _add_check(checks, "source_audit: min_delta_auc", float(audit["min_delta_auc"]), args.min_source_delta_auc)
    _add_check(checks, "source_audit: min_delta_ap", float(audit["min_delta_ap"]), args.min_source_delta_ap)
    _add_check(checks, "source_audit: mean_delta_auc", float(audit["mean_delta_auc"]), args.min_mean_delta_auc)
    _add_check(checks, "source_audit: mean_delta_ap", float(audit["mean_delta_ap"]), args.min_mean_delta_ap)

    checks_df = pd.DataFrame(checks)
    decision = pd.DataFrame(
        [
            {
                "dataset": args.dataset,
                "decision": "PASS" if bool(checks_df["passed"].all()) else "FAIL",
                "n_checks": len(checks_df),
                "n_passed": int(checks_df["passed"].sum()),
            }
        ]
    )
    return decision, checks_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--source-audit-summary-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--checks-csv", type=Path, required=True)
    parser.add_argument("--min-avg-auc", type=float, default=0.0)
    parser.add_argument("--min-avg-ap", type=float, default=0.0)
    parser.add_argument("--min-sources", type=int, default=1)
    parser.add_argument("--max-negative-source-auc", type=int, default=0)
    parser.add_argument("--max-negative-source-ap", type=int, default=0)
    parser.add_argument("--min-source-delta-auc", type=float, default=0.0)
    parser.add_argument("--min-source-delta-ap", type=float, default=0.0)
    parser.add_argument("--min-mean-delta-auc", type=float, default=0.0)
    parser.add_argument("--min-mean-delta-ap", type=float, default=0.0)
    args = parser.parse_args()

    decision, checks = run(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.checks_csv.parent.mkdir(parents=True, exist_ok=True)
    decision.to_csv(args.output_csv, index=False)
    checks.to_csv(args.checks_csv, index=False)
    print(decision.to_string(index=False))
    print(checks.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    if decision.iloc[0]["decision"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
