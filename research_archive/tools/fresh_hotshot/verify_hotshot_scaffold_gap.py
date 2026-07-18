#!/usr/bin/env python3
"""Verify VideoFeedback Hotshot scaffold-gap analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _row(check: str, passed: bool, detail: str) -> dict[str, object]:
    return {"check": check, "passed": bool(passed), "detail": detail}


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    paths = {
        "summary": args.summary_csv,
        "source_gap": args.source_gap_csv,
        "decision": args.decision_csv,
    }
    for label, path in paths.items():
        rows.append(_row(f"exists:{label}", path.exists(), str(path)))
    if not all(path.exists() for path in paths.values()):
        return pd.DataFrame(rows)

    summary = pd.read_csv(args.summary_csv)
    source_gap = pd.read_csv(args.source_gap_csv)
    decision = pd.read_csv(args.decision_csv)

    rows.append(_row("summary_targets_3", len(summary) == 3, str(len(summary))))
    rows.append(_row("decision_single_row", len(decision) == 1, str(len(decision))))
    rows.append(_row("hotshot_rows_37661", int(decision.iloc[0]["hotshot_rows"]) == 37661, str(decision.iloc[0]["hotshot_rows"])))
    rows.append(_row("shared_triplet_rows_37661", int(decision.iloc[0]["shared_triplet_rows"]) == 37661, str(decision.iloc[0]["shared_triplet_rows"])))
    rows.append(_row("missing_full_rows_0", int(decision.iloc[0]["missing_full_rows"]) == 0, str(decision.iloc[0]["missing_full_rows"])))
    rows.append(_row("full_hotshot_ready", bool(decision.iloc[0]["full_hotshot_scaffold_ready"]), str(decision.iloc[0]["full_hotshot_scaffold_ready"])))
    rows.append(_row("shared_subset_ready", bool(decision.iloc[0]["shared_subset_scaffold_ready"]), str(decision.iloc[0]["shared_subset_scaffold_ready"])))
    rows.append(_row("no_blocking_sources", str(decision.iloc[0]["blocking_source_models"]) in {"", "nan"}, str(decision.iloc[0]["blocking_source_models"])))
    rows.append(_row("source_gap_empty", len(source_gap) == 0, str(len(source_gap))))
    rows.append(_row("all_coverage_37661_over_37661", bool((summary["matched_rows"].astype(int) == 37661).all()), summary[["target", "matched_rows"]].to_string(index=False)))
    rows.append(_row("all_missing_0", bool((summary["missing_rows"].astype(int) == 0).all()), summary[["target", "missing_rows"]].to_string(index=False)))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--source-gap-csv", type=Path, required=True)
    parser.add_argument("--decision-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    checks = run(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(args.output_csv, index=False)
    n_passed = int(checks["passed"].sum())
    print(f"PASS {n_passed}/{len(checks)}" if n_passed == len(checks) else f"FAIL {n_passed}/{len(checks)}")
    print(checks.to_string(index=False))
    if n_passed != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
