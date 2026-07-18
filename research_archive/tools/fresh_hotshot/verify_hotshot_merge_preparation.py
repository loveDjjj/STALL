#!/usr/bin/env python3
"""Verify Hotshot merge-preparation decision state."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _row(check: str, passed: bool, detail: str) -> dict[str, object]:
    return {"check": check, "passed": bool(passed), "detail": detail}


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    decision_exists = args.decision_csv.exists()
    coverage_exists = args.coverage_csv.exists()
    rows.append(_row("decision_exists", decision_exists, str(args.decision_csv)))
    rows.append(_row("coverage_exists", coverage_exists, str(args.coverage_csv)))
    if not decision_exists or not coverage_exists:
        return pd.DataFrame(rows)

    decision = pd.read_csv(args.decision_csv)
    try:
        coverage = pd.read_csv(args.coverage_csv)
    except pd.errors.EmptyDataError:
        coverage = pd.DataFrame()
    rows.append(_row("single_decision_row", len(decision) == 1, str(len(decision))))
    state = str(decision.iloc[0]["decision"])
    rows.append(_row("known_decision_state", state in {"WAITING_FOR_HOTSHOT_COMPLETION", "MERGED_FULL_TRIPLET_READY", "MERGED_BUT_INCOMPLETE"}, state))
    if state == "WAITING_FOR_HOTSHOT_COMPLETION":
        rows.append(_row("waiting_patch_missing", not bool(decision.iloc[0]["hotshot_patch_exists"]), str(decision.iloc[0]["hotshot_patch_exists"])))
        rows.append(_row("waiting_persistence_missing", not bool(decision.iloc[0]["hotshot_persistence_exists"]), str(decision.iloc[0]["hotshot_persistence_exists"])))
        rows.append(_row("waiting_coverage_empty", len(coverage) == 0, str(len(coverage))))
        rows.append(_row("waiting_no_merged_patch", not args.merged_patch_csv.exists(), str(args.merged_patch_csv)))
        rows.append(_row("waiting_no_merged_persistence", not args.merged_persistence_csv.exists(), str(args.merged_persistence_csv)))
    elif state == "MERGED_FULL_TRIPLET_READY":
        rows.append(_row("merged_patch_exists", args.merged_patch_csv.exists(), str(args.merged_patch_csv)))
        rows.append(_row("merged_persistence_exists", args.merged_persistence_csv.exists(), str(args.merged_persistence_csv)))
        rows.append(_row("coverage_single_row", len(coverage) == 1, str(len(coverage))))
        if len(coverage) == 1:
            rows.append(_row("coverage_global_rows_37661", int(coverage.iloc[0]["global_rows"]) == 37661, str(coverage.iloc[0]["global_rows"])))
            rows.append(_row("coverage_full_ready", bool(coverage.iloc[0]["full_triplet_ready"]), str(coverage.iloc[0]["full_triplet_ready"])))
    else:
        rows.append(_row("incomplete_state_is_not_ready", state == "MERGED_BUT_INCOMPLETE", state))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-csv", type=Path, required=True)
    parser.add_argument("--coverage-csv", type=Path, required=True)
    parser.add_argument("--merged-patch-csv", type=Path, required=True)
    parser.add_argument("--merged-persistence-csv", type=Path, required=True)
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
