#!/usr/bin/env python3
"""Verify Hotshot-XL patch-cache prefill plan."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _row(check: str, passed: bool, detail: str) -> dict[str, object]:
    return {"check": check, "passed": bool(passed), "detail": detail}


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    exists = args.summary_csv.exists()
    rows.append(_row("summary_exists", exists, str(args.summary_csv)))
    if not exists:
        return pd.DataFrame(rows)
    summary = pd.read_csv(args.summary_csv)
    rows.append(_row("single_summary_row", len(summary) == 1, str(len(summary))))
    row = summary.iloc[0]
    rows.append(_row("csv_is_hotshot_target", str(row["csv"]) == str(args.expected_csv), str(row["csv"])))
    rows.append(_row("cache_is_videofeedback", str(row["patch_emb_cache"]) == str(args.expected_cache), str(row["patch_emb_cache"])))
    rows.append(_row("dry_run_status", str(row["status"]) == "DRY_RUN", str(row["status"])))
    rows.append(_row("execute_false", not bool(row["execute"]), str(row["execute"])))
    rows.append(_row("duration_1", int(row["duration"]) == 1, str(row["duration"])))
    rows.append(_row("compact_true", bool(row["compact"]), str(row["compact"])))
    rows.append(_row("misses_3251", int(row["misses_before"]) == 3251, str(row["misses_before"])))
    rows.append(_row("written_zero", int(row["written"]) == 0, str(row["written"])))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--expected-csv", type=Path, required=True)
    parser.add_argument("--expected-cache", type=Path, required=True)
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
