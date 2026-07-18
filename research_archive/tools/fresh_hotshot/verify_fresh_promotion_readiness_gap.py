#!/usr/bin/env python3
"""Verify the current promotion-grade fresh-validation readiness gap."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _row(check: str, passed: bool, detail: str) -> dict[str, object]:
    return {"check": check, "passed": bool(passed), "detail": detail}


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    exists = args.readiness_csv.exists()
    rows.append(_row("readiness_gap_exists", exists, str(args.readiness_csv)))
    if not exists:
        return pd.DataFrame(rows)

    df = pd.read_csv(args.readiness_csv)
    rows.append(_row("single_row", len(df) == 1, f"rows={len(df)}"))
    required = [
        "status",
        "pipeline_passed",
        "fresh_branch_status",
        "candidate_count",
        "scaffold_ready_count",
        "fresh_like_ready_count",
        "internal_or_variant_ready_count",
        "hotshot_status",
        "hotshot_scaffold_ready",
        "hotshot_duration_caveat",
        "blocking_reason",
        "next_action",
    ]
    missing = [col for col in required if col not in df.columns]
    rows.append(_row("required_columns", not missing, str(missing)))
    if missing or len(df) != 1:
        return pd.DataFrame(rows)

    item = df.iloc[0]
    rows.extend(
        [
            _row("status_waiting_for_true_fresh", str(item["status"]) == "WAITING_FOR_TRUE_FRESH_DATASET", str(item["status"])),
            _row("pipeline_passed", bool(item["pipeline_passed"]), str(item["pipeline_passed"])),
            _row("fresh_branch_ready_not_run", str(item["fresh_branch_status"]) == "READY_NOT_RUN", str(item["fresh_branch_status"])),
            _row("candidate_count_5", int(item["candidate_count"]) == 5, str(item["candidate_count"])),
            _row("scaffold_ready_count_5", int(item["scaffold_ready_count"]) == 5, str(item["scaffold_ready_count"])),
            _row("fresh_like_ready_count_0", int(item["fresh_like_ready_count"]) == 0, str(item["fresh_like_ready_count"])),
            _row("internal_or_variant_ready_count_5", int(item["internal_or_variant_ready_count"]) == 5, str(item["internal_or_variant_ready_count"])),
            _row(
                "hotshot_status_duration_caveat",
                str(item["hotshot_status"]) == "NOT_FRESH_LOCAL_VARIANT_WITH_DURATION_CAVEAT",
                str(item["hotshot_status"]),
            ),
            _row("hotshot_scaffold_ready", bool(item["hotshot_scaffold_ready"]), str(item["hotshot_scaffold_ready"])),
            _row("hotshot_duration_caveat", bool(item["hotshot_duration_caveat"]), str(item["hotshot_duration_caveat"])),
            _row("blocking_reason_mentions_no_fresh_like", "no scaffold-ready candidate" in str(item["blocking_reason"]), str(item["blocking_reason"])),
            _row("next_action_mentions_protocol_compatible", "protocol-compatible fresh dataset" in str(item["next_action"]), str(item["next_action"])),
        ]
    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readiness-csv", type=Path, required=True)
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
