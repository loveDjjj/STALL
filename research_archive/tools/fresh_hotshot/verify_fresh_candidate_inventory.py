#!/usr/bin/env python3
"""Verify the local fresh-candidate inventory."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXPECTED_CANDIDATES = {"comgenvid", "genvideo", "videofeedback", "genvideo_1s", "videofeedback_hotshot"}
INTERNAL_OR_VARIANT = {"comgenvid", "genvideo", "videofeedback", "genvideo_1s"}


def _row(check: str, passed: bool, detail: str) -> dict[str, object]:
    return {"check": check, "passed": bool(passed), "detail": detail}


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    exists = args.inventory_csv.exists()
    rows.append(_row("inventory_exists", exists, str(args.inventory_csv)))
    if not exists:
        return pd.DataFrame(rows)

    df = pd.read_csv(args.inventory_csv)
    required_columns = [
        "candidate",
        "fresh_status",
        "scaffold_ready",
        "overlaps_manifest_inputs",
        "global_exists",
        "raw_patch_exists",
        "persistence_exists",
        "n_real",
        "n_fake",
        "n_fake_sources",
    ]
    missing = [col for col in required_columns if col not in df.columns]
    rows.append(_row("required_columns", not missing, str(missing)))
    if missing:
        return pd.DataFrame(rows)

    candidates = set(df["candidate"].astype(str))
    rows.append(_row("expected_candidates_present", EXPECTED_CANDIDATES.issubset(candidates), str(sorted(candidates))))
    ready = df[df["scaffold_ready"].astype(bool)]
    rows.append(_row("no_local_ready_fresh_candidate", not (ready["fresh_status"].astype(str) == "LOCAL_READY_NEEDS_PROVENANCE").any(), ready[["candidate", "fresh_status"]].to_string(index=False)))
    for candidate in INTERNAL_OR_VARIANT:
        row = df[df["candidate"] == candidate]
        rows.append(_row(f"{candidate}:present", len(row) == 1, str(len(row))))
        if len(row) == 1:
            rows.append(_row(f"{candidate}:not_fresh", str(row.iloc[0]["fresh_status"]) == "NOT_FRESH_INTERNAL_OR_VARIANT", str(row.iloc[0]["fresh_status"])))
            rows.append(_row(f"{candidate}:scaffold_ready", bool(row.iloc[0]["scaffold_ready"]), str(row.iloc[0]["scaffold_ready"])))
    hotshot = df[df["candidate"] == "videofeedback_hotshot"]
    rows.append(_row("videofeedback_hotshot:present", len(hotshot) == 1, str(len(hotshot))))
    if len(hotshot) == 1:
        rows.append(_row("videofeedback_hotshot:not_fresh_duration_caveat", str(hotshot.iloc[0]["fresh_status"]) == "NOT_FRESH_LOCAL_VARIANT_WITH_DURATION_CAVEAT", str(hotshot.iloc[0]["fresh_status"])))
        rows.append(_row("videofeedback_hotshot:scaffold_ready", bool(hotshot.iloc[0]["scaffold_ready"]), str(hotshot.iloc[0]["scaffold_ready"])))
        rows.append(_row("videofeedback_hotshot:global_exists", bool(hotshot.iloc[0]["global_exists"]), str(hotshot.iloc[0]["global_exists"])))
        rows.append(_row("videofeedback_hotshot:raw_patch_exists", bool(hotshot.iloc[0]["raw_patch_exists"]), str(hotshot.iloc[0]["raw_patch_exists"])))
        rows.append(_row("videofeedback_hotshot:persistence_exists", bool(hotshot.iloc[0]["persistence_exists"]), str(hotshot.iloc[0]["persistence_exists"])))
        rows.append(_row("videofeedback_hotshot:eleven_sources", int(hotshot.iloc[0]["n_fake_sources"]) == 11, str(hotshot.iloc[0]["n_fake_sources"])))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-csv", type=Path, required=True)
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
