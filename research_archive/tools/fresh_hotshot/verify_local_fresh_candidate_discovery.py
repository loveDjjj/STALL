#!/usr/bin/env python3
"""Verify local fresh-candidate discovery output."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXPECTED_KNOWN = {"comgenvid", "genvideo", "genvideo_1s", "videofeedback", "videofeedback_hotshot"}


def _row(check: str, passed: bool, detail: str) -> dict[str, object]:
    return {"check": check, "passed": bool(passed), "detail": detail}


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    exists = args.discovery_csv.exists()
    rows.append(_row("discovery_exists", exists, str(args.discovery_csv)))
    if not exists:
        return pd.DataFrame(rows)

    df = pd.read_csv(args.discovery_csv)
    required = [
        "kind",
        "candidate",
        "path",
        "status",
        "already_in_inventory",
        "has_global_schema",
        "has_patch_candidate",
        "has_persistence_candidate",
        "missing_for_fresh",
    ]
    missing = [col for col in required if col not in df.columns]
    rows.append(_row("required_columns", not missing, str(missing)))
    if missing:
        return pd.DataFrame(rows)

    known_rows = df[df["candidate"].astype(str).isin(EXPECTED_KNOWN) & (df["kind"].astype(str) == "global_score_csv")]
    rows.append(_row("known_global_candidates_present", set(known_rows["candidate"].astype(str)) == EXPECTED_KNOWN, str(sorted(known_rows["candidate"].astype(str).unique()))))
    rows.append(_row("known_global_candidates_in_inventory", bool(known_rows["already_in_inventory"].astype(bool).all()), known_rows[["candidate", "already_in_inventory"]].to_string(index=False)))
    rows.append(_row("no_new_local_ready_candidate", not (df["status"].astype(str) == "POTENTIAL_LOCAL_READY_NEEDS_PROVENANCE").any(), df[["candidate", "status"]].to_string(index=False)))

    demo = df[df["candidate"].astype(str) == "demo_dataset"]
    rows.append(_row("demo_dataset_discovered", len(demo) == 1, f"rows={len(demo)}"))
    if len(demo) == 1:
        item = demo.iloc[0]
        rows.append(_row("demo_dataset_video_manifest_only", str(item["status"]) == "VIDEO_MANIFEST_ONLY", str(item["status"])))
        rows.append(_row("demo_dataset_missing_scores", "global scores" in str(item["missing_for_fresh"]), str(item["missing_for_fresh"])))

    global_rows = df[df["kind"].astype(str) == "global_score_csv"]
    rows.append(_row("global_score_rows_5", len(global_rows) == 5, f"rows={len(global_rows)}"))
    manifest_rows = df[df["kind"].astype(str) == "dataset_video_manifest"]
    rows.append(_row("manifest_rows_at_least_1", len(manifest_rows) >= 1, f"rows={len(manifest_rows)}"))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-csv", type=Path, required=True)
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
