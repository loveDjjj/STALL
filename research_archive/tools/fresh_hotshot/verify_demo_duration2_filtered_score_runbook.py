#!/usr/bin/env python3
"""Verify the duration=2 filtered demo score-triplet runbook."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXPECTED_STEPS = [
    "global_scores",
    "patch_cache_prefill",
    "raw_patch_scores",
    "persistence_scores",
    "fresh_scaffold",
]


def _row(check: str, passed: bool, detail: str) -> dict[str, object]:
    return {"check": check, "passed": bool(passed), "detail": detail}


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.append(_row("runbook_csv_exists", args.runbook_csv.exists(), str(args.runbook_csv)))
    rows.append(_row("runbook_md_exists", args.runbook_md.exists(), str(args.runbook_md)))
    if not args.runbook_csv.exists() or not args.runbook_md.exists():
        return pd.DataFrame(rows)

    df = pd.read_csv(args.runbook_csv)
    rows.append(_row("step_count_5", len(df) == 5, f"rows={len(df)}"))
    rows.append(_row("expected_steps", list(df["step"].astype(str)) == EXPECTED_STEPS, str(list(df["step"].astype(str)))))
    rows.append(_row("all_steps_ready", set(df["status"].astype(str)) == {"READY"}, str(sorted(df["status"].astype(str).unique()))))
    joined_commands = "\n".join(df["command"].astype(str))
    joined_notes = "\n".join(df["note"].astype(str))
    markers = [
        "cache/indexes/demo_dataset_duration2_filtered.csv",
        "demo_dataset_duration2_filtered_global_scores.csv",
        "demo_dataset_duration2_filtered_raw_patch_scores.csv",
        "demo_dataset_duration2_filtered_persistence_scores.csv",
        "demo_dataset_duration2_filtered_fresh_validation_scaffold.json",
        "--dataset demo_dataset_duration2_filtered",
        "--duration 2",
        "--calibration-duration 2",
        "--calibration-patch-emb-cache cache/patch_embeddings/videofeedback",
        "--output-summary-csv",
        "demo_dataset_duration2_filtered_patch_cache_prefill_summary.csv",
        "same_grid_second_order",
        "max_frame_mass_thr0p2_real_pct_real",
    ]
    for marker in markers:
        rows.append(_row(f"command_marker:{marker}", marker in joined_commands, marker))
    rows.append(_row("notes_disclose_filtered_subset", "34/36 demo rows" in joined_notes, joined_notes))
    rows.append(_row("notes_disclose_excluded_sources", "HotShot and MoonValley" in joined_notes, joined_notes))
    md_text = args.runbook_md.read_text(encoding="utf-8")
    rows.append(_row("markdown_nonempty", len(md_text) > 100, f"{len(md_text)} chars"))
    rows.append(_row("markdown_dataset_label", "demo_dataset_duration2_filtered" in md_text, md_text[:200]))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runbook-csv", type=Path, required=True)
    parser.add_argument("--runbook-md", type=Path, required=True)
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
