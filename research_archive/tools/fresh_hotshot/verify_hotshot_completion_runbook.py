#!/usr/bin/env python3
"""Verify Hotshot-XL completion manifest and runbook."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _row(check: str, passed: bool, detail: str) -> dict[str, object]:
    return {"check": check, "passed": bool(passed), "detail": detail}


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, path in {
        "target": args.target_csv,
        "cache_status": args.cache_status_csv,
        "summary": args.summary_csv,
        "runbook": args.runbook_md,
    }.items():
        rows.append(_row(f"exists:{label}", path.exists(), str(path)))
    if not all(path.exists() for path in [args.target_csv, args.cache_status_csv, args.summary_csv, args.runbook_md]):
        return pd.DataFrame(rows)

    target = pd.read_csv(args.target_csv)
    cache = pd.read_csv(args.cache_status_csv)
    summary = pd.read_csv(args.summary_csv)
    text = args.runbook_md.read_text(encoding="utf-8")
    rows.append(_row("target_rows_3251", len(target) == 3251, str(len(target))))
    rows.append(_row("target_only_hotshot_xl", set(target["source_model"].astype(str)) == {"Hotshot-XL"}, str(sorted(target["source_model"].astype(str).unique()))))
    rows.append(_row("target_subset_annotated", set(target["subset"].astype(str)) == {"annotated"}, str(sorted(target["subset"].astype(str).unique()))))
    rows.append(_row("target_has_video_path", "video_path" in target.columns, str(list(target.columns))))
    rows.append(_row("target_has_2s_window", "2_sec_idxs" in target.columns, str(list(target.columns))))
    target_paths = target["video_path"].astype(str) if "video_path" in target.columns else pd.Series(dtype=str)
    n_stall_prefixed = int(target_paths.str.startswith("STALL/").sum())
    rows.append(_row("target_video_paths_not_stall_prefixed", n_stall_prefixed == 0, str(n_stall_prefixed)))
    n_existing_paths = int(target_paths.map(lambda value: Path(value).exists()).sum())
    rows.append(_row("target_video_paths_exist_from_cwd", n_existing_paths == len(target), f"{n_existing_paths}/{len(target)}"))
    rows.append(_row("cache_rows_match_target", len(cache) == len(target), f"{len(cache)}/{len(target)}"))
    rows.append(_row("summary_single_row", len(summary) == 1, str(len(summary))))
    rows.append(_row("summary_target_rows_3251", int(summary.iloc[0]["target_rows"]) == 3251, str(summary.iloc[0]["target_rows"])))
    rows.append(_row("summary_duration_1", int(summary.iloc[0]["duration"]) == 1, str(summary.iloc[0]["duration"])))
    rows.append(_row("summary_video_path_prefix_stripped_stall", str(summary.iloc[0]["video_path_prefix_stripped"]) == "STALL", str(summary.iloc[0].get("video_path_prefix_stripped", ""))))
    rows.append(_row("summary_missing_cache_nonnegative", int(summary.iloc[0]["patch_cache_rows_missing"]) >= 0, str(summary.iloc[0]["patch_cache_rows_missing"])))
    markers = [
        "tools/prefill_patch_cache.py",
        "src/eval_patch_fast.py",
        "tools/patch_calibrated_persistence_scores.py",
        "same_grid_second_order",
        "max_frame_mass_thr0p2_real_pct_real",
        "run_fresh_validation_scaffold.py",
        "local variant",
        "duration=1",
        "local-variant caveat",
        "target video paths normalized",
    ]
    for marker in markers:
        rows.append(_row(f"runbook_marker:{marker}", marker in text, marker))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-csv", type=Path, required=True)
    parser.add_argument("--cache-status-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
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
