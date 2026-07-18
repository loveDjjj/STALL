#!/usr/bin/env python3
"""Verify the duration=2 filtered demo index artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _row(check: str, passed: bool, detail: str) -> dict[str, object]:
    return {"check": check, "passed": bool(passed), "detail": detail}


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.append(_row("filtered_index_exists", args.filtered_index_csv.exists(), str(args.filtered_index_csv)))
    rows.append(_row("excluded_rows_exists", args.excluded_rows_csv.exists(), str(args.excluded_rows_csv)))
    rows.append(_row("summary_exists", args.summary_csv.exists(), str(args.summary_csv)))
    if not args.filtered_index_csv.exists() or not args.excluded_rows_csv.exists() or not args.summary_csv.exists():
        return pd.DataFrame(rows)

    filtered = pd.read_csv(args.filtered_index_csv)
    excluded = pd.read_csv(args.excluded_rows_csv)
    summary = pd.read_csv(args.summary_csv)
    rows.append(_row("summary_single_row", len(summary) == 1, f"rows={len(summary)}"))
    if len(summary) == 1:
        item = summary.iloc[0]
        rows.extend(
            [
                _row("candidate_name", str(item["candidate"]) == "demo_dataset_duration2_filtered", str(item["candidate"])),
                _row("status_ready", str(item["status"]) == "READY_FOR_DURATION2_SCORE_GENERATION", str(item["status"])),
                _row("input_rows_36", int(item["input_rows"]) == 36, str(item["input_rows"])),
                _row("kept_rows_34", int(item["kept_rows"]) == 34, str(item["kept_rows"])),
                _row("excluded_rows_2", int(item["excluded_rows"]) == 2, str(item["excluded_rows"])),
                _row("n_real_9", int(item["n_real"]) == 9, str(item["n_real"])),
                _row("n_fake_25", int(item["n_fake"]) == 25, str(item["n_fake"])),
                _row("n_fake_sources_7", int(item["n_fake_sources"]) == 7, str(item["n_fake_sources"])),
                _row("all_kept_have_2_sec_idxs", bool(item["all_kept_have_2_sec_idxs"]), str(item["all_kept_have_2_sec_idxs"])),
                _row("promotion_label_filtered", "not full demo_dataset" in str(item["promotion_label"]), str(item["promotion_label"])),
            ]
        )

    rows.append(_row("filtered_rows_34", len(filtered) == 34, f"rows={len(filtered)}"))
    rows.append(_row("excluded_rows_2_table", len(excluded) == 2, f"rows={len(excluded)}"))
    required_cols = {
        "video_path",
        "subset",
        "source_model",
        "fps",
        "duration_seconds",
        "num_frames",
        "downsample_idxs",
        "1_sec_idxs",
        "2_sec_idxs",
        "3_sec_idxs",
        "4_sec_idxs",
    }
    missing_cols = sorted(required_cols - set(filtered.columns))
    rows.append(_row("filtered_columns", not missing_cols, str(missing_cols)))
    if not missing_cols:
        rows.append(_row("filtered_all_have_2_sec_idxs", filtered["2_sec_idxs"].notna().all(), str(filtered["2_sec_idxs"].isna().sum())))
        rows.append(_row("filtered_has_real_and_annotated", set(filtered["subset"].astype(str)) == {"real", "annotated"}, str(sorted(filtered["subset"].astype(str).unique()))))
    if len(excluded) == 2:
        rows.append(_row("excluded_sources_expected", set(excluded["source_model"].astype(str)) == {"HotShot", "MoonValley"}, str(sorted(excluded["source_model"].astype(str)))))
        rows.append(_row("excluded_duration_one_second", set(excluded["duration_seconds"].astype(float)) == {1.0}, str(sorted(excluded["duration_seconds"].astype(float)))))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtered-index-csv", type=Path, required=True)
    parser.add_argument("--excluded-rows-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
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
