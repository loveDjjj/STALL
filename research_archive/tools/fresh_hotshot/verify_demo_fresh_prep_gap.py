#!/usr/bin/env python3
"""Verify demo_dataset fresh-preparation gap artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _row(check: str, passed: bool, detail: str) -> dict[str, object]:
    return {"check": check, "passed": bool(passed), "detail": detail}


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.append(_row("summary_exists", args.summary_csv.exists(), str(args.summary_csv)))
    rows.append(_row("index_preview_exists", args.index_preview_csv.exists(), str(args.index_preview_csv)))
    if not args.summary_csv.exists() or not args.index_preview_csv.exists():
        return pd.DataFrame(rows)

    summary = pd.read_csv(args.summary_csv)
    preview = pd.read_csv(args.index_preview_csv)
    rows.append(_row("summary_single_row", len(summary) == 1, f"rows={len(summary)}"))
    if len(summary) == 1:
        item = summary.iloc[0]
        rows.extend(
            [
                _row("candidate_demo_dataset", str(item["candidate"]) == "demo_dataset", str(item["candidate"])),
                _row("status_ready_for_score_generation", str(item["status"]) == "READY_FOR_SCORE_GENERATION", str(item["status"])),
                _row("manifest_rows_36", int(item["manifest_rows"]) == 36, str(item["manifest_rows"])),
                _row("n_real_9", int(item["n_real"]) == 9, str(item["n_real"])),
                _row("n_fake_27", int(item["n_fake"]) == 27, str(item["n_fake"])),
                _row("n_fake_sources_9", int(item["n_fake_sources"]) == 9, str(item["n_fake_sources"])),
                _row("all_video_paths_exist", bool(item["all_video_paths_exist"]), str(item["all_video_paths_exist"])),
                _row("can_build_index", bool(item["can_build_index"]), str(item["can_build_index"])),
                _row("global_scores_missing", not bool(item["global_scores_exist"]), str(item["global_scores_exist"])),
                _row("raw_patch_scores_missing", not bool(item["raw_patch_scores_exist"]), str(item["raw_patch_scores_exist"])),
                _row("persistence_scores_missing", not bool(item["persistence_scores_exist"]), str(item["persistence_scores_exist"])),
                _row("score_triplet_missing", not bool(item["score_triplet_exists"]), str(item["score_triplet_exists"])),
                _row("required_next_generate_scores", "generate global scores" in str(item["required_next"]), str(item["required_next"])),
            ]
        )

    required_preview_cols = {
        "subset",
        "source_model",
        "filename",
        "video_path",
        "expected_demo_video_path",
        "path_resolution",
        "original_filename",
        "video_exists",
    }
    missing_preview_cols = sorted(required_preview_cols - set(preview.columns))
    rows.append(_row("index_preview_columns", not missing_preview_cols, str(missing_preview_cols)))
    rows.append(_row("index_preview_rows_36", len(preview) == 36, f"rows={len(preview)}"))
    if not missing_preview_cols:
        rows.append(_row("index_preview_all_videos_exist", bool(preview["video_exists"].astype(bool).all()), str(preview["video_exists"].value_counts().to_dict())))
        fallback = preview[preview["path_resolution"].astype(str) == "original_filename_unique_fallback"]
        rows.append(
            _row(
                "fallback_video_is_lavie_base_4001831",
                len(fallback) == 1
                and str(fallback.iloc[0]["source_model"]) == "LaVie-base"
                and str(fallback.iloc[0]["filename"]) == "4001831.mp4"
                and str(fallback.iloc[0]["expected_demo_video_path"]).endswith("LaVie-base/36785.mp4"),
                fallback.to_string(index=False),
            )
        )
        rows.append(_row("index_preview_has_real_and_annotated", set(preview["subset"].astype(str)) == {"real", "annotated"}, str(sorted(preview["subset"].astype(str).unique()))))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--index-preview-csv", type=Path, required=True)
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
