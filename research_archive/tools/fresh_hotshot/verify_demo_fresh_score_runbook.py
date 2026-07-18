#!/usr/bin/env python3
"""Verify the demo_dataset fresh score-triplet runbook."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXPECTED_STEPS = [
    "write_enriched_index",
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
    rows.append(_row("step_count_6", len(df) == 6, f"rows={len(df)}"))
    rows.append(_row("expected_steps", list(df["step"].astype(str)) == EXPECTED_STEPS, str(list(df["step"].astype(str)))))
    rows.append(_row("write_enriched_index_ready", str(df.iloc[0]["status"]) == "READY", str(df.iloc[0]["status"])))
    rows.append(_row("metadata_note_mentions_sec_idxs", "sec_idxs" in str(df.iloc[0]["note"]), str(df.iloc[0]["note"])))
    joined_commands = "\n".join(df["command"].astype(str))
    markers = [
        "tools/write_demo_fresh_index.py",
        "src/eval.py",
        "tools/prefill_patch_cache.py",
        "--output-summary-csv",
        "src/eval_patch_fast.py",
        "tools/patch_calibrated_persistence_scores.py",
        "--calibration-patch-emb-cache",
        "tools/build_fresh_validation_scaffold.py",
        "demo_dataset_global_scores.csv",
        "demo_dataset_raw_patch_scores.csv",
        "demo_dataset_persistence_scores.csv",
        "max_frame_mass_thr0p2_real_pct_real",
        "same_grid_second_order",
    ]
    for marker in markers:
        rows.append(_row(f"command_marker:{marker}", marker in joined_commands, marker))
    rows.append(_row("write_step_ready", str(df.iloc[0]["status"]) == "READY", str(df.iloc[0]["status"])))
    score_steps = df[df["step"].astype(str) != "write_enriched_index"]
    rows.append(
        _row(
            "score_steps_blocked_by_duration_window_gap",
            set(score_steps["status"].astype(str)) == {"BLOCKED_DURATION_WINDOW_GAP"},
            str(sorted(score_steps["status"].astype(str).unique())),
        )
    )
    rows.append(_row("duration_gap_note_present", "lack 2_sec_idxs" in "\n".join(score_steps["note"].astype(str)), "\n".join(score_steps["note"].astype(str))))
    md_text = args.runbook_md.read_text(encoding="utf-8")
    rows.append(_row("markdown_nonempty", len(md_text) > 100, f"{len(md_text)} chars"))
    rows.append(_row("demo_index_exists", args.demo_index_csv.exists(), str(args.demo_index_csv)))
    if args.demo_index_csv.exists():
        index = pd.read_csv(args.demo_index_csv)
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
        missing_cols = sorted(required_cols - set(index.columns))
        rows.append(_row("demo_index_enriched_columns", not missing_cols, str(missing_cols)))
        rows.append(_row("demo_index_rows_36", len(index) == 36, f"rows={len(index)}"))
        if not missing_cols:
            missing_2s = index[index["2_sec_idxs"].isna()]
            rows.append(_row("demo_index_duration2_gap_is_two_rows", len(missing_2s) == 2, str(len(missing_2s))))
            rows.append(
                _row(
                    "demo_index_duration2_gap_expected_sources",
                    set(missing_2s["source_model"].astype(str)) == {"HotShot", "MoonValley"},
                    str(sorted(missing_2s["source_model"].astype(str).tolist())),
                )
            )
            rows.append(_row("demo_index_has_1_sec_windows", index["1_sec_idxs"].notna().all(), str(index["1_sec_idxs"].isna().sum())))
            rows.append(_row("demo_index_has_real_and_annotated", set(index["subset"].astype(str)) == {"real", "annotated"}, str(sorted(index["subset"].astype(str).unique()))))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runbook-csv", type=Path, required=True)
    parser.add_argument("--runbook-md", type=Path, required=True)
    parser.add_argument("--demo-index-csv", type=Path, default=Path("cache/indexes/demo_dataset.csv"))
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
