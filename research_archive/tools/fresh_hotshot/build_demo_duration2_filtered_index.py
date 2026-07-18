#!/usr/bin/env python3
"""Build a protocol-compatible duration=2 filtered demo index."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.read_csv(args.input_index_csv)
    required = ["video_path", "subset", "source_model", "duration_seconds", "2_sec_idxs"]
    missing = [col for col in required if col not in index.columns]
    if missing:
        raise ValueError(f"{args.input_index_csv} missing columns: {missing}")

    keep = index[index["2_sec_idxs"].notna()].copy().reset_index(drop=True)
    excluded = index[index["2_sec_idxs"].isna()].copy().reset_index(drop=True)
    fake = keep[keep["subset"].astype(str).str.lower() != "real"]
    summary = pd.DataFrame(
        [
            {
                "candidate": "demo_dataset_duration2_filtered",
                "status": "READY_FOR_DURATION2_SCORE_GENERATION" if len(keep) > 0 and len(excluded) > 0 else "CHECK_FILTER",
                "input_rows": int(len(index)),
                "kept_rows": int(len(keep)),
                "excluded_rows": int(len(excluded)),
                "n_real": int((keep["subset"].astype(str).str.lower() == "real").sum()),
                "n_fake": int((keep["subset"].astype(str).str.lower() != "real").sum()),
                "n_fake_sources": int(fake["source_model"].nunique()),
                "excluded_sources": ";".join(sorted(excluded["source_model"].astype(str).unique())),
                "all_kept_have_2_sec_idxs": bool(keep["2_sec_idxs"].notna().all()),
                "required_next": "generate duration=2 global, raw patch, and persistence scores on filtered index",
                "promotion_label": "duration=2 filtered demo subset; not full demo_dataset",
            }
        ]
    )
    return keep, excluded, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-index-csv", type=Path, default=Path("cache/indexes/demo_dataset.csv"))
    parser.add_argument("--output-index-csv", type=Path, required=True)
    parser.add_argument("--output-excluded-csv", type=Path, required=True)
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    args = parser.parse_args()

    keep, excluded, summary = run(args)
    args.output_index_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_excluded_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    keep.to_csv(args.output_index_csv, index=False)
    excluded.to_csv(args.output_excluded_csv, index=False)
    summary.to_csv(args.output_summary_csv, index=False)
    print(summary.to_string(index=False))
    print(f"Saved filtered index -> {args.output_index_csv}")
    print(f"Saved excluded rows -> {args.output_excluded_csv}")


if __name__ == "__main__":
    main()
