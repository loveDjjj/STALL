#!/usr/bin/env python3
"""Build the leakage-free full-data duration-aware evaluation protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Re-exported for compatibility with historical imports. New consumers import
# these protocol definitions directly from alpha_stalled.duration_aware_protocol.
from alpha_stalled.duration_aware_protocol import (  # noqa: E402
    CALIBRATION_SIZES,
    DATASETS,
    DATASET_SPECS,
    EXISTING_SIZE_INDEX,
    K3_EXCLUSIONS,
    N200_INDEX,
    SHORT_DATASETS,
    add_identity,
    build_dataset,
    calibration_size_splits,
    duration_two_windows,
    load_k3_exclusions,
    parse_window,
    proportional_counts,
    record,
    stable_size_rank,
    task_id,
    validate,
)


def run(args: argparse.Namespace) -> None:
    frames = []
    summaries = {}
    for dataset, spec in DATASETS.items():
        frame, summary = build_dataset(dataset, spec)
        frames.append(frame)
        summaries[dataset] = summary
    tasks = pd.concat(frames, ignore_index=True)
    validate(tasks)
    tasks = tasks.sort_values(
        ["dataset", "protocol_split", "subset", "source_model", "filename", "sampling"]
    ).reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.output_dir / "protocol_tasks.tmp.csv"
    tasks.to_csv(temporary, index=False)
    temporary.replace(args.output_dir / "protocol_tasks.csv")

    duration_counts = (
        tasks[tasks["protocol_split"].eq("evaluation") & tasks["subset"].eq("annotated")]
        .groupby(["dataset", "source_model", "protocol_duration_sec"], sort=True)
        .size()
        .rename("video_count")
        .reset_index()
    )
    duration_counts.to_csv(args.output_dir / "fake_duration_counts.csv", index=False)
    memberships = []
    size_memberships = []
    size_index_dir = args.output_dir / "calibration_indexes"
    size_index_dir.mkdir(parents=True, exist_ok=True)
    for dataset, spec in DATASETS.items():
        maximum = add_identity(
            pd.read_csv(spec["calibration"], float_precision="round_trip"), dataset
        )
        n200 = add_identity(
            pd.read_csv(N200_INDEX[dataset], float_precision="round_trip"), dataset
        )
        n200_ids = set(n200["video_id"])
        if len(n200_ids) != 200 or not n200_ids.issubset(set(maximum["video_id"])):
            raise ValueError(f"{dataset}: N=200 is not nested in N=max")
        memberships.extend(
            {
                "dataset": dataset,
                "video_id": value,
                "in_n200": value in n200_ids,
                "in_nmax": True,
            }
            for value in maximum["video_id"]
        )
        splits = calibration_size_splits(dataset, maximum)
        for size, selected_ids in splits.items():
            selected = maximum[maximum["video_id"].isin(selected_ids)].drop(
                columns=["dataset", "filename", "video_id"]
            )
            index_path = size_index_dir / f"{dataset}_n{size}.csv"
            selected.sort_values(["source_model", "video_path"]).to_csv(
                index_path, index=False
            )
            size_memberships.extend(
                {
                    "dataset": dataset,
                    "calibration_size": size,
                    "video_id": value,
                }
                for value in sorted(selected_ids)
            )
    pd.DataFrame(memberships).sort_values(["dataset", "video_id"]).to_csv(
        args.output_dir / "calibration_membership.csv", index=False
    )
    pd.DataFrame(size_memberships).sort_values(
        ["dataset", "calibration_size", "video_id"]
    ).to_csv(args.output_dir / "calibration_size_membership.csv", index=False)
    payload = {
        "schema_version": "alpha_stalled_duration_aware_full_v1",
        "selection": {
            "calibration": "predeclared nested real-only maximum; disjoint from evaluation",
            "two_second": "deterministic uniform K=3, 16 distinct frames at 8 fps",
            "one_second": "current deterministic K=1, 8 distinct frames at 8 fps",
            "fallback": "use 1s only when no valid 2s window exists",
            "metric_real_duration_matching": "exact selected-fake duration counts per generator",
            "calibration_size_selection": "nested real-only sequential holdout curve",
        },
        "calibration_sizes": {
            key: list(value) for key, value in CALIBRATION_SIZES.items()
        },
        "dataset_summary": summaries,
        "task_count": len(tasks),
        "physical_video_count": tasks["video_id"].nunique(),
        "calibration_evaluation_overlap": 0,
    }
    path = args.output_dir / "protocol_summary.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/duration_aware_23source",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
