#!/usr/bin/env python3
"""Exactly reconstruct locked U0 final scores from retained raw shard evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_stalled.u0_analysis import (
    aggregate_and_calibrate_videos,
    calibrate_windows,
)
from alpha_stalled.u0_protocol import (
    load_calibration_references,
    load_raw_windows,
    verify_calibration_reference_windows,
)


DEFAULT_RELEASE = ROOT / "release/u0"


def repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify(
    release_dir: Path = DEFAULT_RELEASE,
    *,
    metadata_path: Path | None = None,
) -> dict[str, object]:
    release_dir = repository_path(release_dir)
    metadata_path = repository_path(
        metadata_path or release_dir / "reproduction_metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config_path = repository_path(metadata["config"])
    raw_dir = repository_path(metadata["raw_directory"])
    calibration_raw_dir = repository_path(metadata["calibration_raw_directory"])
    num_shards = int(metadata["num_shards_per_dataset"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    raw = load_raw_windows(raw_dir, num_shards)
    references = load_calibration_references(calibration_raw_dir, num_shards)
    verify_calibration_reference_windows(references, release_dir)
    calibrated_windows, _ = calibrate_windows(raw, references, config)
    reconstructed, _ = aggregate_and_calibrate_videos(calibrated_windows)
    released = pd.read_csv(
        release_dir / "final_video_scores.csv", float_precision="round_trip"
    )
    reconstructed = reconstructed.sort_values("video_id").reset_index(drop=True)
    released = released.sort_values("video_id").reset_index(drop=True)
    if list(reconstructed.columns) != list(released.columns):
        raise AssertionError(
            "reconstructed/released columns differ: "
            f"{list(reconstructed.columns)} != {list(released.columns)}"
        )
    pd.testing.assert_frame_equal(
        reconstructed,
        released,
        check_exact=True,
        check_dtype=True,
        obj="raw-to-final locked U0 reconstruction",
    )
    if len(calibrated_windows) != int(metadata["raw_window_count"]):
        raise AssertionError("reconstructed raw window count differs from metadata")
    if len(reconstructed) != int(metadata["evaluation_video_count"]):
        raise AssertionError("reconstructed evaluation count differs from metadata")
    return {
        "passed": True,
        "raw_window_count": len(calibrated_windows),
        "evaluation_video_count": len(reconstructed),
        "num_shards_per_dataset": num_shards,
        "comparison": "exact_columns_values_dtypes",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = verify(args.release_dir, metadata_path=args.metadata)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(
            "locked U0 pipeline reconstruction passed: "
            f"windows={payload['raw_window_count']} "
            f"videos={payload['evaluation_video_count']} "
            f"comparison={payload['comparison']}"
        )


if __name__ == "__main__":
    main()
