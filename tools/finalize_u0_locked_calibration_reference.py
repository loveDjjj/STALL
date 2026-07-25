#!/usr/bin/env python3
"""Finalize locked U0 K1 calibration references without importing Torch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def stable_shard(video_id: str, num_shards: int) -> int:
    return int(video_id[:16], 16) % num_shards


def run(args: argparse.Namespace) -> None:
    manifest = json.loads(
        (args.release_dir / "calibration_manifest.json").read_text()
    )["videos"]
    expected_ids = {
        row["video_id"]
        for row in manifest
        if row["dataset"] == args.dataset
        and stable_shard(row["video_id"], args.num_shards) == args.shard_index
    }
    references = json.loads(
        (args.release_dir / "frame_indices.json").read_text()
    )["calibration_reference_windows"]
    checkpoint_dir = (
        args.output_dir
        / "calibration_reference_checkpoints"
        / args.dataset
        / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    )
    parts = sorted(checkpoint_dir.glob("part_*.csv"))
    if not parts:
        raise ValueError(f"no calibration checkpoint parts: {checkpoint_dir}")
    merged = pd.concat(
        [pd.read_csv(path, float_precision="round_trip") for path in parts],
        ignore_index=True,
    )
    if merged["video_id"].duplicated().any():
        raise ValueError("duplicate calibration reference video IDs")
    if set(merged["effective_k"].astype(int)) != {1}:
        raise ValueError("calibration references must each have effective_k=1")
    if set(merged["video_id"]) != expected_ids:
        missing = expected_ids - set(merged["video_id"])
        extra = set(merged["video_id"]) - expected_ids
        raise ValueError(
            f"calibration checkpoints incomplete: missing={len(missing)} extra={len(extra)}"
        )
    expected_frames = {
        video_id: json.dumps(references[video_id], separators=(",", ":"))
        for video_id in expected_ids
    }
    actual_frames = dict(zip(merged["video_id"], merged["frame_indices"]))
    if actual_frames != expected_frames:
        raise ValueError("calibration reference frames differ from locked manifest")
    finite_columns = ["global_spatial_raw", "patch_spatial_raw", "patch_d2_raw"]
    if not np.isfinite(merged[finite_columns].to_numpy()).all():
        raise ValueError("non-finite calibration reference raw score")
    global_t1 = merged["global_t1_raw"].to_numpy()
    if np.isnan(global_t1).any() or np.isneginf(global_t1).any():
        raise ValueError("invalid calibration reference Global T1 score")
    output = (
        args.output_dir
        / "calibration_raw"
        / f"{args.dataset}_shard{args.shard_index:02d}_of_{args.num_shards:02d}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.csv")
    merged.sort_values("video_id").to_csv(temporary, index=False)
    temporary.replace(output)
    print(
        f"finalized calibration-reference dataset={args.dataset} "
        f"shard={args.shard_index}/{args.num_shards} rows={len(merged)} -> {output}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("comgenvid", "videofeedback", "genvideo"), required=True
    )
    parser.add_argument("--release-dir", type=Path, default=ROOT / "release/u0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--shard-index", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
