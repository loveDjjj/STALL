#!/usr/bin/env python3
"""Finalize a locked U0 raw shard from completed CSV checkpoints without Torch."""

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

from alpha_stalled.artifacts import read_csv_files
from alpha_stalled.release_io import video_id_shard


KEY_COLUMNS = [
    "video_id",
    "dataset",
    "protocol_split",
    "subset",
    "source_model",
    "filename",
]
WINDOW_KEYS = [*KEY_COLUMNS, "window_id"]


def release_videos(release_dir: Path, dataset: str, num_shards: int, shard: int) -> list[dict]:
    records = []
    for name in ("calibration_manifest.json", "evaluation_manifest.json"):
        records.extend(json.loads((release_dir / name).read_text())["videos"])
    return [
        row
        for row in records
        if row["dataset"] == dataset
        and video_id_shard(row["video_id"], num_shards) == shard
    ]


def run(args: argparse.Namespace) -> None:
    expected_rows = release_videos(
        args.release_dir, args.dataset, args.num_shards, args.shard_index
    )
    expected_ids = {row["video_id"] for row in expected_rows}
    frame_indices = json.loads(
        (args.release_dir / "frame_indices.json").read_text()
    )["videos"]
    checkpoint_dir = (
        args.output_dir
        / "checkpoints"
        / args.dataset
        / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    )
    parts = sorted(checkpoint_dir.glob("part_*.csv"))
    if not parts:
        raise ValueError(f"no checkpoint parts: {checkpoint_dir}")
    merged = read_csv_files(parts)
    if merged.duplicated(WINDOW_KEYS).any():
        raise ValueError("duplicate locked window keys")
    if set(merged["video_id"]) != expected_ids:
        missing = expected_ids - set(merged["video_id"])
        extra = set(merged["video_id"]) - expected_ids
        raise ValueError(f"checkpoint IDs incomplete: missing={len(missing)} extra={len(extra)}")
    expected_windows = {
        video_id: len(frame_indices[video_id]) for video_id in expected_ids
    }
    actual_windows = merged.groupby("video_id").size().astype(int).to_dict()
    mismatched = {
        video_id: (expected, actual_windows.get(video_id, 0))
        for video_id, expected in expected_windows.items()
        if actual_windows.get(video_id, 0) != expected
    }
    if mismatched:
        raise ValueError(f"checkpoint window counts mismatch: {len(mismatched)}")
    expected_frame_rows = {
        (video_id, window_id): json.dumps(window, separators=(",", ":"))
        for video_id in expected_ids
        for window_id, window in enumerate(frame_indices[video_id])
    }
    actual_frame_rows = {
        (row.video_id, int(row.window_id)): row.frame_indices
        for row in merged[["video_id", "window_id", "frame_indices"]].itertuples(index=False)
    }
    if actual_frame_rows != expected_frame_rows:
        raise ValueError("checkpoint frame indices differ from locked manifest")
    output = (
        args.output_dir
        / "raw"
        / f"{args.dataset}_shard{args.shard_index:02d}_of_{args.num_shards:02d}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.csv")
    merged.sort_values(WINDOW_KEYS).to_csv(temporary, index=False)
    temporary.replace(output)
    print(
        f"finalized dataset={args.dataset} shard={args.shard_index}/{args.num_shards} "
        f"videos={len(expected_ids)} windows={len(merged)} -> {output}"
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
