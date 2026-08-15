#!/usr/bin/env python3
"""Build strict-2s real videos omitted from the frozen calibration/evaluation pools."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from alpha_stalled.release_io import resolve_video, video_id, write_json
from alpha_stalled.sampling import uniform_windows


def run(args: argparse.Namespace) -> None:
    known = set()
    for path in (
        args.calibration_manifest,
        args.evaluation_manifest,
        args.reserve_manifest,
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        known.update(item["video_id"] for item in payload["videos"])

    index = pd.read_csv(args.index, float_precision="round_trip")
    index = index[index["subset"].eq("real")].copy()
    index["dataset"] = "videofeedback"
    index["filename"] = index["video_path"].map(lambda value: Path(str(value)).name)
    index["video_id"] = index.apply(video_id, axis=1)
    videos = []
    for row in index.sort_values(["source_model", "filename"]).itertuples(index=False):
        if row.video_id in known:
            continue
        downsample = [int(value) for value in json.loads(str(row.downsample_idxs))]
        windows = uniform_windows(downsample, 3)
        if not windows or any(
            len(window) != 16 or len(set(window)) != 16 for window in windows
        ):
            continue
        source = resolve_video(str(row.video_path))
        videos.append(
            {
                "video_id": row.video_id,
                "dataset": "videofeedback",
                "protocol_split": "independent_remaining_real",
                "subset": "real",
                "source_model": str(row.source_model),
                "filename": str(row.filename),
                "video_path": str(row.video_path),
                "source_path": str(source),
                "duration_seconds": float(row.duration_seconds),
                "effective_k": len(windows),
                "k3_windows": windows,
            }
        )
    if len(videos) != 3_080:
        raise ValueError(f"expected 3,080 supplemental VideoFeedback real videos, got {len(videos)}")
    ids = [item["video_id"] for item in videos]
    if len(set(ids)) != len(ids) or set(ids) & known:
        raise ValueError("supplemental real IDs are duplicated or overlap frozen pools")
    counts = pd.Series(item["effective_k"] for item in videos).value_counts().to_dict()
    if counts != {3: 2_043, 1: 1_037}:
        raise ValueError(f"unexpected supplemental effective-K distribution: {counts}")
    payload = {
        "schema_version": "u0_independent_remaining_real_v1",
        "selection": "all VideoFeedback real videos outside calibration/evaluation/reserve with valid strict 2s K3 sampling",
        "video_count": len(videos),
        "effective_k_counts": {str(key): int(value) for key, value in sorted(counts.items())},
        "overlap_with_frozen_pools": 0,
        "videos": videos,
    }
    write_json(args.output, payload)
    print(f"wrote {len(videos)} videos -> {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index", type=Path, default=ROOT / "cache/indexes/videofeedback.csv"
    )
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=ROOT / "release/u0/calibration_manifest.json",
    )
    parser.add_argument(
        "--evaluation-manifest",
        type=Path,
        default=ROOT / "release/u0/evaluation_manifest.json",
    )
    parser.add_argument(
        "--reserve-manifest",
        type=Path,
        default=ROOT / "release/u0/calibration_reserve_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "release/u0/independent_remaining_real_manifest.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
