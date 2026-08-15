#!/usr/bin/env python3
"""Build independent real-video reserve and deterministic calibration splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.release_io import resolve_video, video_id, write_json
from alpha_stalled.sampling import uniform_windows
from alpha_stalled.u0_calibration_experiments import (
    CALIBRATION_RESERVE_SPECS as DATASETS,
    SEEDS,
    SIZES,
)


def stable_rank(seed: int, video_id_value: str) -> str:
    return hashlib.sha256(f"{seed}\0{video_id_value}".encode()).hexdigest()


def cache_path(cache_root: Path, row: pd.Series) -> Path:
    return (
        cache_root
        / "real"
        / str(row["source_model"])
        / f"{Path(str(row['filename'])).stem}_2s.pt"
    )


def proportional_counts(frame: pd.DataFrame, size: int) -> dict[str, int]:
    counts = frame["source_model"].value_counts().sort_index()
    exact = counts * (float(size) / float(counts.sum()))
    allocated = np.floor(exact).astype(int)
    remainder = size - int(allocated.sum())
    order = sorted(counts.index, key=lambda key: (-(exact[key] - allocated[key]), key))
    for key in order[:remainder]:
        allocated[key] += 1
    return allocated.astype(int).to_dict()


def select_split(frame: pd.DataFrame, seed: int, size: int) -> pd.DataFrame:
    allocation = proportional_counts(frame, size)
    pieces = []
    for source_model, count in allocation.items():
        group = frame[frame["source_model"] == source_model].copy()
        group["_rank"] = group["video_id"].map(lambda value: stable_rank(seed, value))
        pieces.append(group.sort_values("_rank").head(count))
    selected = pd.concat(pieces, ignore_index=True).drop(columns="_rank")
    if len(selected) != size or selected["video_id"].duplicated().any():
        raise ValueError(f"invalid split seed={seed} size={size}")
    return selected.sort_values("video_id").reset_index(drop=True)


def load_reserve(dataset: str, excluded: set[str]) -> pd.DataFrame:
    index_path, cache_root = DATASETS[dataset]
    frame = pd.read_csv(index_path, float_precision="round_trip")
    frame = frame[frame["subset"].eq("real")].copy()
    frame["dataset"] = dataset
    frame["filename"] = frame["video_path"].map(lambda value: Path(str(value)).name)
    frame["video_id"] = frame.apply(video_id, axis=1)
    frame["k1_windows"] = frame["2_sec_idxs"].map(
        lambda value: [] if pd.isna(value) else [json.loads(str(value))]
    )
    frame["k3_windows"] = frame["downsample_idxs"].map(
        lambda value: uniform_windows([int(index) for index in json.loads(str(value))], 3)
    )
    frame["cache_path"] = frame.apply(lambda row: str(cache_path(cache_root, row)), axis=1)
    frame["source_path"] = frame["video_path"].map(
        lambda value: str(resolve_video(str(value)))
    )
    eligible = frame[
        ~frame["video_id"].isin(excluded)
        & frame["k1_windows"].map(lambda windows: len(windows) == 1 and len(windows[0]) == 16)
        & frame["k3_windows"].map(
            lambda windows: bool(windows)
            and all(len(window) == 16 and len(set(window)) == 16 for window in windows)
        )
        & frame["cache_path"].map(lambda value: Path(value).is_file())
        & frame["source_path"].map(lambda value: Path(value).is_file())
    ].copy()
    if len(eligible) < max(SIZES):
        raise ValueError(f"{dataset}: only {len(eligible)} eligible reserve videos")
    return eligible.sort_values("video_id").reset_index(drop=True)


def run(args: argparse.Namespace) -> None:
    calibration = json.loads(args.calibration_manifest.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation_manifest.read_text(encoding="utf-8"))
    locked_ids = {
        video["video_id"] for payload in (calibration, evaluation) for video in payload["videos"]
    }
    reserve_frames = []
    membership = []
    for dataset in DATASETS:
        reserve = load_reserve(dataset, locked_ids)
        reserve_frames.append(reserve)
        for seed in SEEDS:
            largest = select_split(reserve, seed, max(SIZES))
            for size in SIZES:
                selected = select_split(reserve, seed, size)
                # Each smaller split must be nested in the corresponding 200-video split.
                if not set(selected["video_id"]).issubset(set(largest["video_id"])):
                    raise ValueError(f"non-nested split for {dataset} seed={seed} size={size}")
                for video_id_value in selected["video_id"]:
                    membership.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "calibration_size": size,
                            "video_id": video_id_value,
                        }
                    )

    reserve = pd.concat(reserve_frames, ignore_index=True)
    if set(reserve["video_id"]) & locked_ids:
        raise ValueError("reserve overlaps locked calibration/evaluation videos")
    videos = []
    for row in reserve.itertuples(index=False):
        videos.append(
            {
                "video_id": row.video_id,
                "dataset": row.dataset,
                "subset": row.subset,
                "source_model": row.source_model,
                "filename": row.filename,
                "video_path": row.video_path,
                "source_path": row.source_path,
                "cache_path": str(Path(row.cache_path).relative_to(ROOT)),
                "duration_seconds": float(row.duration_seconds),
                "k1_window": row.k1_windows[0],
                "k3_windows": row.k3_windows,
                "effective_k": len(row.k3_windows),
            }
        )
    payload = {
        "schema_version": "u0_calibration_reserve_v1",
        "selection": "source-stratified stable SHA-256 rank; sizes nested within seed",
        "seeds": list(SEEDS),
        "sizes": list(SIZES),
        "video_count": len(videos),
        "dataset_counts": reserve.groupby("dataset").size().astype(int).to_dict(),
        "locked_overlap_count": 0,
        "videos": videos,
    }
    write_json(args.reserve_manifest, payload)
    membership_frame = pd.DataFrame(membership).sort_values(
        ["dataset", "seed", "calibration_size", "video_id"]
    )
    args.membership.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.membership.with_suffix(".tmp.csv")
    membership_frame.to_csv(temporary, index=False)
    temporary.replace(args.membership)
    print(
        f"reserve={len(reserve)} counts={payload['dataset_counts']} "
        f"memberships={len(membership_frame)} overlap=0"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--membership",
        type=Path,
        default=ROOT / "release/u0/calibration_split_membership.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
