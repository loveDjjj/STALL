#!/usr/bin/env python3
"""Score arbitrary calibration-size candidates with the current DINO extractor."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.duration_aware_protocol import CALIBRATION_SIZES
from alpha_stalled.duration_aware_scoring import KEY_COLUMNS, WINDOW_KEYS, decode_video
from alpha_stalled.release_io import video_id_shard
from alpha_stalled.local_branch import local_d2_features
from alpha_stalled.whitening import (
    GaussianMeanCandidateScorerFloat64,
    StableGaussianParams,
)
from stall_patch import PatchSTALL


def parameter_path(config: dict, params_dir: Path, dataset: str, duration: int, size: int) -> Path:
    if duration == 2 and size == 200:
        return ROOT / config["local_branch"]["params_by_dataset"][dataset]["path"]
    return params_dir / f"{dataset}_{duration}s_n{size}.npz"


def load_scorers(
    config: dict,
    params_dir: Path,
    dataset: str,
    duration: int,
    sizes: tuple[int, ...],
    device: str,
) -> tuple[GaussianMeanCandidateScorerFloat64, GaussianMeanCandidateScorerFloat64]:
    spatial = []
    temporal = []
    for size in sizes:
        path = parameter_path(config, params_dir, dataset, duration, size)
        if not path.is_file():
            raise FileNotFoundError(path)
        spatial.append(
            StableGaussianParams.from_npz(
                str(path), "mu_patch_spat", "W_patch_spat", "calib_patch_spat_scores"
            )
        )
        temporal.append(
            StableGaussianParams.from_npz(
                str(path), "mu_patch_temp", "W_patch_temp", "calib_patch_temp_scores"
            )
        )
    return (
        GaussianMeanCandidateScorerFloat64(spatial, spatial[0].mean, device=device),
        GaussianMeanCandidateScorerFloat64(temporal, temporal[0].mean, device=device),
    )


@torch.inference_mode()
def score_decoded(
    decoded: list[dict],
    extractor: PatchSTALL,
    scorers: dict[int, tuple[GaussianMeanCandidateScorerFloat64, GaussianMeanCandidateScorerFloat64]],
    sizes: tuple[int, ...],
    frame_batch_size: int,
) -> list[dict]:
    extracted = [
        extractor.frames_to_global_patch_embeddings([item["frames"]], batch_size=frame_batch_size)[0]
        for item in decoded
    ]
    grouped: dict[int, list[tuple[dict, int, dict, list[int], np.ndarray]]] = {}
    for item, embeddings in zip(decoded, extracted):
        if tuple(embeddings["grid_size"]) != (14, 14):
            raise ValueError(f"unexpected patch grid {embeddings['grid_size']}")
        for (metadata, window_id, window), positions in zip(item["tasks"], item["positions"]):
            duration = int(metadata["protocol_duration_sec"])
            grouped.setdefault(duration, []).append(
                (metadata, window_id, item, window, embeddings["patch"][positions])
            )
    output = []
    for duration, windows in grouped.items():
        spatial_scorer, temporal_scorer = scorers[duration]
        patch = torch.from_numpy(np.stack([item[4] for item in windows]).astype(np.float32))
        spatial = spatial_scorer.score(patch)
        temporal = temporal_scorer.score(local_d2_features(patch))
        for index, (metadata, window_id, item, frame_indices, _) in enumerate(windows):
            row = {
                **{column: metadata[column] for column in KEY_COLUMNS},
                "duration_seconds": float(metadata["duration_seconds"]),
                "effective_k": int(metadata["effective_k"]),
                "window_id": window_id,
                "frame_indices": json.dumps(frame_indices, separators=(",", ":")),
            }
            for position, size in enumerate(sizes):
                row[f"patch_spatial__n{size}"] = spatial[index, position]
                row[f"patch_d2__n{size}"] = temporal[index, position]
            output.append(row)
    return output


def completed_videos(directory: Path) -> tuple[set[str], list[Path]]:
    parts = sorted(directory.glob("part_*.csv"))
    completed = set()
    for path in parts:
        completed.update(pd.read_csv(path, usecols=["video_id"])["video_id"].unique())
    return completed, parts


def run(args: argparse.Namespace) -> None:
    allowed = CALIBRATION_SIZES[args.dataset]
    sizes = tuple(args.candidate_sizes or allowed)
    if len(set(sizes)) != len(sizes) or any(size not in allowed for size in sizes):
        raise ValueError(f"invalid sizes {sizes}; allowed={allowed}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    tasks = pd.read_csv(args.tasks, float_precision="round_trip")
    tasks = tasks[
        tasks["dataset"].eq(args.dataset)
        & tasks["protocol_split"].isin(args.split)
        & tasks["video_id"].map(
            lambda value: video_id_shard(str(value), args.num_shards) == args.shard_index
        )
    ].copy()
    label = args.label or "n" + "_n".join(str(size) for size in sizes)
    checkpoint = (
        args.output_dir
        / "curve_checkpoints"
        / label
        / args.dataset
        / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    )
    checkpoint.mkdir(parents=True, exist_ok=True)
    completed, parts = completed_videos(checkpoint)
    video_ids = tasks["video_id"].drop_duplicates().tolist()
    pending_ids = [value for value in video_ids if value not in completed]
    print(
        f"dataset={args.dataset} label={label} shard={args.shard_index}/{args.num_shards} "
        f"videos={len(video_ids)} completed={len(completed)} pending={len(pending_ids)}",
        flush=True,
    )
    durations = sorted(tasks["protocol_duration_sec"].astype(int).unique())
    scorers = {
        duration: load_scorers(
            config, args.params_dir, args.dataset, duration, sizes, args.score_device
        )
        for duration in durations
    }
    extractor = PatchSTALL(args.extract_device, data_dict=None, load_dino=True)
    grouped = {key: frame for key, frame in tasks.groupby("video_id", sort=False)}
    failures = []
    buffer = []
    part_index = len(parts)
    started = time.perf_counter()
    for start in range(0, len(pending_ids), args.video_batch_size):
        current = pending_ids[start : start + args.video_batch_size]
        decoded = []
        with ThreadPoolExecutor(max_workers=min(args.decode_workers, len(current))) as pool:
            futures = [
                pool.submit(
                    decode_video, grouped[video_id_value], args.seek_gap, args.decode_attempts
                )
                for video_id_value in current
            ]
            for video_id_value, future in zip(current, futures):
                try:
                    decoded.append(future.result())
                except Exception as error:
                    failures.append({"video_id": video_id_value, "error": repr(error)})
        if decoded:
            buffer.extend(
                score_decoded(decoded, extractor, scorers, sizes, args.frame_batch_size)
            )
        processed = min(start + args.video_batch_size, len(pending_ids))
        if len(buffer) >= args.checkpoint_rows or processed == len(pending_ids):
            if buffer:
                path = checkpoint / f"part_{part_index:06d}.csv"
                temporary = path.with_suffix(".tmp.csv")
                pd.DataFrame(buffer).to_csv(temporary, index=False)
                temporary.replace(path)
                part_index += 1
                buffer.clear()
        if start == 0 or processed == len(pending_ids) or processed % 100 == 0:
            print(
                f"processed={processed}/{len(pending_ids)} "
                f"elapsed={time.perf_counter()-started:.1f}s failures={len(failures)}",
                flush=True,
            )
    completed, parts = completed_videos(checkpoint)
    missing = set(video_ids) - completed
    if failures:
        pd.DataFrame(failures).to_csv(checkpoint / "failures.csv", index=False)
    if missing:
        raise RuntimeError(f"incomplete shard: missing={len(missing)} failures={len(failures)}")
    merged = pd.concat(
        [pd.read_csv(path, float_precision="round_trip") for path in parts], ignore_index=True
    )
    if set(merged["task_id"]) != set(tasks["task_id"]) or merged.duplicated(WINDOW_KEYS).any():
        raise ValueError("candidate raw task coverage or uniqueness mismatch")
    output = (
        args.output_dir
        / "curve_raw"
        / label
        / f"{args.dataset}_shard{args.shard_index:02d}_of_{args.num_shards:02d}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.csv")
    merged.sort_values(WINDOW_KEYS).to_csv(temporary, index=False)
    temporary.replace(output)
    print(f"wrote {len(merged)} windows -> {output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(CALIBRATION_SIZES), required=True)
    parser.add_argument("--candidate-sizes", nargs="+", type=int)
    parser.add_argument("--label")
    parser.add_argument("--split", nargs="+", choices=("calibration", "evaluation"), default=("evaluation",))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--extract-device", default="cuda:0")
    parser.add_argument("--score-device", default="cuda:0")
    parser.add_argument("--video-batch-size", type=int, default=4)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--decode-workers", type=int, default=4)
    parser.add_argument("--seek-gap", type=int, default=96)
    parser.add_argument("--decode-attempts", type=int, default=3)
    parser.add_argument("--checkpoint-rows", type=int, default=1000)
    parser.add_argument(
        "--tasks", type=Path, default=ROOT / "results/duration_aware_23source/protocol_tasks.csv"
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/alpha_stalled_u0_locked.yaml"
    )
    parser.add_argument(
        "--params-dir", type=Path, default=ROOT / "results/duration_aware_23source/params"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/duration_aware_23source"
    )
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require num-shards >= 1 and valid shard-index")
    return args


if __name__ == "__main__":
    run(parse_args())
