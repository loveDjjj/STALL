#!/usr/bin/env python3
"""Stream duration-aware 23-source tasks and persist raw branch scores."""

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

from alpha_stalled.duration_aware_scoring import (
    KEY_COLUMNS,
    MAX_SIZE,
    WINDOW_KEYS,
    decode_video,
)
from alpha_stalled.release_io import video_id_shard
from alpha_stalled.parameters import load_raw_params
from alpha_stalled.global_branch import global_t1_features
from alpha_stalled.local_branch import local_d2_features
from alpha_stalled.whitening import (
    GaussianMeanCandidateScorerFloat64,
    StableGaussianParams,
    score_gaussian_aggregate_float64,
)
from stall_patch import PatchSTALL


def local_param_path(config: dict, params_dir: Path, dataset: str, duration: int, name: str) -> Path:
    if duration == 2 and name == "n200":
        return ROOT / config["local_branch"]["params_by_dataset"][dataset]["path"]
    size = 200 if name == "n200" else MAX_SIZE[dataset]
    return params_dir / f"{dataset}_{duration}s_n{size}.npz"


def load_duration_scorers(
    config: dict, params_dir: Path, dataset: str, duration: int, device: str
) -> tuple[dict, GaussianMeanCandidateScorerFloat64, GaussianMeanCandidateScorerFloat64]:
    locked = load_raw_params(config, dataset)
    spatial = []
    temporal = []
    for name in ("n200", "nmax"):
        path = local_param_path(config, params_dir, dataset, duration, name)
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
        locked,
        GaussianMeanCandidateScorerFloat64(spatial, spatial[0].mean, device=device),
        GaussianMeanCandidateScorerFloat64(temporal, temporal[0].mean, device=device),
    )


@torch.inference_mode()
def score_decoded(
    decoded: list[dict],
    extractor: PatchSTALL,
    scorers: dict[int, tuple[dict, GaussianMeanCandidateScorerFloat64, GaussianMeanCandidateScorerFloat64]],
    score_device: str,
    frame_batch_size: int,
) -> list[dict]:
    extracted = [
        extractor.frames_to_global_patch_embeddings([item["frames"]], batch_size=frame_batch_size)[0]
        for item in decoded
    ]
    grouped: dict[int, list[tuple[dict, int, dict, int, list[int], np.ndarray, np.ndarray]]] = {}
    for item, embeddings in zip(decoded, extracted):
        if tuple(embeddings["grid_size"]) != (14, 14):
            raise ValueError(f"unexpected patch grid {embeddings['grid_size']}")
        for (metadata, window_id, window), positions in zip(item["tasks"], item["positions"]):
            duration = int(metadata["protocol_duration_sec"])
            grouped.setdefault(duration, []).append(
                (
                    metadata,
                    window_id,
                    item,
                    duration,
                    window,
                    embeddings["global"][positions],
                    embeddings["patch"][positions],
                )
            )

    output = []
    for duration, windows in grouped.items():
        locked, spatial_scorer, temporal_scorer = scorers[duration]
        global_batch = torch.from_numpy(np.stack([item[5] for item in windows]).astype(np.float32))
        patch_batch = torch.from_numpy(np.stack([item[6] for item in windows]).astype(np.float32))
        global_spatial, _ = score_gaussian_aggregate_float64(
            global_batch,
            locked["global_spatial"],
            aggregation="max",
            device=score_device,
            compute_percentile=False,
        )
        global_delta, zero = global_t1_features(global_batch)
        global_t1, _ = score_gaussian_aggregate_float64(
            global_delta,
            locked["global_t1"],
            aggregation="min",
            device=score_device,
            invalid_mask=zero,
            compute_percentile=False,
        )
        patch_spatial = spatial_scorer.score(patch_batch)
        patch_d2 = temporal_scorer.score(local_d2_features(patch_batch))
        for index, (metadata, window_id, item, _, frame_indices, _, _) in enumerate(windows):
            output.append(
                {
                    **{column: metadata[column] for column in KEY_COLUMNS},
                    "video_path": metadata["video_path"],
                    "duration_seconds": float(metadata["duration_seconds"]),
                    "effective_k": int(metadata["effective_k"]),
                    "unique_frame_count": len(item["unique_indices"]),
                    "window_id": window_id,
                    "frame_indices": json.dumps(frame_indices, separators=(",", ":")),
                    "global_spatial_raw": global_spatial[index],
                    "global_t1_raw": global_t1[index],
                    "patch_spatial__n200": patch_spatial[index, 0],
                    "patch_d2__n200": patch_d2[index, 0],
                    "patch_spatial__nmax": patch_spatial[index, 1],
                    "patch_d2__nmax": patch_d2[index, 1],
                }
            )
    return output


def completed_videos(directory: Path) -> tuple[set[str], list[Path]]:
    parts = sorted(directory.glob("part_*.csv"))
    completed = set()
    for path in parts:
        completed.update(pd.read_csv(path, usecols=["video_id"])["video_id"].unique())
    return completed, parts


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    tasks = pd.read_csv(args.tasks, float_precision="round_trip")
    tasks = tasks[
        tasks["dataset"].eq(args.dataset)
        & tasks["protocol_split"].isin(args.split)
        & tasks["video_id"].map(
            lambda value: video_id_shard(str(value), args.num_shards)
            == args.shard_index
        )
    ].copy()
    if args.only_video_id:
        tasks = tasks[tasks["video_id"].astype(str).isin(set(args.only_video_id))].copy()
    video_ids = tasks["video_id"].drop_duplicates().tolist()
    if args.debug_videos is not None:
        video_ids = video_ids[: args.debug_videos]
        tasks = tasks[tasks["video_id"].isin(video_ids)].copy()
    checkpoint = (
        args.output_dir
        / "checkpoints"
        / args.dataset
        / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    )
    checkpoint.mkdir(parents=True, exist_ok=True)
    completed, parts = completed_videos(checkpoint)
    pending_ids = [value for value in video_ids if value not in completed]
    print(
        f"dataset={args.dataset} shard={args.shard_index}/{args.num_shards} "
        f"videos={len(video_ids)} completed={len(completed)} pending={len(pending_ids)}",
        flush=True,
    )
    durations = sorted(tasks["protocol_duration_sec"].astype(int).unique())
    scorers = {
        duration: load_duration_scorers(config, args.params_dir, args.dataset, duration, args.score_device)
        for duration in durations
    }
    extractor = PatchSTALL(args.extract_device, data_dict=None, load_dino=True)
    failures = []
    part_index = len(parts)
    buffer: list[dict] = []
    started = time.perf_counter()
    grouped = {key: frame for key, frame in tasks.groupby("video_id", sort=False)}
    for start in range(0, len(pending_ids), args.video_batch_size):
        current = pending_ids[start : start + args.video_batch_size]
        decoded = []
        with ThreadPoolExecutor(max_workers=min(args.decode_workers, len(current))) as pool:
            futures = [
                pool.submit(decode_video, grouped[video_id_value], args.seek_gap, args.decode_attempts)
                for video_id_value in current
            ]
            for video_id_value, future in zip(current, futures):
                try:
                    decoded.append(future.result())
                except Exception as error:
                    failures.append({"video_id": video_id_value, "error": repr(error)})
        if decoded:
            rows = score_decoded(decoded, extractor, scorers, args.score_device, args.frame_batch_size)
            buffer.extend(rows)
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
                f"processed={processed}/{len(pending_ids)} elapsed={time.perf_counter()-started:.1f}s "
                f"failures={len(failures)}",
                flush=True,
            )
    completed, parts = completed_videos(checkpoint)
    missing = set(video_ids) - completed
    if failures:
        pd.DataFrame(failures).to_csv(checkpoint / "failures.csv", index=False)
    if missing:
        raise RuntimeError(f"incomplete shard: missing={len(missing)} failures={len(failures)}")
    merged = pd.concat([pd.read_csv(path, float_precision="round_trip") for path in parts], ignore_index=True)
    expected_tasks = set(tasks["task_id"])
    if set(merged["task_id"]) != expected_tasks or merged.duplicated(WINDOW_KEYS).any():
        raise ValueError("raw task coverage or uniqueness mismatch")
    output = args.output_dir / "raw" / f"{args.dataset}_shard{args.shard_index:02d}_of_{args.num_shards:02d}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.csv")
    merged.sort_values(WINDOW_KEYS).to_csv(temporary, index=False)
    temporary.replace(output)
    print(f"wrote {len(merged)} windows -> {output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(MAX_SIZE), required=True)
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
    parser.add_argument("--split", nargs="+", choices=("calibration", "evaluation"), default=("calibration", "evaluation"))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--extract-device", default="cuda")
    parser.add_argument("--score-device", default="cuda")
    parser.add_argument("--video-batch-size", type=int, default=2)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--decode-workers", type=int, default=4)
    parser.add_argument("--decode-attempts", type=int, default=3)
    parser.add_argument("--seek-gap", type=int, default=64)
    parser.add_argument("--checkpoint-rows", type=int, default=1000)
    parser.add_argument("--debug-videos", type=int)
    parser.add_argument("--only-video-id", nargs="+")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require num_shards >= 1 and valid shard-index")
    return args


if __name__ == "__main__":
    run(parse_args())
