#!/usr/bin/env python3
"""Score a larger window set while exactly reusing previously scored windows."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
TOOLS_DIR = REPO_ROOT / "tools"
for directory in (SRC_DIR, TOOLS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from eval_patch_fast import FastPatchScorer
from alpha_stalled.legacy_window_scoring import (
    KEY_COLUMNS,
    LOCAL_PARAMS,
    SAMPLINGS,
    decode_manifest_row_with_retries,
    load_completed,
    load_windows,
    score_batch,
    stable_shard,
    video_key,
)
from stall import STALL
from stall_patch import PatchSTALL


SCORE_COLUMNS = [
    "global_spatial",
    "global_t1",
    "G_k",
    "patch_spatial",
    "patch_d2",
    "L_k",
    "S_k",
]


def frame_key(value: str | list[int]) -> tuple[int, ...]:
    indices = json.loads(value) if isinstance(value, str) else value
    return tuple(int(index) for index in indices)


def load_reuse_lookup(paths: list[Path]) -> dict[tuple[tuple[str, ...], tuple[int, ...]], dict]:
    lookup: dict[tuple[tuple[str, ...], tuple[int, ...]], dict] = {}
    for path in paths:
        frame = pd.read_csv(path, float_precision="round_trip")
        for row in frame.to_dict("records"):
            key = (video_key(row), frame_key(row["frame_indices"]))
            if key in lookup:
                for column in SCORE_COLUMNS:
                    if not np.isclose(float(lookup[key][column]), float(row[column]), atol=1e-7):
                        raise ValueError(f"conflicting reused score for {key}: {column}")
            else:
                lookup[key] = row
    return lookup


def prepare_row(
    row: pd.Series,
    sampling: str,
    reuse: dict[tuple[tuple[str, ...], tuple[int, ...]], dict],
) -> tuple[list[list[int]], dict[int, dict], list[list[int]]]:
    windows = load_windows(row, sampling)
    reused: dict[int, dict] = {}
    missing: list[list[int]] = []
    key = video_key(row)
    for window_id, window in enumerate(windows):
        prior = reuse.get((key, tuple(window)))
        if prior is None:
            missing.append(window)
        else:
            reused[window_id] = prior
    return windows, reused, missing


def normalized_row(
    source: pd.Series,
    sampling: str,
    windows: list[list[int]],
    window_id: int,
    scores: dict,
) -> dict:
    unique_count = len({index for window in windows for index in window})
    return {
        **{column: source[column] for column in KEY_COLUMNS},
        "video_path": source["video_path"],
        "duration_seconds": source["duration_seconds"],
        "sampling": sampling,
        "effective_k": len(windows),
        "unique_frame_count": unique_count,
        "window_id": window_id,
        "frame_indices": json.dumps(windows[window_id], separators=(",", ":")),
        **{column: float(scores[column]) for column in SCORE_COLUMNS},
    }


def run(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest["dataset"] == args.dataset].copy()
    manifest["active_sampling"] = args.sampling
    manifest = manifest[
        manifest.apply(lambda row: stable_shard(row, args.num_shards) == args.shard_index, axis=1)
    ].reset_index(drop=True)
    reuse = load_reuse_lookup(args.reuse_score)
    checkpoint_dir = (
        args.checkpoint_dir
        / args.sampling
        / args.dataset
        / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    completed, parts = load_completed(checkpoint_dir)
    pending = manifest[
        ~manifest.apply(lambda row: video_key(row) in completed, axis=1)
    ].reset_index(drop=True)

    prepared = [prepare_row(row, args.sampling, reuse) for _, row in pending.iterrows()]
    total_windows = sum(len(item[0]) for item in prepared)
    reused_windows = sum(len(item[1]) for item in prepared)
    print(
        f"dataset={args.dataset} sampling={args.sampling} shard={args.shard_index}/{args.num_shards} "
        f"expected={len(manifest)} completed={len(completed)} pending={len(pending)} "
        f"reuse={reused_windows}/{total_windows}",
        flush=True,
    )

    local_params = args.local_params or LOCAL_PARAMS[args.dataset]
    global_scorer = STALL(
        args.device,
        np.load(args.global_params, allow_pickle=True),
        load_dino=False,
    )
    local_scorer = FastPatchScorer(str(local_params), device=args.device)
    local_scorer.validate("same_grid_second_order")
    extractor = PatchSTALL(args.device, data_dict=None, load_dino=True)

    started = time.perf_counter()
    next_part = len(parts)
    failures: list[dict] = []
    for start in range(0, len(pending), args.video_batch_size):
        batch_frame = pending.iloc[start : start + args.video_batch_size]
        batch_prepared = prepared[start : start + args.video_batch_size]
        decoded_jobs: list[tuple[pd.Series, list[list[int]]]] = []
        rows: list[dict] = []
        for (_, source), (windows, reused, missing) in zip(batch_frame.iterrows(), batch_prepared):
            for window_id, prior in reused.items():
                rows.append(normalized_row(source, args.sampling, windows, window_id, prior))
            if missing:
                decode_row = source.copy()
                decode_row[f"indices_{args.sampling}"] = json.dumps(missing, separators=(",", ":"))
                decoded_jobs.append((decode_row, windows))
        decoded = []
        failed_keys: set[tuple[str, ...]] = set()
        if decoded_jobs:
            with ThreadPoolExecutor(max_workers=min(args.decode_workers, len(decoded_jobs))) as executor:
                futures = [
                    executor.submit(
                        decode_manifest_row_with_retries,
                        source,
                        args.sampling,
                        args.seek_gap,
                        args.decode_attempts,
                    )
                    for source, _ in decoded_jobs
                ]
                valid_jobs: list[tuple[pd.Series, list[list[int]]]] = []
                for (source, windows), future in zip(decoded_jobs, futures):
                    try:
                        decoded.append(future.result())
                        valid_jobs.append((source, windows))
                    except Exception as exc:
                        failed_keys.add(video_key(source))
                        failures.append(
                            {**{column: source[column] for column in KEY_COLUMNS}, "error": str(exc)}
                        )
            if decoded:
                new_rows = score_batch(
                    decoded,
                    extractor,
                    global_scorer,
                    local_scorer,
                    args.frame_batch_size,
                )
                new_lookup = {
                    (video_key(row), frame_key(row["frame_indices"])): row
                    for row in new_rows
                }
                for source, windows in valid_jobs:
                    key = video_key(source)
                    for window_id, window in enumerate(windows):
                        if any(video_key(existing) == key and existing["window_id"] == window_id for existing in rows):
                            continue
                        scores = new_lookup[(key, tuple(window))]
                        rows.append(normalized_row(source, args.sampling, windows, window_id, scores))
        if failed_keys:
            rows = [row for row in rows if video_key(row) not in failed_keys]
        if rows:
            part = checkpoint_dir / f"part_{next_part:06d}.csv"
            temp = part.with_suffix(".tmp.csv")
            pd.DataFrame(rows).to_csv(temp, index=False)
            temp.replace(part)
            next_part += 1
        processed = min(start + args.video_batch_size, len(pending))
        if processed == len(pending) or processed % max(args.video_batch_size * 10, 1) == 0:
            elapsed = time.perf_counter() - started
            print(
                f"processed={processed}/{len(pending)} failures={len(failures)} "
                f"videos_per_sec={processed / elapsed:.3f}",
                flush=True,
            )

    if local_scorer._executor is not None:
        local_scorer._executor.shutdown(wait=True)
    if failures:
        pd.DataFrame(failures).to_csv(checkpoint_dir / "failures.csv", index=False)
    completed, parts = load_completed(checkpoint_dir)
    expected = {video_key(row) for _, row in manifest.iterrows()}
    if expected - completed:
        raise RuntimeError(f"incomplete shard: missing={len(expected - completed)}")
    merged = pd.concat(
        [pd.read_csv(part, float_precision="round_trip") for part in parts],
        ignore_index=True,
    )
    merged = merged.drop_duplicates(KEY_COLUMNS + ["window_id"], keep="last")
    merged = merged.sort_values(KEY_COLUMNS + ["window_id"]).reset_index(drop=True)
    expected_counts = {
        video_key(row): len(load_windows(row, args.sampling))
        for _, row in manifest.iterrows()
    }
    actual_counts = merged.groupby(KEY_COLUMNS, observed=True).size().to_dict()
    mismatched = {
        key: (expected_count, int(actual_counts.get(key, 0)))
        for key, expected_count in expected_counts.items()
        if int(actual_counts.get(key, 0)) != expected_count
    }
    if mismatched:
        raise RuntimeError(f"incomplete window rows for {len(mismatched)} videos")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".tmp.csv")
    merged.to_csv(temp, index=False)
    temp.replace(args.output)
    print(f"complete videos={len(expected)} windows={len(merged)} elapsed_sec={time.perf_counter() - started:.1f}")
    print(f"saved -> {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(LOCAL_PARAMS), required=True)
    parser.add_argument("--sampling", choices=SAMPLINGS, required=True)
    parser.add_argument("--reuse-score", type=Path, action="append", required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "results/multi_window_joint_typicality/multi_window_manifest.csv",
    )
    parser.add_argument(
        "--global-params",
        type=Path,
        default=REPO_ROOT / "precomputed/stall_params_vatex_dino_v3.npz",
    )
    parser.add_argument("--local-params", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--video-batch-size", type=int, default=4)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--decode-workers", type=int, default=4)
    parser.add_argument("--decode-attempts", type=int, default=3)
    parser.add_argument("--seek-gap", type=int, default=64)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("/tmp/alpha_stalled_multi_window_incremental"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sampling == "K1_current":
        parser.error("incremental scoring is intended for a larger target window set")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require num_shards >= 1 and 0 <= shard_index < num_shards")
    if args.decode_attempts < 1:
        parser.error("decode-attempts must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
