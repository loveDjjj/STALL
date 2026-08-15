#!/usr/bin/env python3
"""Stream DINOv3 extraction and frozen Alpha-STALLED scoring for many windows."""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval_patch_fast import FastPatchScorer
from alpha_stalled.legacy_window_scoring import (
    KEY_COLUMNS,
    LOCAL_PARAMS,
    SAMPLINGS,
    decode_manifest_row,
    decode_manifest_row_with_retries as _shared_decode_with_retries,
    load_completed,
    load_windows,
    resolve_video_path,
    score_batch,
    stable_shard,
    video_key,
)
from alpha_stalled.video_io import decode_selected_frames, decode_spans
from stall import STALL
from stall_patch import PatchSTALL


# Backward compatibility for tests and legacy scripts importing the private name.
_decode_spans = decode_spans


def decode_manifest_row_with_retries(
    row: pd.Series,
    sampling: str,
    seek_gap: int,
    attempts: int,
    retry_delay: float = 0.25,
) -> dict:
    return _shared_decode_with_retries(
        row,
        sampling,
        seek_gap,
        attempts,
        retry_delay,
        decoder=decode_manifest_row,
    )


def run(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest["dataset"] == args.dataset].copy()
    if args.protocol_split != "all":
        manifest = manifest[manifest["protocol_split"] == args.protocol_split].copy()
    manifest["active_sampling"] = args.sampling
    manifest = manifest[
        manifest.apply(lambda row: stable_shard(row, args.num_shards) == args.shard_index, axis=1)
    ].reset_index(drop=True)
    if args.debug_videos is not None:
        manifest = manifest.head(args.debug_videos).copy()

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
    print(
        f"dataset={args.dataset} sampling={args.sampling} shard={args.shard_index}/{args.num_shards} "
        f"expected={len(manifest)} completed={len(completed)} pending={len(pending)}",
        flush=True,
    )

    local_params = args.local_params or LOCAL_PARAMS[args.dataset]
    if not local_params.exists():
        raise FileNotFoundError(
            f"missing leakage-free L0 params: {local_params}; run tools/run_local_d2_residuals.py first"
        )
    global_data = np.load(args.global_params, allow_pickle=True)
    global_scorer = STALL(args.device, global_data, load_dino=False)
    local_scorer = FastPatchScorer(str(local_params), device=args.device)
    local_scorer.validate("same_grid_second_order")
    extractor = PatchSTALL(args.device, data_dict=None, load_dino=True)

    started = time.perf_counter()
    next_part = len(parts)
    failures: list[dict] = []
    for start in range(0, len(pending), args.video_batch_size):
        batch = [row for _, row in pending.iloc[start : start + args.video_batch_size].iterrows()]
        with ThreadPoolExecutor(max_workers=min(args.decode_workers, len(batch))) as executor:
            futures = [
                executor.submit(
                    decode_manifest_row_with_retries,
                    row,
                    args.sampling,
                    args.seek_gap,
                    args.decode_attempts,
                )
                for row in batch
            ]
            decoded: list[dict] = []
            for row, future in zip(batch, futures):
                try:
                    decoded.append(future.result())
                except Exception as exc:
                    failures.append(
                        {**{column: row[column] for column in KEY_COLUMNS}, "error": str(exc)}
                    )
        if decoded:
            rows = score_batch(
                decoded,
                extractor,
                global_scorer,
                local_scorer,
                args.frame_batch_size,
            )
            part = checkpoint_dir / f"part_{next_part:06d}.csv"
            temp = part.with_suffix(".tmp.csv")
            pd.DataFrame(rows).to_csv(temp, index=False)
            temp.replace(part)
            next_part += 1
        processed = min(start + args.video_batch_size, len(pending))
        if processed == len(pending) or processed % max(args.video_batch_size * 10, 1) == 0:
            elapsed = time.perf_counter() - started
            rate = processed / elapsed if elapsed else 0.0
            print(
                f"processed={processed}/{len(pending)} failures={len(failures)} "
                f"videos_per_sec={rate:.3f}",
                flush=True,
            )

    if local_scorer._executor is not None:
        local_scorer._executor.shutdown(wait=True)
    if failures:
        pd.DataFrame(failures).to_csv(checkpoint_dir / "failures.csv", index=False)

    completed, parts = load_completed(checkpoint_dir)
    expected = {video_key(row) for _, row in manifest.iterrows()}
    missing = expected - completed
    if missing:
        raise RuntimeError(f"incomplete shard: missing={len(missing)}, failures={len(failures)}")
    merged = pd.concat(
        [pd.read_csv(part, float_precision="round_trip") for part in parts],
        ignore_index=True,
    )
    merged = merged.drop_duplicates(KEY_COLUMNS + ["window_id"], keep="last")
    merged = merged.sort_values(KEY_COLUMNS + ["window_id"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".tmp.csv")
    merged.to_csv(temp, index=False)
    temp.replace(args.output)
    elapsed = time.perf_counter() - started
    print(f"complete videos={len(expected)} windows={len(merged)} elapsed_sec={elapsed:.1f}")
    print(f"saved -> {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(LOCAL_PARAMS), required=True)
    parser.add_argument("--sampling", choices=SAMPLINGS, required=True)
    parser.add_argument(
        "--protocol-split",
        choices=("all", "calibration", "evaluation"),
        default="all",
    )
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
    parser.add_argument("--debug-videos", type=int)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("/tmp/alpha_stalled_multi_window"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require num_shards >= 1 and 0 <= shard_index < num_shards")
    if args.decode_attempts < 1:
        parser.error("decode-attempts must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
