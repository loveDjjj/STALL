#!/usr/bin/env python3
"""Extract and score locked U0 raw window components from source videos."""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.release_io import (
    resolve_required_video as resolve_video,
    video_id_shard as stable_shard,
)
from alpha_stalled.parameters import load_raw_params
from alpha_stalled.artifacts import checkpoint_completed_ids, read_csv_files
from alpha_stalled.u0_protocol import (
    KEY_COLUMNS as U0_KEY_COLUMNS,
    WINDOW_KEYS as U0_WINDOW_KEYS,
    load_release_rows,
)
from alpha_stalled.u0_scoring import (
    decode_row,
    score_batch,
    score_global_raw,
    score_local_raw,
)
from stall_patch import PatchSTALL


KEY_COLUMNS = list(U0_KEY_COLUMNS)
WINDOW_KEYS = list(U0_WINDOW_KEYS)


def completed_videos(checkpoint_dir: Path) -> tuple[set[str], list[Path]]:
    completed, parts = checkpoint_completed_ids(checkpoint_dir, cast_str=True)
    return {str(value) for value in completed}, parts


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest, frame_indices = load_release_rows(args.release_dir)
    manifest = manifest[
        (manifest["dataset"] == args.dataset)
        & manifest["video_id"].map(
            lambda value: stable_shard(value, args.num_shards) == args.shard_index
        )
    ].reset_index(drop=True)
    if args.debug_videos is not None:
        manifest = manifest.head(args.debug_videos).copy()
    checkpoint_dir = (
        args.output_dir
        / "checkpoints"
        / args.dataset
        / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    completed, parts = completed_videos(checkpoint_dir)
    pending = manifest[~manifest["video_id"].isin(completed)].reset_index(drop=True)
    print(
        f"dataset={args.dataset} shard={args.shard_index}/{args.num_shards} "
        f"expected={len(manifest)} completed={len(completed)} pending={len(pending)}",
        flush=True,
    )

    params = load_raw_params(config, args.dataset, args.local_params)
    extractor = PatchSTALL(args.extract_device, data_dict=None, load_dino=True)
    failures = []
    next_part = len(parts)
    started = time.perf_counter()
    for start in range(0, len(pending), args.video_batch_size):
        source_rows = [
            row
            for _, row in pending.iloc[start : start + args.video_batch_size].iterrows()
        ]
        decoded = []
        with ThreadPoolExecutor(max_workers=min(args.decode_workers, len(source_rows))) as pool:
            futures = [
                pool.submit(
                    decode_row,
                    row,
                    frame_indices[row["video_id"]],
                    args.seek_gap,
                    args.decode_attempts,
                )
                for row in source_rows
            ]
            for row, future in zip(source_rows, futures):
                try:
                    decoded.append(future.result())
                except Exception as error:
                    failures.append(
                        {
                            **{column: row[column] for column in KEY_COLUMNS},
                            "error": repr(error),
                        }
                    )
        if decoded:
            rows = score_batch(
                decoded,
                extractor,
                params,
                args.score_device,
                args.frame_batch_size,
            )
            part = checkpoint_dir / f"part_{next_part:06d}.csv"
            temporary = part.with_suffix(".tmp.csv")
            pd.DataFrame(rows).to_csv(temporary, index=False)
            temporary.replace(part)
            next_part += 1
        processed = min(start + args.video_batch_size, len(pending))
        if start == 0 or processed == len(pending) or next_part % 50 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"processed={processed}/{len(pending)} elapsed={elapsed:.1f}s "
                f"failures={len(failures)}",
                flush=True,
            )

    completed, parts = completed_videos(checkpoint_dir)
    expected = set(manifest["video_id"])
    if failures:
        pd.DataFrame(failures).to_csv(checkpoint_dir / "failures.csv", index=False)
    if expected - completed:
        raise RuntimeError(
            f"incomplete locked shard: missing={len(expected-completed)} failures={len(failures)}"
        )
    merged = read_csv_files(parts)
    if merged.duplicated(WINDOW_KEYS).any():
        raise ValueError("duplicate locked window keys")
    output = (
        args.output_dir
        / "raw"
        / f"{args.dataset}_shard{args.shard_index:02d}_of_{args.num_shards:02d}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.csv")
    merged.sort_values(WINDOW_KEYS).to_csv(temporary, index=False)
    temporary.replace(output)
    print(f"wrote {len(merged)} raw windows -> {output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/alpha_stalled_u0_locked.yaml"
    )
    parser.add_argument("--release-dir", type=Path, default=ROOT / "release/u0")
    parser.add_argument(
        "--local-params",
        type=Path,
        help="Explicit real-only Local parameter file for a locked external dataset.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extract-device", default="cuda")
    parser.add_argument("--score-device", default="cuda")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--video-batch-size", type=int, default=4)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--decode-workers", type=int, default=4)
    parser.add_argument("--decode-attempts", type=int, default=3)
    parser.add_argument("--seek-gap", type=int, default=64)
    parser.add_argument("--debug-videos", type=int)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require num_shards >= 1 and valid shard-index")
    return args


if __name__ == "__main__":
    run(parse_args())
