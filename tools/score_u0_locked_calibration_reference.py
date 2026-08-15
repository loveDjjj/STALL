#!/usr/bin/env python3
"""Score the 600 locked K1 windows used by U0 Local first-level CDFs."""

from __future__ import annotations

import argparse
import json
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

from alpha_stalled.u0_protocol import KEY_COLUMNS
from alpha_stalled.u0_scoring import decode_row, score_batch
from alpha_stalled.release_io import video_id_shard
from alpha_stalled.parameters import load_raw_params
from alpha_stalled.artifacts import checkpoint_completed_ids, read_checkpoint_parts
from stall_patch import PatchSTALL


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    calibration = json.loads(
        (args.release_dir / "calibration_manifest.json").read_text()
    )["videos"]
    references = json.loads(
        (args.release_dir / "frame_indices.json").read_text()
    )["calibration_reference_windows"]
    manifest = pd.DataFrame(calibration)
    manifest = manifest[
        (manifest["dataset"] == args.dataset)
        & manifest["video_id"].map(
            lambda value: video_id_shard(value, args.num_shards) == args.shard_index
        )
    ].reset_index(drop=True)
    checkpoint_dir = (
        args.output_dir
        / "calibration_reference_checkpoints"
        / args.dataset
        / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    completed, parts = checkpoint_completed_ids(checkpoint_dir, cast_str=True)
    pending = manifest[~manifest["video_id"].isin(completed)].reset_index(drop=True)
    print(
        f"calibration-reference dataset={args.dataset} shard={args.shard_index}/{args.num_shards} "
        f"expected={len(manifest)} completed={len(completed)} pending={len(pending)}",
        flush=True,
    )
    params = load_raw_params(config, args.dataset, args.local_params)
    extractor = PatchSTALL(args.extract_device, data_dict=None, load_dino=True)
    failures = []
    next_part = len(parts)
    started = time.perf_counter()
    for start in range(0, len(pending), args.video_batch_size):
        rows = [
            row
            for _, row in pending.iloc[start : start + args.video_batch_size].iterrows()
        ]
        decoded = []
        with ThreadPoolExecutor(max_workers=min(args.decode_workers, len(rows))) as pool:
            futures = []
            for row in rows:
                reference_row = row.copy()
                reference_row["effective_k"] = 1
                futures.append(
                    pool.submit(
                        decode_row,
                        reference_row,
                        [references[row["video_id"]]],
                        args.seek_gap,
                        args.decode_attempts,
                    )
                )
            for row, future in zip(rows, futures):
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
            scored = score_batch(
                decoded,
                extractor,
                params,
                args.score_device,
                args.frame_batch_size,
            )
            for row in scored:
                row["window_role"] = "local_cdf_reference_K1_current"
            part = checkpoint_dir / f"part_{next_part:06d}.csv"
            temporary = part.with_suffix(".tmp.csv")
            pd.DataFrame(scored).to_csv(temporary, index=False)
            temporary.replace(part)
            next_part += 1
        processed = min(start + args.video_batch_size, len(pending))
        if start == 0 or processed == len(pending) or next_part % 25 == 0:
            print(
                f"processed={processed}/{len(pending)} "
                f"elapsed={time.perf_counter()-started:.1f}s failures={len(failures)}",
                flush=True,
            )
    merged, parts = read_checkpoint_parts(
        checkpoint_dir,
        empty_error=f"no calibration checkpoint parts: {checkpoint_dir}",
    )
    expected_ids = set(manifest["video_id"])
    if failures:
        pd.DataFrame(failures).to_csv(checkpoint_dir / "failures.csv", index=False)
    if set(merged["video_id"]) != expected_ids or merged["video_id"].duplicated().any():
        raise RuntimeError(
            f"incomplete calibration reference shard: expected={len(expected_ids)} "
            f"actual={merged['video_id'].nunique()} failures={len(failures)}"
        )
    expected_frames = {
        video_id: json.dumps(references[video_id], separators=(",", ":"))
        for video_id in expected_ids
    }
    actual_frames = dict(zip(merged["video_id"], merged["frame_indices"]))
    if actual_frames != expected_frames:
        raise ValueError("calibration reference frames differ from locked manifest")
    output = (
        args.output_dir
        / "calibration_raw"
        / f"{args.dataset}_shard{args.shard_index:02d}_of_{args.num_shards:02d}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.csv")
    merged.sort_values("video_id").to_csv(temporary, index=False)
    temporary.replace(output)
    print(f"wrote {len(merged)} calibration references -> {output}", flush=True)


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
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--video-batch-size", type=int, default=4)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--decode-workers", type=int, default=4)
    parser.add_argument("--decode-attempts", type=int, default=3)
    parser.add_argument("--seek-gap", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
