#!/usr/bin/env python3
"""Score Local patch D1 on locked U0 K=3 windows without caching patch tokens."""

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


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.video_io import decode_selected_frames
from alpha_stalled.release_io import resolve_required_video as resolve_video, video_id_shard
from alpha_stalled.artifacts import checkpoint_completed_ids
from alpha_stalled.local_branch import local_d1_features
from alpha_stalled.u0_protocol import KEY_COLUMNS, load_release_rows
from alpha_stalled.whitening import (
    StableGaussianParams,
    score_gaussian_aggregate_float64,
)
from stall_patch import PatchSTALL


CACHE_ROOTS = {
    "comgenvid": ROOT / "cache/patch_embeddings/comgenvid",
    "videofeedback": ROOT / "cache/patch_embeddings/videofeedback",
    "genvideo": ROOT / "cache/patch_embeddings/genvideo",
}


def load_exact_k1_cache(row: pd.Series, windows: list[list[int]]) -> torch.Tensor | None:
    if len(windows) != 1:
        return None
    path = (
        CACHE_ROOTS[str(row["dataset"])]
        / str(row["subset"])
        / str(row["source_model"])
        / f"{Path(str(row['filename'])).stem}_2s.pt"
    )
    if not path.is_file():
        return None
    payload = torch.load(path, weights_only=True, map_location="cpu")
    actual = [int(value) for value in payload["frame_indices"]]
    expected = [int(value) for value in windows[0]]
    if actual != expected:
        return None
    patch = payload["patch"].float()
    if tuple(patch.shape) != (16, 196, 1024):
        raise ValueError(f"invalid cached patch shape: {path} -> {tuple(patch.shape)}")
    return patch


def decode_row(row: pd.Series, windows: list[list[int]], seek_gap: int, attempts: int) -> dict:
    unique_indices = sorted({int(index) for window in windows for index in window})
    error = None
    for attempt in range(attempts):
        try:
            frames = decode_selected_frames(
                resolve_video(str(row["video_path"])), unique_indices, seek_gap
            )
            positions = {index: position for position, index in enumerate(unique_indices)}
            return {
                "row": row,
                "windows": windows,
                "positions": [[positions[index] for index in window] for window in windows],
                "frames": frames,
            }
        except Exception as caught:
            error = caught
            if attempt + 1 < attempts:
                time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"decode failed after {attempts} attempts: {error}") from error


def completed_videos(directory: Path) -> tuple[set[str], int]:
    completed, paths = checkpoint_completed_ids(directory, cast_str=True)
    return {str(value) for value in completed}, len(paths)


@torch.inference_mode()
def score_patch_windows(
    patches: list[np.ndarray | torch.Tensor],
    metadata: list[tuple[pd.Series, int, list[int], str]],
    params: StableGaussianParams,
    score_device: str,
) -> list[dict]:
    batch = torch.stack([torch.as_tensor(value, dtype=torch.float32) for value in patches])
    raw, _ = score_gaussian_aggregate_float64(
        local_d1_features(batch),
        params,
        aggregation="mean",
        device=score_device,
        compute_percentile=False,
    )
    rows = []
    for index, (source, window_id, indices, feature_source) in enumerate(metadata):
        rows.append(
            {
                **{column: source[column] for column in KEY_COLUMNS},
                "effective_k": int(source["effective_k"]),
                "window_id": window_id,
                "frame_indices": json.dumps(indices, separators=(",", ":")),
                "feature_source": feature_source,
                "patch_d1_raw": float(raw[index]),
            }
        )
    return rows


@torch.inference_mode()
def score_decoded(
    decoded: list[dict],
    extractor: PatchSTALL,
    params: StableGaussianParams,
    extract_batch_size: int,
    score_device: str,
) -> list[dict]:
    extracted = [
        extractor.frames_to_global_patch_embeddings(
            [item["frames"]], batch_size=extract_batch_size
        )[0]
        for item in decoded
    ]
    patches = []
    metadata = []
    for item, output in zip(decoded, extracted):
        if tuple(output["grid_size"]) != (14, 14):
            raise ValueError(f"unexpected patch grid: {output['grid_size']}")
        for window_id, positions in enumerate(item["positions"]):
            patches.append(output["patch"][positions])
            metadata.append(
                (
                    item["row"],
                    window_id,
                    item["windows"][window_id],
                    "dino_forward",
                )
            )
    return score_patch_windows(patches, metadata, params, score_device)


def run(args: argparse.Namespace) -> None:
    manifest, frame_indices = load_release_rows(args.release_dir)
    if args.sampling == "calibration_k1":
        payload = json.loads((args.release_dir / "frame_indices.json").read_text())
        references = payload["calibration_reference_windows"]
        manifest = manifest[manifest["protocol_split"].eq("calibration")].copy()
        frame_indices = {
            video_id: [indices] for video_id, indices in references.items()
        }
        manifest["effective_k"] = 1
    manifest = manifest[
        manifest["dataset"].eq(args.dataset)
        & manifest["video_id"].map(
            lambda value: video_id_shard(str(value), args.num_shards)
            == args.shard_index
        )
    ].reset_index(drop=True)
    if args.debug_videos is not None:
        manifest = manifest.head(args.debug_videos).copy()
    checkpoint = (
        args.output_dir
        / "checkpoints"
        / args.sampling
        / args.dataset
        / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    )
    checkpoint.mkdir(parents=True, exist_ok=True)
    completed, part_index = completed_videos(checkpoint)
    pending = manifest[~manifest["video_id"].isin(completed)].reset_index(drop=True)
    print(
        f"dataset={args.dataset} shard={args.shard_index}/{args.num_shards} "
        f"expected={len(manifest)} completed={len(completed)} pending={len(pending)}",
        flush=True,
    )
    parameter_path = args.params_dir / f"{args.dataset}_region1_mean_d1.npz"
    params = StableGaussianParams.from_npz(
        str(parameter_path),
        mean_key="mu_patch_temp",
        whitening_key="W_patch_temp",
        calibration_key="calib_patch_temp_scores",
    )
    extractor = None
    failures = []
    buffer = []
    started = time.perf_counter()
    for start in range(0, len(pending), args.video_batch_size):
        rows = [row for _, row in pending.iloc[start : start + args.video_batch_size].iterrows()]
        cached = []
        decode_rows = []
        for row in rows:
            windows = frame_indices[str(row["video_id"])]
            patch = load_exact_k1_cache(row, windows) if args.reuse_k1_cache else None
            if patch is None:
                decode_rows.append(row)
            else:
                cached.append((row, windows[0], patch))
        if cached:
            buffer.extend(
                score_patch_windows(
                    [item[2] for item in cached],
                    [(item[0], 0, item[1], "exact_k1_cache") for item in cached],
                    params,
                    args.score_device,
                )
            )
        decoded = []
        if decode_rows:
            with ThreadPoolExecutor(
                max_workers=min(args.decode_workers, len(decode_rows))
            ) as pool:
                futures = [
                    pool.submit(
                        decode_row,
                        row,
                        frame_indices[str(row["video_id"])],
                        args.seek_gap,
                        args.decode_attempts,
                    )
                    for row in decode_rows
                ]
                for row, future in zip(decode_rows, futures):
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
            if extractor is None:
                extractor = PatchSTALL(args.extract_device, data_dict=None, load_dino=True)
            buffer.extend(
                score_decoded(
                    decoded,
                    extractor,
                    params,
                    args.frame_batch_size,
                    args.score_device,
                )
            )
        processed = min(start + args.video_batch_size, len(pending))
        if len(buffer) >= args.checkpoint_rows or processed == len(pending):
            if buffer:
                path = checkpoint / f"part_{part_index:06d}.csv"
                temporary = path.with_suffix(".tmp.csv")
                pd.DataFrame(buffer).to_csv(temporary, index=False)
                temporary.replace(path)
                part_index += 1
                buffer.clear()
        if start == 0 or processed == len(pending) or processed % 100 == 0:
            print(
                f"processed={processed}/{len(pending)} "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )
    failure_path = checkpoint / "failures.csv"
    pd.DataFrame(failures, columns=[*KEY_COLUMNS, "error"]).to_csv(
        failure_path, index=False
    )
    if failures:
        raise RuntimeError(f"{len(failures)} videos failed; see {failure_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("comgenvid", "videofeedback", "genvideo"), required=True)
    parser.add_argument(
        "--sampling", choices=("k3", "calibration_k1"), default="k3"
    )
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--extract-device", default="cuda:0")
    parser.add_argument("--score-device", default="cuda:0")
    parser.add_argument("--video-batch-size", type=int, default=1)
    parser.add_argument(
        "--frame-batch-size",
        type=int,
        default=32,
        help="must remain 32 for the locked U0 D1/D2 comparison",
    )
    parser.add_argument("--decode-workers", type=int, default=2)
    parser.add_argument("--seek-gap", type=int, default=48)
    parser.add_argument("--decode-attempts", type=int, default=3)
    parser.add_argument("--checkpoint-rows", type=int, default=1000)
    parser.add_argument("--debug-videos", type=int)
    parser.add_argument(
        "--no-reuse-k1-cache",
        dest="reuse_k1_cache",
        action="store_false",
        help="disable exact locked K1 patch-cache reuse",
    )
    parser.set_defaults(reuse_k1_cache=False)
    parser.add_argument("--release-dir", type=Path, default=ROOT / "release/u0")
    parser.add_argument(
        "--params-dir",
        type=Path,
        default=ROOT / "results/second_order_independent_calibration/params",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/second_order_independent_calibration/d1_windows_b32",
    )
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, num-shards)")
    return args


if __name__ == "__main__":
    run(parse_args())
