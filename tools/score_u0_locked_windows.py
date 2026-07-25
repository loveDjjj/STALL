#!/usr/bin/env python3
"""Extract and score locked U0 raw window components from source videos."""

from __future__ import annotations

import argparse
import hashlib
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
for directory in (ROOT / "src", ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from score_multi_window import decode_selected_frames
from stable_whitening import (
    StableGaussianParams,
    l2_normalized_first_order,
    l2_normalized_second_order,
    score_gaussian_aggregate_float64,
    stable_sorted,
)
from stall_patch import PatchSTALL


KEY_COLUMNS = [
    "video_id",
    "dataset",
    "protocol_split",
    "subset",
    "source_model",
    "filename",
]
WINDOW_KEYS = [*KEY_COLUMNS, "window_id"]
RAW_COLUMNS = [
    "global_spatial_raw",
    "global_t1_raw",
    "patch_spatial_raw",
    "patch_d2_raw",
]


def stable_shard(video_id: str, num_shards: int) -> int:
    return int(video_id[:16], 16) % num_shards


def resolve_video(value: str) -> Path:
    path = Path(value)
    candidates = (path, ROOT / path, ROOT.parent / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"video not found: {value}")


def load_release_rows(release_dir: Path) -> tuple[pd.DataFrame, dict[str, list[list[int]]]]:
    records = []
    for name in ("calibration_manifest.json", "evaluation_manifest.json"):
        payload = json.loads((release_dir / name).read_text(encoding="utf-8"))
        records.extend(payload["videos"])
    frame_payload = json.loads(
        (release_dir / "frame_indices.json").read_text(encoding="utf-8")
    )
    frame_indices = frame_payload["videos"]
    rows = pd.DataFrame(records)
    if rows["video_id"].duplicated().any():
        raise ValueError("duplicate video_id in locked release manifests")
    if set(rows["video_id"]) != set(frame_indices):
        raise ValueError("release manifests and frame-index keys differ")
    return rows, frame_indices


def load_raw_params(
    config: dict,
    dataset: str,
    local_params_override: Path | None = None,
) -> dict[str, StableGaussianParams]:
    global_data = np.load(ROOT / config["global_branch"]["params"], allow_pickle=True)
    global_spatial_reference = stable_sorted(
        np.max(global_data["calib_ll_spat"].astype(np.float64), axis=1)
    )
    global_temporal_reference = stable_sorted(
        np.min(global_data["calib_ll_temp"].astype(np.float64), axis=1)
    )
    local_path = (
        local_params_override
        if local_params_override is not None
        else ROOT / config["local_branch"]["params_by_dataset"][dataset]["path"]
    )
    local_data = np.load(local_path, allow_pickle=True)
    # Local CDF references are rebuilt from this release traversal. The stored
    # arrays are placeholders here because this stage persists raw scores only.
    placeholder = np.array([0.0], dtype=np.float64)
    return {
        "global_spatial": StableGaussianParams(
            mean=global_data["mu_spat"].astype(np.float64),
            whitening=global_data["W_spat"].astype(np.float64),
            calibration_raw=global_spatial_reference,
        ),
        "global_t1": StableGaussianParams(
            mean=global_data["mu_temp"].astype(np.float64),
            whitening=global_data["W_temp"].astype(np.float64),
            calibration_raw=global_temporal_reference,
        ),
        "patch_spatial": StableGaussianParams(
            mean=local_data["mu_patch_spat"].astype(np.float64),
            whitening=local_data["W_patch_spat"].astype(np.float64),
            calibration_raw=placeholder,
        ),
        "patch_d2": StableGaussianParams(
            mean=local_data["mu_patch_temp"].astype(np.float64),
            whitening=local_data["W_patch_temp"].astype(np.float64),
            calibration_raw=placeholder,
        ),
    }


def decode_row(
    row: pd.Series,
    windows: list[list[int]],
    seek_gap: int,
    attempts: int,
) -> dict:
    if len(windows) != int(row["effective_k"]):
        raise ValueError(f"effective_k mismatch for {row['video_id']}")
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
                "unique_indices": unique_indices,
            }
        except Exception as caught:
            error = caught
            if attempt + 1 < attempts:
                time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"decode failed after {attempts} attempts: {error}") from error


@torch.inference_mode()
def score_batch(
    decoded: list[dict],
    extractor: PatchSTALL,
    params: dict[str, StableGaussianParams],
    score_device: str,
    frame_batch_size: int,
) -> list[dict]:
    # Keep DINO's frame grouping independent of the outer video/I/O batch.
    # Concatenating several videos changes the final frame-batch shape and can
    # perturb patch tokens at roughly 1e-5 even though the model is in eval
    # mode. Each video therefore has its own fixed extraction call.
    extracted = [
        extractor.frames_to_global_patch_embeddings(
            [item["frames"]], batch_size=frame_batch_size
        )[0]
        for item in decoded
    ]
    global_windows = []
    patch_windows = []
    metadata = []
    for item, output in zip(decoded, extracted):
        if tuple(output["grid_size"]) != (14, 14):
            raise ValueError(f"unexpected patch grid: {output['grid_size']}")
        for window_id, positions in enumerate(item["positions"]):
            global_windows.append(output["global"][positions])
            patch_windows.append(output["patch"][positions])
            metadata.append((item, window_id))
    global_batch = torch.from_numpy(np.stack(global_windows).astype(np.float32))
    patch_batch = torch.from_numpy(np.stack(patch_windows).astype(np.float32))

    global_spatial_raw, _ = score_gaussian_aggregate_float64(
        global_batch,
        params["global_spatial"],
        aggregation="max",
        device=score_device,
        compute_percentile=False,
    )
    global_delta, zero_mask = l2_normalized_first_order(global_batch)
    global_t1_raw, _ = score_gaussian_aggregate_float64(
        global_delta,
        params["global_t1"],
        aggregation="min",
        device=score_device,
        invalid_mask=zero_mask,
        compute_percentile=False,
    )
    patch_spatial_raw, _ = score_gaussian_aggregate_float64(
        patch_batch,
        params["patch_spatial"],
        aggregation="mean",
        device=score_device,
        compute_percentile=False,
    )
    patch_d2 = l2_normalized_second_order(patch_batch)
    patch_d2_raw, _ = score_gaussian_aggregate_float64(
        patch_d2,
        params["patch_d2"],
        aggregation="mean",
        device=score_device,
        compute_percentile=False,
    )

    rows = []
    for index, (item, window_id) in enumerate(metadata):
        source = item["row"]
        rows.append(
            {
                **{column: source[column] for column in KEY_COLUMNS},
                "video_path": source["video_path"],
                "duration_seconds": float(source["duration_seconds"]),
                "effective_k": int(source["effective_k"]),
                "unique_frame_count": len(item["unique_indices"]),
                "window_id": window_id,
                "frame_indices": json.dumps(
                    item["windows"][window_id], separators=(",", ":")
                ),
                "global_spatial_raw": global_spatial_raw[index],
                "global_t1_raw": global_t1_raw[index],
                "patch_spatial_raw": patch_spatial_raw[index],
                "patch_d2_raw": patch_d2_raw[index],
            }
        )
    return rows


def completed_videos(checkpoint_dir: Path) -> tuple[set[str], list[Path]]:
    parts = sorted(checkpoint_dir.glob("part_*.csv"))
    completed: set[str] = set()
    for part in parts:
        completed.update(pd.read_csv(part, usecols=["video_id"])["video_id"].unique())
    return completed, parts


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
    merged = pd.concat(
        [pd.read_csv(path, float_precision="round_trip") for path in parts],
        ignore_index=True,
    )
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
