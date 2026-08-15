#!/usr/bin/env python3
"""Score K=3 spatial-mean residual D2 and common-motion diagnostics."""

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


REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT / "src", REPO_ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from eval_patch_fast import FastPatchScorer
from alpha_stalled.legacy_window_scoring import (
    KEY_COLUMNS,
    decode_manifest_row_with_retries,
    load_completed,
    load_windows,
    stable_shard,
    video_key,
)
from stall_patch import PatchSTALL


SAMPLING = "K3_uniform"


def _midpoint_median(values: torch.Tensor, dim: int) -> torch.Tensor:
    ordered = torch.sort(values, dim=dim).values
    middle = ordered.shape[dim] // 2
    if ordered.shape[dim] % 2:
        return ordered.select(dim, middle)
    return 0.5 * (ordered.select(dim, middle - 1) + ordered.select(dim, middle))


def common_motion_diagnostics(patch: torch.Tensor) -> dict[str, np.ndarray]:
    """Compute diagnostics on unpooled [N,16,196,D] final-layer patch tokens."""
    acceleration = patch[:, 2:] - 2.0 * patch[:, 1:-1] + patch[:, :-2]
    mean_common = acceleration.mean(dim=2)
    median_common = _midpoint_median(acceleration, dim=2)
    total_energy = acceleration.square().sum(dim=-1).mean(dim=2).sum(dim=1)
    eps = torch.finfo(acceleration.dtype).eps
    rho_mean = mean_common.square().sum(dim=-1).sum(dim=1) / (total_energy + eps)
    rho_median = median_common.square().sum(dim=-1).sum(dim=1) / (total_energy + eps)
    return {
        "rho_mean": rho_mean.cpu().numpy(),
        "rho_median": rho_median.cpu().numpy(),
        "mean_motion_magnitude": acceleration.norm(dim=-1).mean(dim=(1, 2)).cpu().numpy(),
        "mean_common_magnitude": mean_common.norm(dim=-1).mean(dim=1).cpu().numpy(),
        "median_common_magnitude": median_common.norm(dim=-1).mean(dim=1).cpu().numpy(),
    }


@torch.inference_mode()
def score_patch_windows(
    patch_windows: np.ndarray,
    scorer: FastPatchScorer,
    batch_size: int,
) -> dict[str, np.ndarray]:
    outputs: dict[str, list[np.ndarray]] = {
        "residual_raw_likelihood": [],
        "residual_d2": [],
        "rho_mean": [],
        "rho_median": [],
        "mean_motion_magnitude": [],
        "mean_common_magnitude": [],
        "median_common_magnitude": [],
    }
    device = scorer.device
    _, _, mu_temp, W_temp = scorer._params_for_device(device)
    aggregation = scorer.aggregation_config.get("mode", "bottomk_mean")
    for start in range(0, len(patch_windows), batch_size):
        patch = torch.as_tensor(
            patch_windows[start : start + batch_size], dtype=torch.float32, device=device
        )
        diagnostics = common_motion_diagnostics(patch)
        temporal = scorer.temporal_features(
            patch,
            "spatial_mean_residual_second_order",
            scorer.params_patch_region_size,
        )
        white = torch.matmul(temporal - mu_temp, W_temp)
        likelihood = scorer.log_likelihood_from_white(white)
        raw = scorer._aggregate(
            likelihood,
            aggregation,
            scorer.params_bottomk_ratio,
            scorer.params_temporal_run_length,
        ).cpu().numpy()
        percentile = scorer.percentile(raw, scorer.calib_temp)
        outputs["residual_raw_likelihood"].append(raw)
        outputs["residual_d2"].append(percentile)
        for name, values in diagnostics.items():
            outputs[name].append(values)
    return {name: np.concatenate(parts).astype(np.float32) for name, parts in outputs.items()}


def score_decoded(
    decoded: list[dict],
    extractor: PatchSTALL,
    scorer: FastPatchScorer,
    frame_batch_size: int,
    score_batch_size: int,
) -> list[dict]:
    embeddings = extractor.frames_to_global_patch_embeddings(
        [item["frames"] for item in decoded], batch_size=frame_batch_size
    )
    patches: list[np.ndarray] = []
    metadata: list[tuple[dict, int]] = []
    for item, embedding in zip(decoded, embeddings):
        for window_id, positions in enumerate(item["window_positions"]):
            patches.append(embedding["patch"][positions])
            metadata.append((item, window_id))
    scores = score_patch_windows(np.stack(patches), scorer, score_batch_size)
    rows: list[dict] = []
    for index, (item, window_id) in enumerate(metadata):
        source = item["row"]
        rows.append(
            {
                **{column: source[column] for column in KEY_COLUMNS},
                "video_path": source["video_path"],
                "duration_seconds": source["duration_seconds"],
                "effective_k": len(item["windows"]),
                "window_id": window_id,
                "frame_indices": json.dumps(item["windows"][window_id], separators=(",", ":")),
                **{name: float(values[index]) for name, values in scores.items()},
            }
        )
    return rows


def run(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest["dataset"] == args.dataset].copy()
    manifest["active_sampling"] = SAMPLING
    manifest = manifest[
        manifest.apply(lambda row: stable_shard(row, args.num_shards) == args.shard_index, axis=1)
    ].reset_index(drop=True)
    if args.debug_videos is not None:
        manifest = manifest.head(args.debug_videos).copy()
    checkpoint_dir = (
        args.checkpoint_dir
        / args.dataset
        / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    completed, parts = load_completed(checkpoint_dir)
    pending = manifest[
        ~manifest.apply(lambda row: video_key(row) in completed, axis=1)
    ].reset_index(drop=True)
    print(
        f"dataset={args.dataset} shard={args.shard_index}/{args.num_shards} "
        f"expected={len(manifest)} completed={len(completed)} pending={len(pending)}",
        flush=True,
    )
    scorer = FastPatchScorer(str(args.residual_params), device=args.device)
    scorer.validate("spatial_mean_residual_second_order")
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
                    SAMPLING,
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
            rows = score_decoded(
                decoded, extractor, scorer, args.frame_batch_size, args.score_batch_size
            )
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
                f"videos_per_sec={processed / max(elapsed, 1e-9):.3f}",
                flush=True,
            )
    if scorer._executor is not None:
        scorer._executor.shutdown(wait=True)
    if failures:
        pd.DataFrame(failures).to_csv(checkpoint_dir / "failures.csv", index=False)
    completed, parts = load_completed(checkpoint_dir)
    expected = {video_key(row) for _, row in manifest.iterrows()}
    if expected - completed:
        raise RuntimeError(f"incomplete residual shard: missing={len(expected - completed)}")
    merged = pd.concat(
        [pd.read_csv(part, float_precision="round_trip") for part in parts], ignore_index=True
    )
    merged = merged.drop_duplicates(KEY_COLUMNS + ["window_id"], keep="last")
    merged = merged.sort_values(KEY_COLUMNS + ["window_id"]).reset_index(drop=True)
    expected_windows = sum(len(load_windows(row, SAMPLING)) for _, row in manifest.iterrows())
    if len(merged) != expected_windows:
        raise RuntimeError(f"expected {expected_windows} windows, found {len(merged)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".tmp.csv")
    merged.to_csv(temp, index=False)
    temp.replace(args.output)
    print(f"saved {len(merged)} windows -> {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("comgenvid", "videofeedback", "genvideo"), required=True
    )
    parser.add_argument("--residual-params", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "results/multi_window_joint_typicality/multi_window_manifest.csv",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("/tmp/alpha_stalled_local_residual")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--video-batch-size", type=int, default=4)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--score-batch-size", type=int, default=4)
    parser.add_argument("--debug-videos", type=int)
    parser.add_argument("--decode-workers", type=int, default=4)
    parser.add_argument("--decode-attempts", type=int, default=3)
    parser.add_argument("--seek-gap", type=int, default=64)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require num_shards >= 1 and 0 <= shard_index < num_shards")
    return args


if __name__ == "__main__":
    run(parse_args())
