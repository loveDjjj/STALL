#!/usr/bin/env python3
"""Recalibrate patch params with a new aggregation using fast torch batches.

This reuses whitening matrices from an existing params file and only recomputes
per-real-video calibration scores. It is useful when exploring aggregation
variants that should not refit the underlying real patch distribution.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval_patch_fast import FastPatchScorer, iter_cache_jobs, load_cache_batch  # noqa: E402


def _load_config(value) -> dict:
    if isinstance(value, np.ndarray):
        value = value.item()
    return json.loads(str(value))


def _real_jobs(csv_path: str, patch_cache: str, duration: int, compact: bool) -> list[dict]:
    return [
        job
        for job in iter_cache_jobs(csv_path, patch_cache, duration, compact, debug_n=None)
        if str(job["subset"]).lower() == "real"
    ]


@torch.inference_mode()
def _raw_scores_for_batch(
    scorer: FastPatchScorer,
    patch_batch: torch.Tensor,
    device: torch.device,
    patch_temp_mode: str,
    aggregation: str,
    bottomk_ratio: float,
    temporal_run_length: int,
    patch_region_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    patch = patch_batch.float().to(device, non_blocking=True)
    mu_spat, W_spat, mu_temp, W_temp = scorer._params_for_device(device)

    spat_white = torch.matmul(patch - mu_spat, W_spat)
    spat_ll = scorer.log_likelihood_from_white(spat_white)
    spat_agg = scorer._aggregate(spat_ll, aggregation, bottomk_ratio, temporal_run_length)

    temp = scorer.temporal_features(patch, patch_temp_mode, patch_region_size)
    temp_white = torch.matmul(temp - mu_temp, W_temp)
    temp_ll = scorer.log_likelihood_from_white(temp_white)
    temp_agg = scorer._aggregate(temp_ll, aggregation, bottomk_ratio, temporal_run_length)

    return (
        spat_agg.detach().cpu().numpy().astype(np.float32),
        temp_agg.detach().cpu().numpy().astype(np.float32),
    )


def compute_calibration(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    scorer = FastPatchScorer(args.base_params, device=args.score_device)
    scorer.validate(args.patch_temp_mode)
    scorer.params_aggregation_region_size = args.aggregation_region_size
    if scorer.params_patch_region_size != args.patch_region_size:
        raise ValueError(
            f"Base params expect patch_region_size={scorer.params_patch_region_size}, "
            f"got {args.patch_region_size}"
        )

    jobs = _real_jobs(args.csv, args.patch_emb_cache, args.duration, args.compact)
    if args.max_real_videos is not None:
        jobs = jobs[: args.max_real_videos]
    if not jobs:
        raise ValueError("No real patch-cache jobs found")

    print(
        f"Fast recalibration: real_videos={len(jobs)}, devices={','.join(str(d) for d in scorer.devices)}, "
        f"batch={args.score_batch_size}, aggregation={args.aggregation}, bottomk={args.bottomk_ratio}, "
        f"run_length={args.temporal_run_length}, agg_region={scorer.params_aggregation_region_size}",
        flush=True,
    )

    spat_scores = []
    temp_scores = []
    for start in tqdm(range(0, len(jobs), args.score_batch_size), desc="Recalibrating", unit="batch"):
        batch_jobs = jobs[start : start + args.score_batch_size]
        patch_batch, grid_size = load_cache_batch(batch_jobs)
        if grid_size != scorer.patch_grid_size:
            raise ValueError(f"Params grid_size={scorer.patch_grid_size}, cache grid_size={grid_size}")

        if len(scorer.devices) > 1 and patch_batch.shape[0] >= len(scorer.devices):
            chunks = torch.tensor_split(patch_batch, len(scorer.devices), dim=0)
            parts = []
            for chunk, device in zip(chunks, scorer.devices):
                if chunk.shape[0] == 0:
                    continue
                parts.append(
                    _raw_scores_for_batch(
                        scorer,
                        chunk,
                        device,
                        args.patch_temp_mode,
                        args.aggregation,
                        args.bottomk_ratio,
                        args.temporal_run_length,
                        args.patch_region_size,
                    )
                )
            spat_scores.append(np.concatenate([part[0] for part in parts], axis=0))
            temp_scores.append(np.concatenate([part[1] for part in parts], axis=0))
        else:
            spat, temp = _raw_scores_for_batch(
                scorer,
                patch_batch,
                scorer.device,
                args.patch_temp_mode,
                args.aggregation,
                args.bottomk_ratio,
                args.temporal_run_length,
                args.patch_region_size,
            )
            spat_scores.append(spat)
            temp_scores.append(temp)

    return np.concatenate(spat_scores, axis=0), np.concatenate(temp_scores, axis=0)


def write_params(args: argparse.Namespace, calib_spat: np.ndarray, calib_temp: np.ndarray) -> None:
    base = np.load(args.base_params, allow_pickle=True)
    config = _load_config(base["aggregation_config"])
    config.update(
        {
            "mode": args.aggregation,
            "bottomk_ratio": args.bottomk_ratio,
            "temporal_run_length": args.temporal_run_length,
            "aggregation_region_size": args.aggregation_region_size,
            "implementation": "fast_recalibrated_from_base_whitening",
            "base_params": args.base_params,
            "max_real_videos": args.max_real_videos,
        }
    )

    out = {
        "mu_patch_spat": base["mu_patch_spat"].astype(np.float32),
        "W_patch_spat": base["W_patch_spat"].astype(np.float32),
        "calib_patch_spat_scores": calib_spat.astype(np.float32),
        "mu_patch_temp": base["mu_patch_temp"].astype(np.float32),
        "W_patch_temp": base["W_patch_temp"].astype(np.float32),
        "calib_patch_temp_scores": calib_temp.astype(np.float32),
        "patch_grid_size": base["patch_grid_size"].astype(np.int32),
        "duration": base["duration"].astype(np.int32),
        "aggregation_config": np.array(json.dumps(config)),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    print(f"Saved recalibrated params: {path}", flush=True)
    print(f"  calib_patch_spat_scores: {out['calib_patch_spat_scores'].shape}", flush=True)
    print(f"  calib_patch_temp_scores: {out['calib_patch_temp_scores'].shape}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast recalibrate patch params for a new aggregation.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--patch-emb-cache", required=True)
    parser.add_argument("--base-params", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration", type=int, default=2)
    parser.add_argument("--compact", action="store_true", default=False)
    parser.add_argument("--score-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--score-batch-size", type=int, default=64)
    parser.add_argument("--max-real-videos", type=int, default=None)
    parser.add_argument("--patch-temp-mode", required=True)
    parser.add_argument(
        "--aggregation",
        choices=[
            "mean",
            "min",
            "bottomk_mean",
            "temporal_run_bottomk_mean",
            "spatiotemporal_run_bottomk_mean",
        ],
        required=True,
    )
    parser.add_argument("--bottomk-ratio", type=float, required=True)
    parser.add_argument("--temporal-run-length", type=int, default=3)
    parser.add_argument("--aggregation-region-size", type=int, default=3)
    parser.add_argument("--patch-region-size", type=int, default=1)
    args = parser.parse_args()

    # FastPatchScorer reads aggregation_region_size from params, so for this
    # exploratory recalibration we temporarily write the desired value into the
    # scorer after construction via the saved config path in write_params.
    calib_spat, calib_temp = compute_calibration(args)
    write_params(args, calib_spat, calib_temp)


if __name__ == "__main__":
    main()
