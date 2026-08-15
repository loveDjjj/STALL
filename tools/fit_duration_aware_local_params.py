#!/usr/bin/env python3
"""Fit Local region1/mean parameters for duration-aware calibration banks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from create_patch_params import (
    _fit_whitening,
    _get_mu_W,
    collect_fit_samples,
)
from alpha_stalled.duration_aware_protocol import CALIBRATION_SIZES


INDEX_ROOT = ROOT / "results/duration_aware_23source/calibration_indexes"
JOBS = tuple(
    (dataset, duration, size, INDEX_ROOT / f"{dataset}_n{size}.csv")
    for dataset, sizes in CALIBRATION_SIZES.items()
    for duration in ((2,) if dataset == "comgenvid" else (1, 2))
    for size in sizes
    if not (duration == 2 and size == 200)
)


def selected_jobs(num_shards: int, shard_index: int) -> list[tuple[str, int, int, Path]]:
    return [job for index, job in enumerate(JOBS) if index % num_shards == shard_index]


def fit_whitening_only(dataset: str, duration: int, index_path: Path) -> dict:
    patch_fit, temporal_fit, grid_size = collect_fit_samples(
        csv_path=str(index_path),
        patch_emb_cache_dir=str(ROOT / "cache/patch_embeddings" / dataset),
        duration_sec=duration,
        compact=True,
        max_patches_for_fit=300_000,
        seed=42,
        patch_temp_mode="same_grid_second_order",
        match_radius=2,
        top_m=4,
        temperature=0.07,
        lambda_dist=0.01,
        patch_region_size=1,
        max_real_videos=None,
    )
    print(f"fit spatial whitening from {len(patch_fit)} patch tokens", flush=True)
    spatial = _fit_whitening(patch_fit)
    mu_spatial, whitening_spatial = _get_mu_W(spatial)
    print(f"fit temporal whitening from {len(temporal_fit)} D2 tokens", flush=True)
    temporal = _fit_whitening(temporal_fit)
    mu_temporal, whitening_temporal = _get_mu_W(temporal)
    # The formal scorer rebuilds every window/video CDF from its locked raw
    # calibration traversal. Stored calibration arrays are therefore explicit
    # placeholders, not references used at inference.
    placeholder = np.array([0.0], dtype=np.float32)
    return {
        "mu_patch_spat": mu_spatial.astype(np.float32),
        "W_patch_spat": whitening_spatial.astype(np.float32),
        "calib_patch_spat_scores": placeholder,
        "mu_patch_temp": mu_temporal.astype(np.float32),
        "W_patch_temp": whitening_temporal.astype(np.float32),
        "calib_patch_temp_scores": placeholder,
        "patch_grid_size": np.asarray(grid_size, dtype=np.int32),
        "duration": np.asarray([duration], dtype=np.int32),
        "aggregation_config": np.array("{}"),
    }


def run(args: argparse.Namespace) -> None:
    for dataset, duration, size, index_path in selected_jobs(args.num_shards, args.shard_index):
        output = args.output_dir / f"{dataset}_{duration}s_n{size}.npz"
        if output.is_file() and not args.force:
            print(f"reuse {output}", flush=True)
            continue
        started = time.perf_counter()
        params = fit_whitening_only(dataset, duration, index_path)
        config = json.loads(str(params["aggregation_config"]))
        config.update(
            {
                "protocol": "duration_aware_23source_v1",
                "dataset": dataset,
                "duration_sec": duration,
                "calibration_videos": size,
                "calibration_index": str(index_path.relative_to(ROOT)),
                "generated_videos": 0,
                "stored_calibration_arrays": "placeholder_not_used",
                "cdf_reference": "rebuilt_from_locked_raw_calibration_tasks",
                "fit_elapsed_seconds": time.perf_counter() - started,
            }
        )
        params["aggregation_config"] = np.array(json.dumps(config, sort_keys=True))
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp.npz")
        np.savez(temporary, **params)
        temporary.replace(output)
        print(f"wrote {output} elapsed={config['fit_elapsed_seconds']:.1f}s", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/duration_aware_23source/params",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require num_shards >= 1 and valid shard-index")
    return args


if __name__ == "__main__":
    run(parse_args())
