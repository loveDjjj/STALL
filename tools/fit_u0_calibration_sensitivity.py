#!/usr/bin/env python3
"""Fit U0 Local parameters for independent calibration size/seed splits."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]

from build_u0_calibration_reserve import DATASETS, SEEDS, SIZES
from create_patch_params import build_patch_params


def jobs(num_shards: int, shard_index: int) -> list[tuple[str, int, int]]:
    all_jobs = [
        (dataset, seed, size)
        for dataset in DATASETS
        for seed in SEEDS
        for size in SIZES
    ]
    return [job for index, job in enumerate(all_jobs) if index % num_shards == shard_index]


def validate_and_write_index(
    dataset: str,
    seed: int,
    size: int,
    reserve: dict[str, dict],
    membership: pd.DataFrame,
    output: Path,
) -> None:
    index_path, _ = DATASETS[dataset]
    source = pd.read_csv(index_path, float_precision="round_trip")
    source["filename"] = source["video_path"].map(lambda value: Path(str(value)).name)
    selected_ids = set(
        membership[
            membership["dataset"].eq(dataset)
            & membership["seed"].eq(seed)
            & membership["calibration_size"].eq(size)
        ]["video_id"]
    )
    selected = [reserve[video_id] for video_id in sorted(selected_ids)]
    keys = {(item["source_model"], item["filename"]): item for item in selected}
    frame = source[
        source.apply(
            lambda row: (str(row["source_model"]), str(row["filename"])) in keys,
            axis=1,
        )
    ].copy()
    if len(frame) != size:
        raise ValueError(f"{dataset}/{seed}/{size}: expected {size} index rows, got {len(frame)}")
    for row in frame.itertuples(index=False):
        item = keys[(str(row.source_model), str(row.filename))]
        cache = ROOT / item["cache_path"]
        payload = torch.load(cache, weights_only=True, map_location="cpu")
        actual = [int(value) for value in payload["frame_indices"]]
        expected = [int(value) for value in item["k1_window"]]
        if actual != expected:
            raise ValueError(f"K1 cache/index mismatch: {cache}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.csv")
    frame.sort_values(["source_model", "filename"]).to_csv(temporary, index=False)
    temporary.replace(output)


def run(args: argparse.Namespace) -> None:
    payload = json.loads(args.reserve_manifest.read_text(encoding="utf-8"))
    reserve = {item["video_id"]: item for item in payload["videos"]}
    membership = pd.read_csv(args.membership)
    for dataset, seed, size in jobs(args.num_shards, args.shard_index):
        output = args.output_dir / "params" / f"{dataset}_seed{seed}_n{size}.npz"
        if output.is_file() and not args.force:
            print(f"reuse {output}", flush=True)
            continue
        index = args.output_dir / "indexes" / f"{dataset}_seed{seed}_n{size}.csv"
        validate_and_write_index(dataset, seed, size, reserve, membership, index)
        _, cache_root = DATASETS[dataset]
        started = time.perf_counter()
        params = build_patch_params(
            csv_path=str(index),
            patch_emb_cache_dir=str(cache_root),
            duration_sec=2,
            compact=True,
            max_patches_for_fit=300_000,
            aggregation="mean",
            bottomk_ratio=0.2,
            temporal_run_length=3,
            aggregation_region_size=3,
            seed=seed,
            patch_temp_mode="same_grid_second_order",
            match_radius=2,
            top_m=4,
            temperature=0.07,
            lambda_dist=0.01,
            patch_region_size=1,
            max_real_videos=None,
        )
        config = json.loads(str(params["aggregation_config"]))
        config.update(
            {
                "calibration_dataset": dataset,
                "calibration_seed": seed,
                "calibration_videos": size,
                "independent_reserve": True,
                "fit_elapsed_seconds": time.perf_counter() - started,
            }
        )
        params["aggregation_config"] = np.array(json.dumps(config, sort_keys=True))
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp.npz")
        np.savez(temporary, **params)
        temporary.replace(output)
        print(
            f"wrote {output} elapsed={config['fit_elapsed_seconds']:.1f}s",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reserve-manifest",
        type=Path,
        default=ROOT / "release/u0/calibration_reserve_manifest.json",
    )
    parser.add_argument(
        "--membership",
        type=Path,
        default=ROOT / "release/u0/calibration_split_membership.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/u0_calibration_sensitivity",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be in [0, num-shards)")
    return args


if __name__ == "__main__":
    run(parse_args())
