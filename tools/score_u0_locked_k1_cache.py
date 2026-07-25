#!/usr/bin/env python3
"""Score unified U0 K1 evaluation windows from locked patch-token caches."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from analyze_u0_locked import (
    cdf_with_positive_infinity,
    global_references,
    load_calibration_references,
)
from build_u0_release_manifests import video_id
from score_u0_locked_windows import load_raw_params, stable_shard
from stable_whitening import (
    empirical_cdf_right_inclusive,
    l2_normalized_first_order,
    l2_normalized_second_order,
    score_gaussian_aggregate_float64,
    stable_sorted,
)


CACHE_ROOTS = {
    "comgenvid": ROOT / "cache/patch_embeddings/comgenvid",
    "videofeedback": ROOT / "cache/patch_embeddings/videofeedback",
    "genvideo": ROOT / "cache/patch_embeddings/genvideo",
}
KEY_COLUMNS = [
    "video_id",
    "dataset",
    "protocol_split",
    "subset",
    "source_model",
    "filename",
]


def cache_path(row: pd.Series) -> Path:
    return (
        CACHE_ROOTS[str(row["dataset"])]
        / str(row["subset"])
        / str(row["source_model"])
        / f"{Path(str(row['filename'])).stem}_2s.pt"
    )


def load_manifest(source_path: Path, dataset: str, num_shards: int, shard: int) -> pd.DataFrame:
    source = pd.read_csv(source_path, float_precision="round_trip")
    source = source[
        (source["dataset"] == dataset) & (source["protocol_split"] == "evaluation")
    ].copy()
    source["video_id"] = source.apply(video_id, axis=1)
    source = source[
        source["video_id"].map(lambda value: stable_shard(value, num_shards) == shard)
    ].reset_index(drop=True)
    return source


def load_batch(rows: list[pd.Series]) -> tuple[torch.Tensor, torch.Tensor]:
    global_features = []
    patch_features = []
    for row in rows:
        path = cache_path(row)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, weights_only=True, map_location="cpu")
        expected = json.loads(str(row["indices_K1_current"]))
        if len(expected) != 1 or [int(value) for value in payload["frame_indices"]] != [
            int(value) for value in expected[0]
        ]:
            raise ValueError(f"K1 cache frame indices differ from lock: {path}")
        if tuple(payload["global"].shape) != (16, 1024):
            raise ValueError(f"invalid global cache shape: {path}")
        if tuple(payload["patch"].shape) != (16, 196, 1024):
            raise ValueError(f"invalid patch cache shape: {path}")
        if tuple(int(value) for value in payload["grid_size"]) != (14, 14):
            raise ValueError(f"invalid patch grid: {path}")
        global_features.append(payload["global"].float())
        patch_features.append(payload["patch"].float())
    return torch.stack(global_features), torch.stack(patch_features)


def score_raw_batch(
    global_batch: torch.Tensor,
    patch_batch: torch.Tensor,
    params: dict,
    device: str,
) -> dict[str, np.ndarray]:
    global_spatial, _ = score_gaussian_aggregate_float64(
        global_batch,
        params["global_spatial"],
        aggregation="max",
        device=device,
        compute_percentile=False,
    )
    global_delta, zero = l2_normalized_first_order(global_batch)
    global_t1, _ = score_gaussian_aggregate_float64(
        global_delta,
        params["global_t1"],
        aggregation="min",
        device=device,
        invalid_mask=zero,
        compute_percentile=False,
    )
    patch_spatial, _ = score_gaussian_aggregate_float64(
        patch_batch,
        params["patch_spatial"],
        aggregation="mean",
        device=device,
        compute_percentile=False,
    )
    patch_d2, _ = score_gaussian_aggregate_float64(
        l2_normalized_second_order(patch_batch),
        params["patch_d2"],
        aggregation="mean",
        device=device,
        compute_percentile=False,
    )
    return {
        "global_spatial_raw": global_spatial,
        "global_t1_raw": global_t1,
        "patch_spatial_raw": patch_spatial,
        "patch_d2_raw": patch_d2,
    }


def calibration_raw_references(
    calibration_raw_dir: Path, num_shards: int, dataset: str, config: dict
) -> dict[str, np.ndarray]:
    calibration = load_calibration_references(calibration_raw_dir, num_shards)
    calibration = calibration[calibration["dataset"] == dataset]
    if len(calibration) != 200:
        raise ValueError(f"{dataset}: expected 200 K1 calibration references")
    global_spatial, global_t1 = global_references(config)
    return {
        "global_spatial": global_spatial,
        "global_t1": global_t1,
        "patch_spatial": stable_sorted(calibration["patch_spatial_raw"].to_numpy()),
        "patch_d2": stable_sorted(calibration["patch_d2_raw"].to_numpy()),
    }


def calibrate_raw(raw: dict[str, np.ndarray], references: dict[str, np.ndarray]) -> dict:
    global_spatial = empirical_cdf_right_inclusive(
        raw["global_spatial_raw"], references["global_spatial"]
    )
    global_t1 = cdf_with_positive_infinity(
        raw["global_t1_raw"], references["global_t1"]
    )
    patch_spatial = empirical_cdf_right_inclusive(
        raw["patch_spatial_raw"], references["patch_spatial"]
    )
    patch_d2 = empirical_cdf_right_inclusive(
        raw["patch_d2_raw"], references["patch_d2"]
    )
    return {
        "global_spatial": global_spatial,
        "global_t1": global_t1,
        "patch_spatial": patch_spatial,
        "patch_d2": patch_d2,
        "G_k": 0.5 * global_spatial + 0.5 * global_t1,
        "L_k": 0.1 * patch_spatial + 0.9 * patch_d2,
    }


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = load_manifest(
        args.source_manifest, args.dataset, args.num_shards, args.shard_index
    )
    params = load_raw_params(config, args.dataset)
    references = calibration_raw_references(
        args.calibration_raw_dir, args.num_shards, args.dataset, config
    )
    rows = []
    started = time.perf_counter()
    for start in range(0, len(manifest), args.batch_size):
        source_rows = [
            row for _, row in manifest.iloc[start : start + args.batch_size].iterrows()
        ]
        global_batch, patch_batch = load_batch(source_rows)
        raw = score_raw_batch(global_batch, patch_batch, params, args.device)
        calibrated = calibrate_raw(raw, references)
        for index, source in enumerate(source_rows):
            rows.append(
                {
                    **{column: source[column] for column in KEY_COLUMNS},
                    "video_path": source["video_path"],
                    "duration_seconds": float(source["duration_seconds"]),
                    "frame_indices": json.dumps(
                        json.loads(str(source["indices_K1_current"]))[0],
                        separators=(",", ":"),
                    ),
                    **{key: value[index] for key, value in raw.items()},
                    **{key: value[index] for key, value in calibrated.items()},
                }
            )
        completed = min(start + args.batch_size, len(manifest))
        if start == 0 or completed == len(manifest) or completed % 200 == 0:
            print(
                f"{args.dataset} shard={args.shard_index} "
                f"processed={completed}/{len(manifest)} "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )
    output = (
        args.output_dir
        / "k1_raw"
        / f"{args.dataset}_shard{args.shard_index:02d}_of_{args.num_shards:02d}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(KEY_COLUMNS)
    if len(frame) != len(manifest) or frame["video_id"].duplicated().any():
        raise ValueError("K1 cache score output is incomplete or duplicated")
    temporary = output.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False)
    temporary.replace(output)
    print(f"wrote {len(frame)} K1 cache scores -> {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("comgenvid", "videofeedback", "genvideo"), required=True
    )
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/alpha_stalled_u0_locked.yaml"
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "results/multi_window_joint_typicality/multi_window_manifest.csv",
    )
    parser.add_argument(
        "--calibration-raw-dir",
        type=Path,
        default=ROOT / "results/u0_locked_reproduction/calibration_raw",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/u0_core_ablation"
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
