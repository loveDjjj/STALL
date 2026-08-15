#!/usr/bin/env python3
"""Score nested calibration-size candidates on the real-only reserve.

The output is used only to choose a dataset-specific calibration size. It reads
the compact real-video patch caches and never touches generated videos.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.duration_aware_protocol import (
    CALIBRATION_SIZES,
    DATASETS,
    add_identity,
)
from dataset_utils_patch import _get_patch_cache_path
from alpha_stalled.local_branch import local_d2_features
from alpha_stalled.whitening import (
    GaussianMeanCandidateScorerFloat64,
    StableGaussianParams,
)


def parameter_path(config: dict, params_dir: Path, dataset: str, duration: int, size: int) -> Path:
    if duration == 2 and size == 200:
        return ROOT / config["local_branch"]["params_by_dataset"][dataset]["path"]
    return params_dir / f"{dataset}_{duration}s_n{size}.npz"


def load_scorers(
    config: dict,
    params_dir: Path,
    dataset: str,
    duration: int,
    device: str,
) -> tuple[tuple[int, ...], GaussianMeanCandidateScorerFloat64, GaussianMeanCandidateScorerFloat64]:
    sizes = CALIBRATION_SIZES[dataset]
    spatial = []
    temporal = []
    for size in sizes:
        path = parameter_path(config, params_dir, dataset, duration, size)
        if not path.is_file():
            raise FileNotFoundError(path)
        spatial.append(
            StableGaussianParams.from_npz(
                str(path), "mu_patch_spat", "W_patch_spat", "calib_patch_spat_scores"
            )
        )
        temporal.append(
            StableGaussianParams.from_npz(
                str(path), "mu_patch_temp", "W_patch_temp", "calib_patch_temp_scores"
            )
        )
    return (
        sizes,
        GaussianMeanCandidateScorerFloat64(spatial, spatial[0].mean, device=device),
        GaussianMeanCandidateScorerFloat64(temporal, temporal[0].mean, device=device),
    )


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    spec = DATASETS[args.dataset]
    maximum = add_identity(
        pd.read_csv(spec["calibration"], float_precision="round_trip"), args.dataset
    )
    durations = (2,) if args.dataset == "comgenvid" else (1, 2)
    output_rows = []
    for duration in durations:
        sizes, spatial_scorer, temporal_scorer = load_scorers(
            config, args.params_dir, args.dataset, duration, args.device
        )
        for index, row in maximum.iterrows():
            cache_path = _get_patch_cache_path(
                ROOT / "cache/patch_embeddings" / args.dataset,
                str(row["subset"]),
                str(row["source_model"]),
                Path(str(row["video_path"])).stem,
                duration,
                True,
            )
            if not cache_path.is_file():
                raise FileNotFoundError(cache_path)
            payload = torch.load(cache_path, weights_only=True, map_location="cpu")
            patch = payload["patch"].to(dtype=torch.float32).unsqueeze(0)
            expected_frames = 8 * duration
            if tuple(patch.shape[1:3]) != (expected_frames, 196):
                raise ValueError(f"unexpected patch shape {tuple(patch.shape)}: {cache_path}")
            spatial = spatial_scorer.score(patch)[0]
            temporal = temporal_scorer.score(local_d2_features(patch))[0]
            for position, size in enumerate(sizes):
                output_rows.append(
                    {
                        "dataset": args.dataset,
                        "protocol_duration_sec": duration,
                        "video_id": row["video_id"],
                        "source_model": row["source_model"],
                        "calibration_size": size,
                        "patch_spatial_raw": spatial[position],
                        "patch_d2_raw": temporal[position],
                    }
                )
            if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(maximum):
                print(
                    f"{args.dataset}/{duration}s {index + 1}/{len(maximum)}",
                    flush=True,
                )
    output = pd.DataFrame(output_rows)
    expected = len(maximum) * len(CALIBRATION_SIZES[args.dataset]) * len(durations)
    if len(output) != expected or output.duplicated(
        ["video_id", "protocol_duration_sec", "calibration_size"]
    ).any():
        raise ValueError(f"invalid output rows={len(output)} expected={expected}")
    if not np.isfinite(output[["patch_spatial_raw", "patch_d2_raw"]].to_numpy()).all():
        raise ValueError("non-finite real-only scores")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{args.dataset}.csv"
    temporary = path.with_suffix(".tmp.csv")
    output.to_csv(temporary, index=False)
    temporary.replace(path)
    print(f"wrote {len(output)} rows -> {path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/alpha_stalled_u0_locked.yaml"
    )
    parser.add_argument(
        "--params-dir",
        type=Path,
        default=ROOT / "results/duration_aware_23source/params",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/duration_aware_23source/real_only_curve/raw",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
