#!/usr/bin/env python3
"""Fit locked-U0 Local D1 parameters from the same 200 real calibration videos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from alpha_stalled.release_io import video_id
from alpha_stalled.local_branch import local_d1_features
from create_patch_params import build_patch_params
from alpha_stalled.whitening import (
    StableGaussianParams,
    score_gaussian_aggregate_float64,
)


DATASETS = {
    "comgenvid": {
        "index": ROOT / "cache/indexes/comgenvid_calib_real200.csv",
        "cache": ROOT / "cache/patch_embeddings/comgenvid",
    },
    "videofeedback": {
        "index": ROOT / "cache/indexes/videofeedback_small_calib_real200.csv",
        "cache": ROOT / "cache/patch_embeddings/videofeedback",
    },
    "genvideo": {
        "index": ROOT / "cache/indexes/genvideo_calib_real200.csv",
        "cache": ROOT / "cache/patch_embeddings/genvideo",
    },
}


def validate_index(dataset: str, index: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    frame = index.copy()
    frame["dataset"] = dataset
    frame["filename"] = frame["video_path"].map(lambda value: Path(str(value)).name)
    frame["video_id"] = frame.apply(video_id, axis=1)
    expected = {
        item["video_id"]
        for item in manifest["videos"]
        if item["dataset"] == dataset
    }
    actual = set(frame.loc[frame["subset"].eq("real"), "video_id"])
    if actual != expected or len(actual) != 200:
        raise ValueError(
            f"{dataset}: calibration index identities differ from locked 200-real manifest"
        )
    return frame


def cache_path(dataset: str, row: pd.Series) -> Path:
    return (
        DATASETS[dataset]["cache"]
        / str(row["subset"])
        / str(row["source_model"])
        / f"{Path(str(row['filename'])).stem}_2s.pt"
    )


def stable_k1_scores(
    dataset: str,
    index: pd.DataFrame,
    frame_indices: dict,
    params: StableGaussianParams,
    device: str,
) -> pd.DataFrame:
    rows = []
    real = index[index["subset"].eq("real")].sort_values("video_id")
    for position, (_, row) in enumerate(real.iterrows(), start=1):
        path = cache_path(dataset, row)
        payload = torch.load(path, weights_only=True, map_location="cpu")
        actual = [int(value) for value in payload["frame_indices"]]
        expected = [int(value) for value in frame_indices[row["video_id"]]]
        if actual != expected:
            raise ValueError(f"{path}: cache indices differ from locked K1 reference")
        patch = payload["patch"].float().unsqueeze(0)
        raw, _ = score_gaussian_aggregate_float64(
            local_d1_features(patch),
            params,
            aggregation="mean",
            device=device,
            compute_percentile=False,
        )
        rows.append(
            {
                "video_id": row["video_id"],
                "dataset": dataset,
                "protocol_split": "calibration",
                "subset": "real",
                "source_model": row["source_model"],
                "filename": row["filename"],
                "frame_indices": json.dumps(actual, separators=(",", ":")),
                "patch_d1_raw": float(raw[0]),
            }
        )
        if position % 50 == 0:
            print(f"[{dataset}] stable D1 K1 calibration={position}/200", flush=True)
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    spec = DATASETS[args.dataset]
    manifest = json.loads(args.calibration_manifest.read_text(encoding="utf-8"))
    frame_payload = json.loads(args.frame_indices.read_text(encoding="utf-8"))
    index = validate_index(
        args.dataset,
        pd.read_csv(spec["index"], float_precision="round_trip"),
        manifest,
    )
    candidate = build_patch_params(
        csv_path=str(spec["index"]),
        patch_emb_cache_dir=str(spec["cache"]),
        duration_sec=2,
        compact=True,
        max_patches_for_fit=300_000,
        aggregation="mean",
        bottomk_ratio=0.2,
        temporal_run_length=3,
        aggregation_region_size=3,
        seed=42,
        patch_temp_mode="same_grid_lag1",
        match_radius=2,
        top_m=4,
        temperature=0.07,
        lambda_dist=0.01,
        patch_region_size=1,
        max_real_videos=None,
    )
    locked_path = ROOT / "release/u0/params" / f"{args.dataset}_region1_mean.npz"
    locked = np.load(locked_path, allow_pickle=True)
    for key in ("mu_patch_spat", "W_patch_spat", "calib_patch_spat_scores"):
        candidate[key] = locked[key]
    config = json.loads(str(candidate["aggregation_config"].item()))
    config.update(
        {
            "calibration_videos": 200,
            "implementation": "u0_locked_d1_ablation_v1",
            "local_spatial_source": str(locked_path.relative_to(ROOT)),
            "temporal_formula": "P[t+1,i]-P[t,i]",
            "numerical_scoring": "float32_difference_float64_whitening_likelihood",
        }
    )
    candidate["aggregation_config"] = np.array(json.dumps(config, sort_keys=True))
    temporal_params = StableGaussianParams(
        mean=np.asarray(candidate["mu_patch_temp"], dtype=np.float64),
        whitening=np.asarray(candidate["W_patch_temp"], dtype=np.float64),
        calibration_raw=np.array([0.0], dtype=np.float64),
    )
    calibration = stable_k1_scores(
        args.dataset,
        index,
        frame_payload["calibration_reference_windows"],
        temporal_params,
        args.device,
    )
    candidate["calib_patch_temp_scores"] = calibration["patch_d1_raw"].to_numpy(
        dtype=np.float64
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.calibration_output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.dataset}_region1_mean_d1.npz"
    temporary = output.with_suffix(".tmp.npz")
    np.savez(temporary, **candidate)
    temporary.replace(output)
    calibration_path = args.calibration_output_dir / f"{args.dataset}.csv"
    temporary_csv = calibration_path.with_suffix(".tmp.csv")
    calibration.to_csv(temporary_csv, index=False)
    temporary_csv.replace(calibration_path)
    print(f"wrote {output} and {calibration_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=ROOT / "release/u0/calibration_manifest.json",
    )
    parser.add_argument(
        "--frame-indices", type=Path, default=ROOT / "release/u0/frame_indices.json"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/second_order_independent_calibration/params",
    )
    parser.add_argument(
        "--calibration-output-dir",
        type=Path,
        default=ROOT / "results/second_order_independent_calibration/calibration_k1",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
