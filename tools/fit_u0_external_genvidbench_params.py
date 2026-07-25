#!/usr/bin/env python3
"""Fit locked U0 Local parameters from GenVidBench calibration reals only."""

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

from build_u0_release_manifests import sha256_file, write_json
from create_patch_params import build_patch_params


DATASET = "genvidbench_pair1"


def cache_path(row: pd.Series, cache_root: Path) -> Path:
    return (
        cache_root
        / str(row["subset"])
        / str(row["source_model"])
        / f"{Path(str(row['filename'])).stem}_2s.pt"
    )


def validate_locked_cache(
    manifest_dir: Path, calibration_index: Path, cache_root: Path
) -> None:
    manifest = json.loads(
        (manifest_dir / "calibration_manifest.json").read_text(encoding="utf-8")
    )["videos"]
    frame_payload = json.loads(
        (manifest_dir / "frame_indices.json").read_text(encoding="utf-8")
    )
    references = frame_payload["calibration_reference_windows"]
    index = pd.read_csv(calibration_index, float_precision="round_trip")
    if len(manifest) != 199 or len(index) != 199:
        raise ValueError("external calibration must contain exactly 199 indexed real videos")
    if set(index["subset"]) != {"real"}:
        raise ValueError("external parameter fitting accepts real calibration videos only")
    by_name = {item["filename"]: item for item in manifest}
    if set(index["video_path"].map(lambda value: Path(str(value)).name)) != set(by_name):
        raise ValueError("external calibration index and locked manifest differ")
    for _, source in index.iterrows():
        filename = Path(str(source["video_path"])).name
        item = by_name[filename]
        path = cache_path(pd.Series(item), cache_root)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, weights_only=True, map_location="cpu")
        actual = [int(value) for value in payload["frame_indices"]]
        expected = [int(value) for value in references[item["video_id"]]]
        if actual != expected:
            raise ValueError(f"K1 cache differs from locked calibration window: {path}")
        if tuple(payload["patch"].shape) != (16, 196, 1024):
            raise ValueError(f"invalid external patch cache shape: {path}")


def run(args: argparse.Namespace) -> None:
    validate_locked_cache(args.release_dir, args.calibration_index, args.cache_root)
    params = build_patch_params(
        csv_path=str(args.calibration_index),
        patch_emb_cache_dir=str(args.cache_root),
        duration_sec=2,
        compact=True,
        max_patches_for_fit=300_000,
        aggregation="mean",
        bottomk_ratio=0.2,
        temporal_run_length=3,
        aggregation_region_size=3,
        seed=42,
        patch_temp_mode="same_grid_second_order",
        match_radius=2,
        top_m=4,
        temperature=0.07,
        lambda_dist=0.01,
        patch_region_size=1,
        max_real_videos=None,
    )
    if len(params["calib_patch_spat_scores"]) != 199:
        raise ValueError("parameter fit did not score all 199 calibration videos")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.stem}.tmp.npz")
    np.savez(temporary, **params)
    temporary.replace(args.output)
    write_json(
        args.metadata,
        {
            "schema_version": "u0_external_local_params_v1",
            "dataset": DATASET,
            "generated_video_count": 0,
            "real_calibration_count": 199,
            "region": 1,
            "aggregation": "mean",
            "layer": 23,
            "temporal_feature": "same_grid_second_order_then_feature_axis_l2",
            "reservoir_seed": 42,
            "reservoir_size": 300000,
            "calibration_index": str(args.calibration_index.relative_to(ROOT)),
            "calibration_index_sha256": sha256_file(args.calibration_index),
            "params": str(args.output.relative_to(ROOT)),
            "params_sha256": sha256_file(args.output),
        },
    )
    print(f"wrote locked external Local params -> {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-index",
        type=Path,
        default=ROOT / "cache/indexes/genvidbench_pair1_ms_vript_calib.csv",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "cache/patch_embeddings/genvidbench_pair1_ms_vript",
    )
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=ROOT / "release/u0_external_genvidbench",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "release/u0_external_genvidbench/params/region1_mean.npz",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "release/u0_external_genvidbench/local_params_metadata.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
