#!/usr/bin/env python3
"""Fit the clean-universal layer-17 region-1 mean temporal model."""

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
SRC_DIR = REPO_ROOT / "src"
TOOLS_DIR = REPO_ROOT / "tools"
for path in (SRC_DIR, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from create_patch_params import _fit_whitening, _get_mu_W, reservoir_update
from eval_patch_fast import FastPatchScorer
from fit_intermediate_layer_params import cache_name, checkpoint_hash
from patch_matching import patch_temporal_delta
from score_multi_window import (
    KEY_COLUMNS,
    decode_manifest_row_with_retries,
    video_key,
)
from stall import DINO_V3_WEIGHTS
from stall_patch import PatchSTALL


LAYER = 17


def load_temporal(path: Path) -> torch.Tensor:
    payload = torch.load(path, weights_only=True, map_location="cpu")
    temporal = payload["temporal"]
    return temporal[LAYER] if isinstance(temporal, dict) else temporal


def extract_calibration(args: argparse.Namespace, cache_dir: Path) -> None:
    manifest = pd.read_csv(args.manifest)
    calibration = manifest[
        (manifest["dataset"] == args.dataset)
        & (manifest["protocol_split"] == "calibration")
        & (manifest["subset"] == "real")
    ].reset_index(drop=True)
    if len(calibration) != 200:
        raise ValueError(f"{args.dataset}: expected 200 real calibration videos")
    cache_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        row
        for _, row in calibration.iterrows()
        if not (cache_dir / cache_name(row)).exists()
    ]
    print(
        f"[{args.dataset}] calibration=200 cached={200-len(pending)} "
        f"pending={len(pending)}",
        flush=True,
    )
    if not pending:
        return

    extractor = PatchSTALL(args.device, data_dict=None, load_dino=True)
    started = time.perf_counter()
    for start in range(0, len(pending), args.video_batch_size):
        rows = pending[start : start + args.video_batch_size]
        decoded = []
        with ThreadPoolExecutor(max_workers=min(args.decode_workers, len(rows))) as pool:
            futures = [
                pool.submit(
                    decode_manifest_row_with_retries,
                    row,
                    "K1_current",
                    args.seek_gap,
                    args.decode_attempts,
                )
                for row in rows
            ]
            for row, future in zip(rows, futures):
                try:
                    decoded.append(future.result())
                except Exception as error:
                    raise RuntimeError(
                        f"failed calibration decode: {video_key(row)}"
                    ) from error
        outputs = extractor.frames_to_layer_patch_embeddings(
            [item["frames"] for item in decoded],
            layers=(LAYER,),
            batch_size=args.frame_batch_size,
        )
        for item, output in zip(decoded, outputs):
            positions = item["window_positions"][0]
            patch = output["layers"][LAYER][positions].astype(np.float32)
            temporal = torch.from_numpy(
                patch_temporal_delta(
                    patch,
                    grid_size=tuple(output["grid_size"]),
                    mode="same_grid_second_order",
                    region_size=1,
                )
            )
            source = item["row"]
            payload = {
                "temporal": temporal,
                "grid_size": tuple(output["grid_size"]),
                "frame_indices": item["windows"][0],
                "key": tuple(str(source[column]) for column in KEY_COLUMNS),
                "layer": LAYER,
                "region": 1,
            }
            destination = cache_dir / cache_name(source)
            temporary = destination.with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.rename(destination)
        processed = min(start + args.video_batch_size, len(pending))
        if start == 0 or processed == len(pending) or processed % 40 == 0:
            print(
                f"[{args.dataset}] extracted={processed}/{len(pending)} "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )


def calibration_paths(args: argparse.Namespace, cache_dir: Path) -> list[Path]:
    if args.dataset == "videofeedback" and not args.force_extract:
        source = (
            REPO_ROOT
            / "results/unified_multiscale_layers/layer_calibration_cache/videofeedback"
        )
        paths = sorted(source.glob("*.pt"))
    else:
        paths = sorted(cache_dir.glob("*.pt"))
    if len(paths) != 200:
        raise ValueError(f"{args.dataset}: expected 200 calibration caches, got {len(paths)}")
    return paths


def fit(args: argparse.Namespace, paths: list[Path]) -> None:
    rng = np.random.RandomState(args.seed)
    reservoir = None
    seen = 0
    for index, path in enumerate(paths, start=1):
        temporal = load_temporal(path)
        values = temporal.numpy().reshape(-1, temporal.shape[-1]).astype(np.float32)
        reservoir, seen = reservoir_update(
            reservoir, values, args.max_patches_for_fit, seen, rng
        )
        if index % 50 == 0:
            print(f"[{args.dataset}/L17] reservoir videos={index} seen={seen}")
    if reservoir is None:
        raise ValueError("no layer-17 calibration features")
    fit_values = reservoir[: min(args.max_patches_for_fit, seen)]
    transform = _fit_whitening(fit_values)
    mu, whitening = _get_mu_W(transform)
    eigenvalues = transform.get_eigenvalues().detach().cpu().numpy().astype(np.float64)

    base_path = args.params_dir / f"{args.dataset}_region1_mean.npz"
    base = np.load(base_path, allow_pickle=True)
    scorer = FastPatchScorer(
        str(base_path), device="cuda" if torch.cuda.is_available() else "cpu"
    )
    device = scorer.device
    mu_tensor = torch.as_tensor(mu, dtype=torch.float32, device=device)
    whitening_tensor = torch.as_tensor(whitening, dtype=torch.float32, device=device)
    calibration = []
    with torch.inference_mode():
        for path in paths:
            temporal = load_temporal(path).to(device=device, dtype=torch.float32)
            likelihood = scorer.log_likelihood_from_white(
                torch.matmul(temporal - mu_tensor, whitening_tensor)
            ).unsqueeze(0)
            calibration.append(float(likelihood.mean().item()))
    config = json.loads(str(base["aggregation_config"].item()))
    config.update(
        {
            "mode": "mean",
            "bottomk_ratio": 0.2,
            "patch_region_size": 1,
            "intermediate_layer": LAYER,
            "checkpoint_sha256": checkpoint_hash(Path(DINO_V3_WEIGHTS)),
            "calibration_videos": 200,
            "max_patches_for_fit": args.max_patches_for_fit,
            "implementation": "clean_universal_layer17_real200_v1",
        }
    )
    payload = {key: base[key] for key in base.files}
    payload.update(
        {
            "mu_patch_temp": mu.astype(np.float32),
            "W_patch_temp": whitening.astype(np.float32),
            "calib_patch_temp_scores": np.asarray(calibration, dtype=np.float32),
            "whitening_eigenvalues": eigenvalues,
            "aggregation_config": np.array(json.dumps(config, sort_keys=True)),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.dataset}_layer17_region1_mean.npz"
    np.savez(output, **payload)
    print(
        f"wrote {output} W={whitening.shape} eigen="
        f"[{eigenvalues.min():.9g},{eigenvalues.max():.9g}]",
        flush=True,
    )
    if scorer._executor is not None:
        scorer._executor.shutdown(wait=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("comgenvid", "videofeedback", "genvideo"), required=True
    )
    parser.add_argument("--stage", choices=("extract", "fit", "all"), default="all")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT
        / "results/multi_window_joint_typicality/multi_window_manifest.csv",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=REPO_ROOT / "results/clean_universal/layer17_calibration_cache",
    )
    parser.add_argument(
        "--params-dir",
        type=Path,
        default=REPO_ROOT / "results/clean_universal/params",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results/clean_universal/params",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-batch-size", type=int, default=4)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--decode-workers", type=int, default=4)
    parser.add_argument("--decode-attempts", type=int, default=3)
    parser.add_argument("--seek-gap", type=int, default=64)
    parser.add_argument("--max-patches-for-fit", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-extract", action="store_true")
    args = parser.parse_args()
    cache_dir = args.cache_root / args.dataset
    if args.stage in {"extract", "all"} and (
        args.dataset != "videofeedback" or args.force_extract
    ):
        extract_calibration(args, cache_dir)
    if args.stage in {"fit", "all"}:
        fit(args, calibration_paths(args, cache_dir))


if __name__ == "__main__":
    main()
