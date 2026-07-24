#!/usr/bin/env python3
"""Extract real-only intermediate-layer D2 caches and fit independent parameters."""

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


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
TOOLS_DIR = REPO_ROOT / "tools"
for path in (SRC_DIR, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from create_patch_params import _fit_whitening, _get_mu_W, reservoir_update
from eval_patch_fast import FastPatchScorer
from patch_matching import patch_temporal_delta
from score_multi_window import (
    KEY_COLUMNS,
    decode_manifest_row_with_retries,
    video_key,
)
from stall import DINO_V3_WEIGHTS
from stall_patch import PatchSTALL


LAYERS = (11, 17, 23)
SPECS = {
    "comgenvid": {
        "region": 3,
        "base_params": REPO_ROOT
        / "results/unified_multiscale_layers/region_params/comgenvid_region3.npz",
    },
    "videofeedback": {
        "region": 1,
        "base_params": REPO_ROOT
        / "results/unified_multiscale_layers/region_params/videofeedback_region1.npz",
    },
    "genvideo": {
        "region": 2,
        "base_params": REPO_ROOT
        / "results/unified_multiscale_layers/region_params/genvideo_region2.npz",
    },
}


def checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_name(row: pd.Series) -> str:
    key = "|".join(str(row[column]) for column in KEY_COLUMNS)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24] + ".pt"


def extract_calibration(args: argparse.Namespace, cache_dir: Path) -> None:
    manifest = pd.read_csv(args.manifest)
    calibration = manifest[
        (manifest["dataset"] == args.dataset)
        & (manifest["protocol_split"] == "calibration")
        & (manifest["subset"] == "real")
    ].reset_index(drop=True)
    if args.debug_videos is not None:
        calibration = calibration.head(args.debug_videos)
    elif len(calibration) != 200:
        raise ValueError(f"{args.dataset}: expected 200 real calibration videos")
    cache_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        row for _, row in calibration.iterrows() if not (cache_dir / cache_name(row)).exists()
    ]
    print(
        f"[{args.dataset}] calibration={len(calibration)} cached={len(calibration)-len(pending)} "
        f"pending={len(pending)}",
        flush=True,
    )
    if not pending:
        return

    extractor = PatchSTALL(args.device, data_dict=None, load_dino=True)
    region = int(SPECS[args.dataset]["region"])
    started = time.perf_counter()
    validated_final = False
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
                    raise RuntimeError(f"failed calibration decode: {video_key(row)}") from error

        outputs = extractor.frames_to_layer_patch_embeddings(
            [item["frames"] for item in decoded], layers=LAYERS, batch_size=args.frame_batch_size
        )
        if not validated_final:
            standard = extractor.frames_to_global_patch_embeddings(
                [decoded[0]["frames"]], batch_size=args.frame_batch_size
            )[0]["patch"]
            error = float(np.max(np.abs(standard - outputs[0]["layers"][23])))
            print(f"[{args.dataset}] layer23_max_abs_error={error:.9g}", flush=True)
            if error >= 1e-6:
                raise ValueError(f"layer 23 differs from frozen final token: {error}")
            validated_final = True

        for item, output in zip(decoded, outputs):
            positions = item["window_positions"][0]
            temporal = {}
            for layer in LAYERS:
                patch = output["layers"][layer][positions].astype(np.float32)
                temporal[layer] = torch.from_numpy(
                    patch_temporal_delta(
                        patch,
                        grid_size=tuple(output["grid_size"]),
                        mode="same_grid_second_order",
                        region_size=region,
                    )
                )
            source = item["row"]
            payload = {
                "temporal": temporal,
                "grid_size": tuple(output["grid_size"]),
                "frame_indices": item["windows"][0],
                "key": tuple(str(source[column]) for column in KEY_COLUMNS),
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


def fit_layer(
    dataset: str,
    layer: int,
    cache_paths: list[Path],
    output: Path,
    max_samples: int,
    seed: int,
    model_hash: str,
) -> None:
    rng = np.random.RandomState(seed)
    reservoir = None
    seen = 0
    for index, path in enumerate(cache_paths, start=1):
        temporal = torch.load(path, weights_only=True, map_location="cpu")["temporal"][layer]
        values = temporal.numpy().reshape(-1, temporal.shape[-1]).astype(np.float32)
        reservoir, seen = reservoir_update(reservoir, values, max_samples, seen, rng)
        if index % 50 == 0:
            print(f"[{dataset}/L{layer}] reservoir videos={index} seen={seen}", flush=True)
    if reservoir is None:
        raise ValueError(f"{dataset}/L{layer}: no calibration features")
    fit = reservoir[: min(max_samples, seen)]
    transform = _fit_whitening(fit)
    mu, whitening = _get_mu_W(transform)
    del reservoir, fit, transform
    torch.cuda.empty_cache()

    base_path = Path(SPECS[dataset]["base_params"])
    base = np.load(base_path, allow_pickle=True)
    config = json.loads(str(base["aggregation_config"].item()))
    scorer = FastPatchScorer(str(base_path), device="cuda" if torch.cuda.is_available() else "cpu")
    device = scorer.device
    mu_tensor = torch.as_tensor(mu, dtype=torch.float32, device=device)
    whitening_tensor = torch.as_tensor(whitening, dtype=torch.float32, device=device)
    calibration = []
    with torch.inference_mode():
        for path in cache_paths:
            temporal = torch.load(path, weights_only=True, map_location="cpu")["temporal"][layer]
            temporal = temporal.to(device=device, dtype=torch.float32)
            white = torch.matmul(temporal - mu_tensor, whitening_tensor)
            ll = scorer.log_likelihood_from_white(white).unsqueeze(0)
            raw = scorer._aggregate(
                ll,
                config["mode"],
                float(config.get("bottomk_ratio", 0.2)),
                int(config.get("temporal_run_length", 3)),
            )
            calibration.append(float(raw.item()))
    config.update(
        {
            "intermediate_layer": layer,
            "checkpoint_sha256": model_hash,
            "calibration_videos": len(cache_paths),
            "max_patches_for_fit": max_samples,
            "implementation": "intermediate_layer_real_only_v1",
        }
    )
    payload = {key: base[key] for key in base.files}
    payload.update(
        {
            "mu_patch_temp": mu.astype(np.float32),
            "W_patch_temp": whitening.astype(np.float32),
            "calib_patch_temp_scores": np.asarray(calibration, dtype=np.float32),
            "aggregation_config": np.array(json.dumps(config)),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **payload)
    print(
        f"[{dataset}/L{layer}] wrote {output} samples={min(max_samples, seen)} "
        f"calibration={len(calibration)}",
        flush=True,
    )
    if scorer._executor is not None:
        scorer._executor.shutdown(wait=True)


def fit_all(args: argparse.Namespace, cache_dir: Path) -> None:
    paths = sorted(cache_dir.glob("*.pt"))
    expected = args.debug_videos or 200
    if len(paths) != expected:
        raise ValueError(f"{args.dataset}: expected {expected} cache files, found {len(paths)}")
    model_hash = checkpoint_hash(Path(DINO_V3_WEIGHTS))
    for layer in LAYERS:
        output = args.params_dir / f"{args.dataset}_layer{layer}.npz"
        if output.exists() and not args.force_fit:
            print(f"[{args.dataset}/L{layer}] reuse {output}", flush=True)
            continue
        fit_layer(
            args.dataset,
            layer,
            paths,
            output,
            args.max_patches_for_fit,
            args.seed,
            model_hash,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(SPECS), required=True)
    parser.add_argument("--stage", choices=("extract", "fit", "all"), default="all")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "results/multi_window_joint_typicality/multi_window_manifest.csv")
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "results/unified_multiscale_layers/layer_calibration_cache")
    parser.add_argument("--params-dir", type=Path, default=REPO_ROOT / "results/unified_multiscale_layers/layer_params")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-batch-size", type=int, default=4)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--decode-workers", type=int, default=4)
    parser.add_argument("--decode-attempts", type=int, default=3)
    parser.add_argument("--seek-gap", type=int, default=64)
    parser.add_argument("--max-patches-for-fit", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug-videos", type=int)
    parser.add_argument("--force-fit", action="store_true")
    args = parser.parse_args()
    cache_dir = args.cache_root / args.dataset
    if args.stage in {"extract", "all"}:
        extract_calibration(args, cache_dir)
    if args.stage in {"fit", "all"}:
        fit_all(args, cache_dir)


if __name__ == "__main__":
    main()
