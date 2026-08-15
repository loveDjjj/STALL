#!/usr/bin/env python3
"""Fit pooled cross-domain banks and the predeclared OAS temporal candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from alpha_stalled.release_io import write_json
from create_patch_params import reservoir_update
from patch_matching import patch_temporal_delta
from whitening_transform import WhiteningTransform


DATASETS = ("comgenvid", "videofeedback", "genvideo")
SEEDS = (17, 29, 43, 71, 101)
CACHE_ROOT = ROOT / "cache/patch_embeddings"


def stable_rank(value: str) -> str:
    return hashlib.sha256(f"42\0{value}".encode()).hexdigest()


def cache_path(item: dict) -> Path:
    return (
        CACHE_ROOT
        / item["dataset"]
        / "real"
        / item["source_model"]
        / f"{Path(item['filename']).stem}_2s.pt"
    )


def validate_records(records: list[dict], expected_windows: dict[str, list[int]]) -> None:
    for item in records:
        path = cache_path(item)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, weights_only=True, map_location="cpu")
        actual = [int(value) for value in payload["frame_indices"]]
        if actual != [int(value) for value in expected_windows[item["video_id"]]]:
            raise ValueError(f"K1 cache differs from locked window: {path}")


def collect_samples(
    records: list[dict], seed: int, max_samples: int = 300_000
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    spatial_reservoir = None
    temporal_reservoir = None
    spatial_seen = 0
    temporal_seen = 0
    for item in records:
        payload = torch.load(cache_path(item), weights_only=True, map_location="cpu")
        patch = payload["patch"].numpy().astype(np.float32)
        spatial_reservoir, spatial_seen = reservoir_update(
            spatial_reservoir,
            patch.reshape(-1, patch.shape[-1]),
            max_samples,
            spatial_seen,
            rng,
        )
        temporal = patch_temporal_delta(
            patch,
            grid_size=tuple(int(value) for value in payload["grid_size"]),
            mode="same_grid_second_order",
            radius=2,
            top_m=4,
            temperature=0.07,
            lambda_dist=0.01,
            region_size=1,
            global_seq=payload["global"].numpy().astype(np.float32),
        )
        temporal_reservoir, temporal_seen = reservoir_update(
            temporal_reservoir,
            temporal.reshape(-1, temporal.shape[-1]).astype(np.float32),
            max_samples,
            temporal_seen,
            rng,
        )
    if spatial_reservoir is None or temporal_reservoir is None:
        raise ValueError("empty calibration bank")
    return (
        spatial_reservoir[: min(spatial_seen, max_samples)],
        temporal_reservoir[: min(temporal_seen, max_samples)],
    )


def standard_whitening(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transform = WhiteningTransform(data=samples)
    return (
        transform.mean_.cpu().numpy().astype(np.float64),
        transform.whitening_matrix_.cpu().numpy().astype(np.float64),
    )


def oas_whitening(
    samples: np.ndarray, device: str
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    target = torch.device(device)
    values = torch.as_tensor(samples, dtype=torch.float64, device=target)
    n_samples, n_features = values.shape
    mean = values.mean(dim=0, dtype=torch.float64)
    centered = values - mean
    covariance = torch.mm(centered.T, centered) / float(n_samples)
    alpha = torch.mean(covariance * covariance)
    mu = torch.trace(covariance) / float(n_features)
    mu_squared = mu * mu
    denominator = (n_samples + 1.0) * (alpha - mu_squared / float(n_features))
    shrinkage = (
        torch.tensor(1.0, dtype=torch.float64, device=target)
        if float(denominator) == 0.0
        else torch.clamp((alpha + mu_squared) / denominator, max=1.0)
    )
    covariance.mul_(1.0 - shrinkage)
    covariance.diagonal().add_(shrinkage * mu)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if not torch.all(eigenvalues > 0):
        raise ValueError("OAS covariance is not positive definite")
    whitening = eigenvectors * torch.rsqrt(eigenvalues + 1e-5).unsqueeze(0)
    metadata = {
        "oas_shrinkage": float(shrinkage.cpu()),
        "eigenvalue_min": float(eigenvalues[-1].cpu()),
        "eigenvalue_max": float(eigenvalues[0].cpu()),
        "condition_number": float((eigenvalues[0] / eigenvalues[-1]).cpu()),
        "effective_rank": int(n_features),
        "fit_samples": int(n_samples),
    }
    return mean.cpu().numpy(), whitening.cpu().numpy(), metadata


def save_params(
    path: Path,
    spatial: tuple[np.ndarray, np.ndarray],
    temporal: tuple[np.ndarray, np.ndarray],
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(
        temporary,
        mu_patch_spat=spatial[0],
        W_patch_spat=spatial[1],
        calib_patch_spat_scores=np.array([0.0], dtype=np.float64),
        mu_patch_temp=temporal[0],
        W_patch_temp=temporal[1],
        calib_patch_temp_scores=np.array([0.0], dtype=np.float64),
        patch_grid_size=np.array([14, 14], dtype=np.int32),
        duration=np.array([2], dtype=np.int32),
        aggregation_config=np.array(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(path)


def locked_records(release_dir: Path) -> tuple[list[dict], dict[str, list[int]]]:
    payload = json.loads((release_dir / "calibration_manifest.json").read_text())
    frames = json.loads((release_dir / "frame_indices.json").read_text())
    records = payload["videos"]
    windows = frames["calibration_reference_windows"]
    validate_records(records, windows)
    return records, windows


def pooled_200(records: list[dict]) -> list[dict]:
    allocation = {"comgenvid": 67, "videofeedback": 67, "genvideo": 66}
    selected = []
    for dataset, count in allocation.items():
        group = sorted(
            [item for item in records if item["dataset"] == dataset],
            key=lambda item: stable_rank(item["video_id"]),
        )
        selected.extend(group[:count])
    return sorted(selected, key=lambda item: item["video_id"])


def fit_pooled(records: list[dict], name: str, output_dir: Path) -> None:
    path = output_dir / "cross" / f"{name}.npz"
    if path.is_file():
        print(f"reuse {path}", flush=True)
        return
    started = time.perf_counter()
    spatial_samples, temporal_samples = collect_samples(records, seed=42)
    spatial = standard_whitening(spatial_samples)
    temporal = standard_whitening(temporal_samples)
    save_params(
        path,
        spatial,
        temporal,
        {
            "algorithm": "current_effective_rank_whitening",
            "calibration_bank": name,
            "calibration_videos": len(records),
            "seed": 42,
            "region": 1,
            "aggregation": "mean",
            "patch_temp_mode": "same_grid_second_order",
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    print(f"wrote {path}", flush=True)


def fit_oas(
    records: list[dict],
    base_path: Path,
    output: Path,
    seed: int,
    bank_name: str,
    device: str,
) -> None:
    if output.is_file():
        print(f"reuse {output}", flush=True)
        return
    started = time.perf_counter()
    _, temporal_samples = collect_samples(records, seed=seed)
    base = np.load(base_path, allow_pickle=True)
    spatial = (
        base["mu_patch_spat"].astype(np.float64),
        base["W_patch_spat"].astype(np.float64),
    )
    temporal_mean, temporal_whitening, metadata = oas_whitening(
        temporal_samples, device
    )
    metadata.update(
        {
            "algorithm": "OAS",
            "calibration_bank": bank_name,
            "calibration_videos": len(records),
            "seed": seed,
            "region": 1,
            "aggregation": "mean",
            "patch_temp_mode": "same_grid_second_order",
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    save_params(
        output,
        spatial,
        (temporal_mean, temporal_whitening),
        metadata,
    )
    print(f"wrote {output} shrinkage={metadata['oas_shrinkage']:.6g}", flush=True)


def run(args: argparse.Namespace) -> None:
    records, windows = locked_records(args.release_dir)
    pooled200 = pooled_200(records)
    banks = {
        **{dataset: [item for item in records if item["dataset"] == dataset] for dataset in DATASETS},
        "pooled200": pooled200,
        "pooled600": records,
    }
    write_json(
        args.bank_manifest,
        {
            "schema_version": "u0_cross_calibration_banks_v1",
            "selection": "locked real-only calibration; pooled200 uses fixed 67/67/66 stable-hash allocation",
            "banks": {
                name: {
                    "video_count": len(bank),
                    "dataset_counts": pd.Series([item["dataset"] for item in bank]).value_counts().sort_index().to_dict(),
                    "video_ids": [item["video_id"] for item in bank],
                }
                for name, bank in banks.items()
            },
        },
    )
    fit_pooled(pooled200, "pooled200", args.output_dir)
    fit_pooled(records, "pooled600", args.output_dir)

    config = json.loads(args.local_paths.read_text()) if args.local_paths else None
    if config is None:
        import yaml

        config = yaml.safe_load(args.config.read_text())
        local_paths = {
            dataset: ROOT / config["local_branch"]["params_by_dataset"][dataset]["path"]
            for dataset in DATASETS
        }
    else:
        local_paths = {dataset: Path(path) for dataset, path in config.items()}
    for dataset in DATASETS:
        fit_oas(
            banks[dataset],
            local_paths[dataset],
            args.output_dir / "oas" / f"{dataset}_locked200.npz",
            42,
            f"{dataset}_locked200",
            args.device,
        )

    reserve_payload = json.loads(args.reserve_manifest.read_text())
    reserve = {item["video_id"]: item for item in reserve_payload["videos"]}
    membership = pd.read_csv(args.membership)
    for dataset in DATASETS:
        for seed in SEEDS:
            ids = membership[
                membership["dataset"].eq(dataset)
                & membership["seed"].eq(seed)
                & membership["calibration_size"].eq(200)
            ]["video_id"].tolist()
            bank = [reserve[video_id] for video_id in ids]
            # Reserve entries already carry explicit cache paths, but cache_path()
            # resolves to the same canonical location and was validated in phase 6.
            base = args.calibration_params / f"{dataset}_seed{seed}_n200.npz"
            fit_oas(
                bank,
                base,
                args.output_dir / "oas" / f"{dataset}_seed{seed}_n200.npz",
                seed,
                f"{dataset}_reserve_seed{seed}_n200",
                args.device,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=ROOT / "release/u0")
    parser.add_argument(
        "--bank-manifest",
        type=Path,
        default=ROOT / "release/u0/cross_calibration_banks.json",
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/alpha_stalled_u0_locked.yaml"
    )
    parser.add_argument("--local-paths", type=Path)
    parser.add_argument(
        "--reserve-manifest",
        type=Path,
        default=ROOT / "release/u0/calibration_reserve_manifest.json",
    )
    parser.add_argument(
        "--membership", type=Path, default=ROOT / "release/u0/calibration_split_membership.csv"
    )
    parser.add_argument(
        "--calibration-params",
        type=Path,
        default=ROOT / "results/u0_calibration_sensitivity/params",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/u0_cross_and_oas"
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
