#!/usr/bin/env python3
"""Training-free spatial-neighbor temporal coherence scores for patch tokens.

The probe treats cached patch embeddings as a local motion field. For each
patch, it compares temporal deltas and accelerations against its four spatial
neighbors. Generated videos can contain local temporal breaks where a patch
trajectory is inconsistent with its immediate neighborhood even when global
temporal features look plausible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm


KEY_COLUMNS = ["subset", "source_model", "filename"]


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or str(value) == "nan"


def _cache_path(cache_root: Path, subset: str, source_model: str, video_path: str, duration: int) -> Path:
    return cache_root / subset / source_model / f"{Path(video_path).stem}_{duration}s.pt"


def _stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return {f"{prefix}_{k}": 0.0 for k in ["mean", "std", "p10", "p50", "p90", "p95"]}
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std()),
        f"{prefix}_p10": float(np.quantile(arr, 0.10)),
        f"{prefix}_p50": float(np.quantile(arr, 0.50)),
        f"{prefix}_p90": float(np.quantile(arr, 0.90)),
        f"{prefix}_p95": float(np.quantile(arr, 0.95)),
    }


def _grid_hw(grid_size, patch_count: int) -> tuple[int, int]:
    if isinstance(grid_size, torch.Tensor):
        grid_size = grid_size.detach().cpu().tolist()
    if isinstance(grid_size, (list, tuple)) and len(grid_size) == 2:
        return int(grid_size[0]), int(grid_size[1])
    side = int(round(np.sqrt(patch_count)))
    if side * side == patch_count:
        return side, side
    return 1, patch_count


def _neighbor_pairs(height: int, width: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            if x + 1 < width:
                pairs.append((idx, idx + 1))
            if y + 1 < height:
                pairs.append((idx, idx + width))
    return pairs


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9
    return (a * b).sum(axis=-1) / denom


def _coherence_features(patch: np.ndarray, grid_size, prefix: str = "neigh") -> dict[str, float]:
    """Compute coherence features from patch embeddings shaped [T, P, D]."""
    arr = patch.astype(np.float32, copy=False)
    if arr.ndim != 3 or arr.shape[0] < 3:
        out = {}
        for name in [
            "delta_cos",
            "accel_cos",
            "delta_resid",
            "accel_resid",
            "isolation",
            "spatial_roughness",
        ]:
            out.update(_stats(f"{prefix}_{name}", np.zeros(0, dtype=np.float32)))
        return out

    height, width = _grid_hw(grid_size, arr.shape[1])
    pairs = _neighbor_pairs(height, width)
    if not pairs:
        return _coherence_features(arr[:0], (0, 0), prefix)
    left = np.asarray([a for a, _ in pairs], dtype=np.int64)
    right = np.asarray([b for _, b in pairs], dtype=np.int64)

    delta = arr[1:] - arr[:-1]
    accel = arr[2:] - 2.0 * arr[1:-1] + arr[:-2]

    delta_left = delta[:, left, :]
    delta_right = delta[:, right, :]
    accel_left = accel[:, left, :]
    accel_right = accel[:, right, :]

    delta_cos = _cosine_similarity(delta_left, delta_right).reshape(-1)
    accel_cos = _cosine_similarity(accel_left, accel_right).reshape(-1)
    delta_resid = np.linalg.norm(delta_left - delta_right, axis=-1) / (
        np.linalg.norm(delta_left, axis=-1) + np.linalg.norm(delta_right, axis=-1) + 1e-9
    )
    accel_resid = np.linalg.norm(accel_left - accel_right, axis=-1) / (
        np.linalg.norm(accel_left, axis=-1) + np.linalg.norm(accel_right, axis=-1) + 1e-9
    )

    # Per-patch local isolation: how far a patch's delta is from the mean of its
    # direct neighbors. A high tail means isolated patch motion breaks.
    grid_delta = delta.reshape(delta.shape[0], height, width, delta.shape[2])
    isolation_parts = []
    for y in range(height):
        for x in range(width):
            neigh = []
            if x > 0:
                neigh.append(grid_delta[:, y, x - 1, :])
            if x + 1 < width:
                neigh.append(grid_delta[:, y, x + 1, :])
            if y > 0:
                neigh.append(grid_delta[:, y - 1, x, :])
            if y + 1 < height:
                neigh.append(grid_delta[:, y + 1, x, :])
            if not neigh:
                continue
            center = grid_delta[:, y, x, :]
            neigh_mean = np.stack(neigh, axis=0).mean(axis=0)
            resid = np.linalg.norm(center - neigh_mean, axis=-1) / (
                np.linalg.norm(center, axis=-1) + np.linalg.norm(neigh_mean, axis=-1) + 1e-9
            )
            isolation_parts.append(resid)
    isolation = np.concatenate(isolation_parts, axis=0).reshape(-1) if isolation_parts else np.zeros(0, dtype=np.float32)

    # Spatial roughness of frame-to-frame displacement magnitudes.
    mag = np.linalg.norm(delta, axis=-1).reshape(delta.shape[0], height, width)
    rough_parts = []
    rough_parts.append(np.abs(mag[:, :, 1:] - mag[:, :, :-1]).reshape(-1))
    rough_parts.append(np.abs(mag[:, 1:, :] - mag[:, :-1, :]).reshape(-1))
    spatial_roughness = np.concatenate(rough_parts, axis=0) if rough_parts else np.zeros(0, dtype=np.float32)

    out = {}
    out.update(_stats(f"{prefix}_delta_cos", delta_cos))
    out.update(_stats(f"{prefix}_accel_cos", accel_cos))
    out.update(_stats(f"{prefix}_delta_resid", delta_resid.reshape(-1)))
    out.update(_stats(f"{prefix}_accel_resid", accel_resid.reshape(-1)))
    out.update(_stats(f"{prefix}_isolation", isolation))
    out.update(_stats(f"{prefix}_spatial_roughness", spatial_roughness))
    return out


def _score_from_features(df: pd.DataFrame, mode: str) -> np.ndarray:
    """Realness scores. Higher should mean more real-like."""
    if mode == "delta_coherence":
        return (
            df["neigh_delta_cos_mean"].to_numpy()
            + 0.5 * df["neigh_delta_cos_p10"].to_numpy()
            - 0.5 * df["neigh_delta_resid_p90"].to_numpy()
        )
    if mode == "accel_coherence":
        return (
            df["neigh_accel_cos_mean"].to_numpy()
            + 0.5 * df["neigh_accel_cos_p10"].to_numpy()
            - 0.5 * df["neigh_accel_resid_p90"].to_numpy()
        )
    if mode == "anti_isolation":
        return (
            -df["neigh_isolation_p95"].to_numpy()
            -0.5 * df["neigh_spatial_roughness_p95"].to_numpy()
            +0.25 * df["neigh_delta_cos_p10"].to_numpy()
        )
    if mode == "neighbor_combo":
        return (
            df["neigh_delta_cos_mean"].to_numpy()
            +0.5 * df["neigh_accel_cos_mean"].to_numpy()
            -0.75 * df["neigh_isolation_p90"].to_numpy()
            -0.25 * df["neigh_delta_resid_p90"].to_numpy()
        )
    if mode == "roughness_anomaly":
        return (
            df["neigh_spatial_roughness_p90"].to_numpy()
            +0.5 * df["neigh_isolation_p90"].to_numpy()
        )
    raise ValueError(f"Unknown score mode: {mode}")


def _available_score_modes() -> list[str]:
    return ["delta_coherence", "accel_coherence", "anti_isolation", "neighbor_combo", "roughness_anomaly"]


def _metrics(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    real = df[df["subset"].astype(str).str.lower() == "real"]
    fake = df[df["subset"].astype(str).str.lower() != "real"]
    rows = []
    for model, group in fake.groupby("source_model", sort=True):
        y = np.concatenate([np.ones(len(real)), np.zeros(len(group))])
        s = np.concatenate([real[score_col].to_numpy(float), group[score_col].to_numpy(float)])
        rows.append(
            {
                "source_model": model,
                "n_real": len(real),
                "n_fake": len(group),
                "auc": float(roc_auc_score(y, s)),
                "ap": float(average_precision_score(y, s)),
                "auc_neg": float(roc_auc_score(y, -s)),
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out.loc[len(out)] = {
            "source_model": "Average",
            "n_real": int(round(out["n_real"].mean())),
            "n_fake": int(round(out["n_fake"].mean())),
            "auc": float(out["auc"].mean()),
            "ap": float(out["ap"].mean()),
            "auc_neg": float(out["auc_neg"].mean()),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--patch-cache", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--duration", type=int, default=2)
    parser.add_argument("--debug", type=int, default=None)
    parser.add_argument("--max-total", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--score-mode", default="neighbor_combo")
    parser.add_argument("--score-all-modes", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.shuffle:
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    if args.debug:
        df = (
            df.groupby(["subset", "source_model"], group_keys=False)
            .apply(lambda g: g.head(args.debug))
            .reset_index(drop=True)
        )
    if args.max_total:
        df = df.head(args.max_total).reset_index(drop=True)

    rows = []
    cache_root = Path(args.patch_cache)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Patch neighbor coherence"):
        rowd = row.to_dict()
        if _is_missing(rowd.get(f"{args.duration}_sec_idxs")):
            continue
        path = _cache_path(cache_root, rowd["subset"], rowd["source_model"], rowd["video_path"], args.duration)
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=True)
        patch = payload["patch"].numpy()
        feats = _coherence_features(patch, payload.get("grid_size"))
        feats.update(
            {
                "subset": rowd["subset"],
                "source_model": rowd["source_model"],
                "filename": Path(rowd["video_path"]).name,
            }
        )
        rows.append(feats)

    out = pd.DataFrame(rows)
    modes = _available_score_modes() if args.score_all_modes else [args.score_mode]
    for mode in modes:
        out[f"score_{mode}"] = _score_from_features(out, mode)
    out["final_score"] = out[f"score_{args.score_mode}"]
    out["score_mode"] = args.score_mode

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    for mode in modes:
        print(f"\n== {mode} ==")
        metrics = _metrics(out, f"score_{mode}")
        print(
            metrics.to_string(
                index=False,
                formatters={
                    "auc": lambda x: f"{x:.4f}",
                    "ap": lambda x: f"{x:.4f}",
                    "auc_neg": lambda x: f"{x:.4f}",
                },
            )
        )
    print(f"Saved -> {output}")


if __name__ == "__main__":
    main()
