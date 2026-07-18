#!/usr/bin/env python3
"""Spatial local incoherence on patch motion maps.

This applies FALCON/SPLIT-style local-pattern aggregation to stronger motion
maps instead of raw patch likelihood maps:
  - forward-backward cycle uncertainty
  - trajectory acceleration / incoherence

It is training-free and uses only cached patch embeddings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from patch_matching import matching_diagnostics_for_pair, patch_id_to_rc  # noqa: E402


KEY_COLUMNS = ["subset", "source_model", "filename"]


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or str(value) == "nan"


def _cache_path(cache_root: Path, subset: str, source_model: str, video_path: str, duration: int) -> Path:
    return cache_root / subset / source_model / f"{Path(video_path).stem}_{duration}s.pt"


def _coords(patch_ids: np.ndarray, grid_size: tuple[int, int]) -> np.ndarray:
    return np.asarray([patch_id_to_rc(int(x), grid_size) for x in patch_ids], dtype=np.float32)


def _rc_distance(a: np.ndarray, b: np.ndarray, grid_size: tuple[int, int]) -> np.ndarray:
    ca = _coords(a, grid_size)
    cb = _coords(b, grid_size)
    delta = ca - cb
    return np.sqrt((delta * delta).sum(axis=1)).astype(np.float32)


def _stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    keys = ("mean", "std", "p10", "p50", "p90", "p95")
    if arr.size == 0:
        return {f"{prefix}_{k}": 0.0 for k in keys}
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std()),
        f"{prefix}_p10": float(np.quantile(arr, 0.10)),
        f"{prefix}_p50": float(np.quantile(arr, 0.50)),
        f"{prefix}_p90": float(np.quantile(arr, 0.90)),
        f"{prefix}_p95": float(np.quantile(arr, 0.95)),
    }


def _topk_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    out: dict[str, float] = {}
    if arr.size == 0:
        for ratio in (0.05, 0.10, 0.20):
            out[f"{prefix}_top{int(round(ratio * 100)):02d}_mean"] = 0.0
        return out
    order = np.sort(arr)[::-1]
    for ratio in (0.05, 0.10, 0.20):
        k = max(1, int(round(arr.size * ratio)))
        out[f"{prefix}_top{int(round(ratio * 100)):02d}_mean"] = float(order[:k].mean())
    return out


def _gini(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    arr = arr - arr.min()
    if arr.sum() <= 1e-12:
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * (idx * arr).sum()) / (n * arr.sum()) - (n + 1.0) / n)


def _neighbor_mean(x: np.ndarray) -> np.ndarray:
    # x: [T,H,W]
    padded = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="edge")
    total = np.zeros_like(x, dtype=np.float32)
    count = 0
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            total += padded[:, dy : dy + x.shape[1], dx : dx + x.shape[2]]
            count += 1
    return total / float(count)


def _directional_diffs(x: np.ndarray) -> np.ndarray:
    padded = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="edge")
    center = x
    diffs = []
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            neigh = padded[:, dy : dy + x.shape[1], dx : dx + x.shape[2]]
            diffs.append(center - neigh)
    return np.stack(diffs, axis=-1)


def _crop_values(x: np.ndarray) -> np.ndarray:
    h, w = x.shape[1:]
    h_mid, w_mid = h // 2, w // 2
    masks = [
        np.s_[:, :h_mid, :w_mid],
        np.s_[:, :h_mid, w_mid:],
        np.s_[:, h_mid:, :w_mid],
        np.s_[:, h_mid:, w_mid:],
        np.s_[:, h // 4 : max(h // 4 + 1, 3 * h // 4), w // 4 : max(w // 4 + 1, 3 * w // 4)],
        np.s_[:, 1:-1, 1:-1] if h > 2 and w > 2 else np.s_[:, :, :],
    ]
    vals = []
    for mask in masks:
        part = x[mask]
        vals.append(float(part.mean()) if part.size else float(x.mean()))
    return np.asarray(vals, dtype=np.float32)


def _local_pattern_features(prefix: str, values: np.ndarray, grid_size: tuple[int, int]) -> dict[str, float]:
    if values.size == 0:
        values = np.zeros((1, grid_size[0] * grid_size[1]), dtype=np.float32)
    gh, gw = grid_size
    x = values.reshape(values.shape[0], gh, gw).astype(np.float32)
    residual = np.abs(x - _neighbor_mean(x))
    diffs = np.abs(_directional_diffs(x))
    sign_balance = np.abs((_directional_diffs(x) > 0).mean(axis=-1) - 0.5) * 2.0
    crop = _crop_values(x)
    temporal = np.abs(np.diff(x, axis=0)) if x.shape[0] >= 2 else np.zeros_like(x)
    residual_temporal = np.abs(np.diff(residual, axis=0)) if residual.shape[0] >= 2 else np.zeros_like(residual)

    out: dict[str, float] = {}
    out.update(_stats(f"{prefix}_map", x))
    out.update(_topk_stats(f"{prefix}_map", x))
    out.update(_stats(f"{prefix}_neighbor_residual", residual))
    out.update(_topk_stats(f"{prefix}_neighbor_residual", residual))
    out.update(_stats(f"{prefix}_lvp_absdiff", diffs))
    out.update(_topk_stats(f"{prefix}_lvp_absdiff", diffs))
    out.update(_stats(f"{prefix}_lvp_sign_balance", sign_balance))
    out.update(_stats(f"{prefix}_temporal_abs", temporal))
    out.update(_topk_stats(f"{prefix}_temporal_abs", temporal))
    out.update(_stats(f"{prefix}_residual_temporal", residual_temporal))
    out.update(_topk_stats(f"{prefix}_residual_temporal", residual_temporal))
    out[f"{prefix}_residual_gini"] = _gini(residual)
    out[f"{prefix}_lvp_gini"] = _gini(diffs)
    out[f"{prefix}_crop_std"] = float(crop.std())
    out[f"{prefix}_crop_range"] = float(crop.max() - crop.min())
    out[f"{prefix}_crop_top_gap"] = float(crop.max() - np.median(crop))
    return out


def _motion_maps(
    patch: np.ndarray,
    grid_size: tuple[int, int],
    radius: int,
    top_m: int,
    temperature: float,
    lambda_dist: float,
    mode: str,
) -> dict[str, np.ndarray]:
    arr = patch.astype(np.float32, copy=False)
    num_frames, num_patches = arr.shape[:2]
    patch_ids = np.arange(num_patches, dtype=np.int64)
    start_coords = _coords(patch_ids, grid_size)

    fb_uncertain = []
    fb_error_maps = []
    velocity = []
    margin = []
    entropy = []

    for t in range(num_frames - 1):
        fwd = matching_diagnostics_for_pair(
            arr[t], arr[t + 1], grid_size=grid_size, radius=radius,
            top_m=top_m, temperature=temperature, lambda_dist=lambda_dist, mode=mode
        )
        bwd = matching_diagnostics_for_pair(
            arr[t + 1], arr[t], grid_size=grid_size, radius=radius,
            top_m=top_m, temperature=temperature, lambda_dist=lambda_dist, mode=mode
        )
        dst = fwd["matched_ids"]
        back_to = bwd["matched_ids"][dst]
        fb_error = _rc_distance(patch_ids, back_to, grid_size)
        fwd_margin = fwd["top1_scores"] - fwd["top2_scores"]
        bwd_margin = bwd["top1_scores"][dst] - bwd["top2_scores"][dst]
        fwd_entropy = np.nan_to_num(fwd["entropy"], nan=0.0)
        bwd_entropy = np.nan_to_num(bwd["entropy"][dst], nan=0.0)
        uncertain = fb_error + 0.25 * (fwd_entropy + bwd_entropy) - 0.25 * (fwd_margin + bwd_margin)
        fb_uncertain.append(uncertain.astype(np.float32))
        fb_error_maps.append(fb_error.astype(np.float32))

        dst_coords = _coords(dst, grid_size)
        velocity.append((dst_coords - start_coords).astype(np.float32))
        margin.append(fwd_margin.astype(np.float32))
        entropy.append(fwd_entropy.astype(np.float32))

    maps: dict[str, np.ndarray] = {
        "fb_uncertain": np.stack(fb_uncertain, axis=0) if fb_uncertain else np.zeros((0, num_patches), dtype=np.float32),
        "fb_error": np.stack(fb_error_maps, axis=0) if fb_error_maps else np.zeros((0, num_patches), dtype=np.float32),
    }
    if velocity:
        vel = np.stack(velocity, axis=0)
        speed = np.sqrt((vel * vel).sum(axis=-1))
        maps["traj_speed"] = speed.astype(np.float32)
        margin_arr = np.stack(margin, axis=0)
        entropy_arr = np.stack(entropy, axis=0)
        if vel.shape[0] >= 2:
            accel = vel[1:] - vel[:-1]
            accel_norm = np.sqrt((accel * accel).sum(axis=-1))
            margin_drop = np.maximum(0.0, margin_arr[:-1] - margin_arr[1:])
            entropy_jump = np.maximum(0.0, entropy_arr[1:] - entropy_arr[:-1])
            maps["traj_accel"] = accel_norm.astype(np.float32)
            maps["traj_incoherence"] = (accel_norm + 0.25 * entropy_jump + 0.25 * margin_drop).astype(np.float32)
        else:
            maps["traj_accel"] = np.zeros((0, num_patches), dtype=np.float32)
            maps["traj_incoherence"] = np.zeros((0, num_patches), dtype=np.float32)
    return maps


def _score_modes(df: pd.DataFrame) -> dict[str, np.ndarray]:
    scores = {
        # Higher-is-real convention. Raw map intensity uses the known useful
        # orientation from earlier trajectory/FB runs.
        "fb_spatial_incoherence_real": -(
            df["fb_uncertain_neighbor_residual_top10_mean"].to_numpy(float)
            + df["fb_uncertain_crop_range"].to_numpy(float)
        ),
        "fb_spatial_smooth_real": (
            df["fb_uncertain_neighbor_residual_top10_mean"].to_numpy(float)
            + df["fb_uncertain_crop_range"].to_numpy(float)
        ),
        "traj_spatial_incoherence_real": -(
            df["traj_incoherence_neighbor_residual_top10_mean"].to_numpy(float)
            + df["traj_incoherence_crop_range"].to_numpy(float)
        ),
        "anti_traj_spatial_incoherence_real": (
            df["traj_incoherence_neighbor_residual_top10_mean"].to_numpy(float)
            + df["traj_incoherence_crop_range"].to_numpy(float)
        ),
        "fb_map_intensity_real": df["fb_uncertain_map_top10_mean"].to_numpy(float),
        "traj_map_smooth_real": -df["traj_incoherence_map_top10_mean"].to_numpy(float),
    }
    real_mask = df["subset"].astype(str).str.lower() == "real"
    for name in [
        "fb_uncertain_neighbor_residual_top10_mean",
        "traj_incoherence_neighbor_residual_top10_mean",
        "fb_uncertain_crop_range",
        "traj_incoherence_crop_range",
    ]:
        values = df[name].to_numpy(float)
        rv = values[real_mask.to_numpy()]
        med = float(np.median(rv))
        mad = float(np.median(np.abs(rv - med)))
        scale = max(1.4826 * mad, 1e-8)
        scores[f"{name}_centerdev_real"] = -np.abs(values - med) / scale
    return scores


def _metrics(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    real = df[df["subset"].astype(str).str.lower() == "real"]
    fake = df[df["subset"].astype(str).str.lower() != "real"]
    rows = []
    for source, group in fake.groupby("source_model", sort=True):
        y = np.concatenate([np.ones(len(real)), np.zeros(len(group))])
        s = np.concatenate([real[score_col].to_numpy(float), group[score_col].to_numpy(float)])
        rows.append({
            "source_model": source,
            "n_real": int(len(real)),
            "n_fake": int(len(group)),
            "auc": float(roc_auc_score(y, s)),
            "ap": float(average_precision_score(y, s)),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out.loc[len(out)] = {
            "source_model": "Average",
            "n_real": int(round(out["n_real"].mean())),
            "n_fake": int(round(out["n_fake"].mean())),
            "auc": float(out["auc"].mean()),
            "ap": float(out["ap"].mean()),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--patch-cache", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--duration", type=int, default=2)
    parser.add_argument("--debug", type=int, default=None)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--top-m", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda-dist", type=float, default=0.01)
    parser.add_argument("--match-mode", choices=["hard", "soft"], default="soft")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.debug is not None:
        df = (
            df.groupby(["subset", "source_model"], group_keys=False)
            .apply(lambda g: g.head(args.debug))
            .reset_index(drop=True)
        )

    rows = []
    cache_root = Path(args.patch_cache)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Motion local incoherence"):
        rowd = row.to_dict()
        if _is_missing(rowd.get(f"{args.duration}_sec_idxs")):
            continue
        path = _cache_path(cache_root, str(rowd["subset"]), str(rowd["source_model"]), str(rowd["video_path"]), args.duration)
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=True)
        grid_size = tuple(int(x) for x in payload["grid_size"])
        maps = _motion_maps(
            payload["patch"].numpy(),
            grid_size=grid_size,
            radius=args.radius,
            top_m=args.top_m,
            temperature=args.temperature,
            lambda_dist=args.lambda_dist,
            mode=args.match_mode,
        )
        feats = {}
        for name, values in maps.items():
            feats.update(_local_pattern_features(name, values, grid_size))
        feats.update({
            "subset": rowd["subset"],
            "source_model": rowd["source_model"],
            "filename": Path(str(rowd["video_path"])).name,
            "match_mode": args.match_mode,
            "radius": args.radius,
            "top_m": args.top_m,
            "temperature": args.temperature,
            "lambda_dist": args.lambda_dist,
        })
        rows.append(feats)

    out = pd.DataFrame(rows)
    for mode, scores in _score_modes(out).items():
        out[f"score_{mode}"] = scores
    out["final_score"] = out["score_fb_spatial_incoherence_real"]
    out["score_mode"] = "fb_spatial_incoherence_real"
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    score_cols = [c for c in out.columns if c.startswith("score_") and c != "score_mode"]
    for col in score_cols:
        print(f"\n== {col.removeprefix('score_')} ==")
        print(_metrics(out, col).to_string(index=False, formatters={
            "auc": lambda x: f"{x:.4f}",
            "ap": lambda x: f"{x:.4f}",
        }))
    print(f"Saved -> {output}")


if __name__ == "__main__":
    main()
