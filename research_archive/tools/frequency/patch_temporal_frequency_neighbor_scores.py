#!/usr/bin/env python3
"""Training-free neighbor temporal-frequency scores from cached patch tokens.

This probe follows the local temporal-frequency artifact idea, but works on
existing DINO patch-token trajectories instead of pixels. It measures whether
neighboring patches have inconsistent temporal spectra across time.
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


def _topk_stats(prefix: str, values: np.ndarray, ratios: tuple[float, ...] = (0.03, 0.05, 0.10, 0.20)) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    out: dict[str, float] = {}
    if arr.size == 0:
        for ratio in ratios:
            out[f"{prefix}_top{int(round(ratio * 100)):02d}_mean"] = 0.0
        return out
    order = np.sort(arr)[::-1]
    for ratio in ratios:
        k = max(1, int(round(arr.size * ratio)))
        out[f"{prefix}_top{int(round(ratio * 100)):02d}_mean"] = float(order[:k].mean())
    return out


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


def _band_power(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-patch temporal band distribution and total non-DC power.

    Input is [T, P, D]. Output band_power is [P, 3] for low/mid/high temporal
    frequencies, normalized per patch.
    """
    x = arr.astype(np.float32, copy=False)
    if x.ndim != 3 or x.shape[0] < 4:
        p = x.shape[1] if x.ndim == 3 else 0
        return np.zeros((p, 3), dtype=np.float32), np.zeros(p, dtype=np.float32)

    x = x - x.mean(axis=0, keepdims=True)
    fft = np.fft.rfft(x, axis=0)
    power = (fft.real**2 + fft.imag**2).astype(np.float32)[1:]  # [F, P, D]
    f = power.shape[0]
    if f == 0:
        return np.zeros((x.shape[1], 3), dtype=np.float32), np.zeros(x.shape[1], dtype=np.float32)
    pos = np.arange(1, f + 1, dtype=np.float32) / float(f)
    masks = [pos <= 0.34, (pos > 0.34) & (pos <= 0.67), pos > 0.67]
    parts = []
    for mask in masks:
        if mask.any():
            parts.append(power[mask].sum(axis=(0, 2)))
        else:
            parts.append(np.zeros(x.shape[1], dtype=np.float32))
    bands = np.stack(parts, axis=1).astype(np.float32)  # [P, 3]
    total = bands.sum(axis=1)
    dist = bands / (total[:, None] + 1e-9)
    return dist, total


def _frequency_neighbor_features(patch: np.ndarray, grid_size) -> dict[str, float]:
    arr = patch.astype(np.float32, copy=False)
    if arr.ndim != 3 or arr.shape[0] < 4:
        return _empty_features()

    height, width = _grid_hw(grid_size, arr.shape[1])
    pairs = _neighbor_pairs(height, width)
    if not pairs:
        return _empty_features()
    left = np.asarray([a for a, _ in pairs], dtype=np.int64)
    right = np.asarray([b for _, b in pairs], dtype=np.int64)

    band, total = _band_power(arr)
    delta_band, delta_total = _band_power(arr[1:] - arr[:-1])
    accel_band, accel_total = _band_power(arr[2:] - 2.0 * arr[1:-1] + arr[:-2])

    js = _jensen_shannon(band[left], band[right])
    delta_js = _jensen_shannon(delta_band[left], delta_band[right])
    accel_js = _jensen_shannon(accel_band[left], accel_band[right])
    high_gap = np.abs(band[left, 2] - band[right, 2])
    mid_gap = np.abs(band[left, 1] - band[right, 1])
    dom_flip = (np.argmax(band[left], axis=1) != np.argmax(band[right], axis=1)).astype(np.float32)
    total_gap = np.abs(np.log1p(total[left]) - np.log1p(total[right]))
    delta_total_gap = np.abs(np.log1p(delta_total[left]) - np.log1p(delta_total[right]))

    # Patch-level local spectrum isolation against four-neighbor mean.
    grid_band = band.reshape(height, width, 3)
    grid_total = np.log1p(total).reshape(height, width)
    isolation = []
    high_isolation = []
    total_isolation = []
    patch_isolation = np.zeros(arr.shape[1], dtype=np.float32)
    patch_high_isolation = np.zeros(arr.shape[1], dtype=np.float32)
    patch_total_isolation = np.zeros(arr.shape[1], dtype=np.float32)
    for y in range(height):
        for x in range(width):
            neigh = []
            neigh_total = []
            if x > 0:
                neigh.append(grid_band[y, x - 1])
                neigh_total.append(grid_total[y, x - 1])
            if x + 1 < width:
                neigh.append(grid_band[y, x + 1])
                neigh_total.append(grid_total[y, x + 1])
            if y > 0:
                neigh.append(grid_band[y - 1, x])
                neigh_total.append(grid_total[y - 1, x])
            if y + 1 < height:
                neigh.append(grid_band[y + 1, x])
                neigh_total.append(grid_total[y + 1, x])
            if not neigh:
                continue
            neigh_mean = np.stack(neigh, axis=0).mean(axis=0)
            center = grid_band[y, x]
            idx = y * width + x
            iso = float(_jensen_shannon(center[None, :], neigh_mean[None, :])[0])
            high_iso = float(abs(center[2] - neigh_mean[2]))
            total_iso = float(abs(grid_total[y, x] - np.mean(neigh_total)))
            isolation.append(iso)
            high_isolation.append(high_iso)
            total_isolation.append(total_iso)
            patch_isolation[idx] = iso
            patch_high_isolation[idx] = high_iso
            patch_total_isolation[idx] = total_iso

    patch_edge_count = np.zeros(arr.shape[1], dtype=np.float32)
    patch_js = np.zeros(arr.shape[1], dtype=np.float32)
    patch_delta_js = np.zeros(arr.shape[1], dtype=np.float32)
    patch_accel_js = np.zeros(arr.shape[1], dtype=np.float32)
    patch_high_gap = np.zeros(arr.shape[1], dtype=np.float32)
    patch_delta_total_gap = np.zeros(arr.shape[1], dtype=np.float32)
    patch_dom_flip = np.zeros(arr.shape[1], dtype=np.float32)
    for edge_idx, (a, b) in enumerate(pairs):
        for idx in (a, b):
            patch_edge_count[idx] += 1.0
            patch_js[idx] += js[edge_idx]
            patch_delta_js[idx] += delta_js[edge_idx]
            patch_accel_js[idx] += accel_js[edge_idx]
            patch_high_gap[idx] += high_gap[edge_idx]
            patch_delta_total_gap[idx] += delta_total_gap[edge_idx]
            patch_dom_flip[idx] += dom_flip[edge_idx]
    denom = np.maximum(patch_edge_count, 1.0)
    patch_js /= denom
    patch_delta_js /= denom
    patch_accel_js /= denom
    patch_high_gap /= denom
    patch_delta_total_gap /= denom
    patch_dom_flip /= denom

    patch_break = patch_js + 0.5 * patch_delta_js + 0.5 * patch_high_gap + 0.25 * patch_dom_flip
    patch_jitter = patch_delta_js + 0.5 * patch_accel_js + 0.5 * patch_delta_total_gap
    patch_consistency_break = patch_isolation + 0.5 * patch_high_isolation + 0.25 * patch_total_isolation
    patch_high_anomaly = band[:, 2] + 0.5 * patch_high_gap + 0.25 * patch_high_isolation

    out: dict[str, float] = {}
    for prefix, values in [
        ("tfreq_js", js),
        ("tfreq_delta_js", delta_js),
        ("tfreq_accel_js", accel_js),
        ("tfreq_high_gap", high_gap),
        ("tfreq_mid_gap", mid_gap),
        ("tfreq_dom_flip", dom_flip),
        ("tfreq_total_gap", total_gap),
        ("tfreq_delta_total_gap", delta_total_gap),
        ("tfreq_isolation", np.asarray(isolation, dtype=np.float32)),
        ("tfreq_high_isolation", np.asarray(high_isolation, dtype=np.float32)),
        ("tfreq_total_isolation", np.asarray(total_isolation, dtype=np.float32)),
        ("tfreq_patch_high", band[:, 2]),
        ("tfreq_patch_mid", band[:, 1]),
        ("tfreq_patch_break", patch_break),
        ("tfreq_patch_jitter", patch_jitter),
        ("tfreq_patch_consistency_break", patch_consistency_break),
        ("tfreq_patch_high_anomaly", patch_high_anomaly),
    ]:
        out.update(_stats(prefix, values))
    for prefix, values in [
        ("tfreq_patch_break", patch_break),
        ("tfreq_patch_jitter", patch_jitter),
        ("tfreq_patch_consistency_break", patch_consistency_break),
        ("tfreq_patch_high_anomaly", patch_high_anomaly),
    ]:
        out.update(_topk_stats(prefix, values))
    return out


def _empty_features() -> dict[str, float]:
    out: dict[str, float] = {}
    for prefix in [
        "tfreq_js",
        "tfreq_delta_js",
        "tfreq_accel_js",
        "tfreq_high_gap",
        "tfreq_mid_gap",
        "tfreq_dom_flip",
        "tfreq_total_gap",
        "tfreq_delta_total_gap",
        "tfreq_isolation",
        "tfreq_high_isolation",
        "tfreq_total_isolation",
        "tfreq_patch_high",
        "tfreq_patch_mid",
        "tfreq_patch_break",
        "tfreq_patch_jitter",
        "tfreq_patch_consistency_break",
        "tfreq_patch_high_anomaly",
    ]:
        out.update(_stats(prefix, np.zeros(0, dtype=np.float32)))
    for prefix in [
        "tfreq_patch_break",
        "tfreq_patch_jitter",
        "tfreq_patch_consistency_break",
        "tfreq_patch_high_anomaly",
    ]:
        out.update(_topk_stats(prefix, np.zeros(0, dtype=np.float32)))
    return out


def _jensen_shannon(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float32), 1e-9, None)
    q = np.clip(q.astype(np.float32), 1e-9, None)
    p = p / (p.sum(axis=1, keepdims=True) + 1e-9)
    q = q / (q.sum(axis=1, keepdims=True) + 1e-9)
    m = 0.5 * (p + q)
    kl_pm = (p * (np.log(p) - np.log(m))).sum(axis=1)
    kl_qm = (q * (np.log(q) - np.log(m))).sum(axis=1)
    return (0.5 * (kl_pm + kl_qm) / np.log(2.0)).astype(np.float32)


def _score_from_features(df: pd.DataFrame, mode: str) -> np.ndarray:
    """Realness scores. Higher should mean more real-like."""
    if mode == "anti_freq_break":
        return -(
            df["tfreq_js_p90"].to_numpy()
            + 0.5 * df["tfreq_delta_js_p90"].to_numpy()
            + 0.5 * df["tfreq_high_gap_p90"].to_numpy()
            + 0.25 * df["tfreq_dom_flip_mean"].to_numpy()
        )
    if mode == "freq_consistency":
        return -(
            df["tfreq_isolation_p90"].to_numpy()
            + 0.5 * df["tfreq_high_isolation_p90"].to_numpy()
            + 0.25 * df["tfreq_total_gap_p90"].to_numpy()
        )
    if mode == "freq_jitter_anomaly":
        return (
            df["tfreq_delta_js_p90"].to_numpy()
            + 0.5 * df["tfreq_accel_js_p90"].to_numpy()
            + 0.5 * df["tfreq_delta_total_gap_p90"].to_numpy()
        )
    if mode == "anti_freq_jitter":
        return -_score_from_features(df, "freq_jitter_anomaly")
    if mode == "freq_high_anomaly":
        return (
            df["tfreq_patch_high_p90"].to_numpy()
            + 0.5 * df["tfreq_high_gap_p90"].to_numpy()
            + 0.25 * df["tfreq_high_isolation_p90"].to_numpy()
        )
    if mode == "anti_highfreq_neighbor":
        return -_score_from_features(df, "freq_high_anomaly")
    if mode == "anti_patch_break_top05":
        return -df["tfreq_patch_break_top05_mean"].to_numpy()
    if mode == "anti_patch_jitter_top05":
        return -df["tfreq_patch_jitter_top05_mean"].to_numpy()
    if mode == "anti_patch_jitter_top10":
        return -df["tfreq_patch_jitter_top10_mean"].to_numpy()
    if mode == "anti_patch_consistency_break_top05":
        return -df["tfreq_patch_consistency_break_top05_mean"].to_numpy()
    if mode == "patch_high_anomaly_top05":
        return df["tfreq_patch_high_anomaly_top05_mean"].to_numpy()
    raise ValueError(f"Unknown score mode: {mode}")


def _available_score_modes() -> list[str]:
    return [
        "anti_freq_break",
        "freq_consistency",
        "freq_jitter_anomaly",
        "anti_freq_jitter",
        "freq_high_anomaly",
        "anti_highfreq_neighbor",
        "anti_patch_break_top05",
        "anti_patch_jitter_top05",
        "anti_patch_jitter_top10",
        "anti_patch_consistency_break_top05",
        "patch_high_anomaly_top05",
    ]


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
    parser.add_argument("--score-mode", default="anti_freq_break")
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
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Patch temporal frequency neighbor"):
        rowd = row.to_dict()
        if _is_missing(rowd.get(f"{args.duration}_sec_idxs")):
            continue
        path = _cache_path(cache_root, rowd["subset"], rowd["source_model"], rowd["video_path"], args.duration)
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=True)
        feats = _frequency_neighbor_features(payload["patch"].numpy(), payload.get("grid_size"))
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
