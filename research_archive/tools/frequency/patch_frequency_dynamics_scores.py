#!/usr/bin/env python3
"""Training-free frequency dynamics scores from cached patch embeddings.

The features treat each patch token as a short temporal trajectory and measure
how much of its energy lives in low, mid, and high temporal frequencies. This is
intended as a patch-signal probe, not as another global-score fusion.
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
        return {f"{prefix}_{k}": 0.0 for k in ["mean", "std", "p10", "p50", "p90"]}
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std()),
        f"{prefix}_p10": float(np.quantile(arr, 0.10)),
        f"{prefix}_p50": float(np.quantile(arr, 0.50)),
        f"{prefix}_p90": float(np.quantile(arr, 0.90)),
    }


def _sample_patch_axis(patch: np.ndarray, sample_patches: int | None) -> np.ndarray:
    if sample_patches is None or patch.shape[1] <= sample_patches:
        return patch
    idx = np.linspace(0, patch.shape[1] - 1, sample_patches).round().astype(int)
    return patch[:, idx]


def _band_masks(n_freq: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # rfft bins exclude DC before this helper is called. Split remaining bins by
    # normalized temporal frequency so the definition works for different T.
    if n_freq <= 0:
        empty = np.zeros(0, dtype=bool)
        return empty, empty, empty
    pos = np.arange(1, n_freq + 1, dtype=np.float32) / float(n_freq)
    low = pos <= 0.34
    mid = (pos > 0.34) & (pos <= 0.67)
    high = pos > 0.67
    return low, mid, high


def _frequency_features(seq: np.ndarray, prefix: str) -> dict[str, float]:
    """Compute temporal FFT features for [T, N, D] or [T, D]."""
    arr = seq.astype(np.float32)
    if arr.ndim == 2:
        arr = arr[:, None, :]
    t, n, d = arr.shape
    if t < 4:
        zero = np.zeros(n, dtype=np.float32)
        out = {}
        for name in ["low_ratio", "mid_ratio", "high_ratio", "high_low_ratio", "spectral_entropy"]:
            out.update(_stats(f"{prefix}_{name}", zero))
        return out

    arr = arr - arr.mean(axis=0, keepdims=True)
    fft = np.fft.rfft(arr, axis=0)
    power = (fft.real**2 + fft.imag**2).astype(np.float32)
    power = power[1:]  # drop DC
    total = power.sum(axis=(0, 2)) + 1e-9  # [N]
    low_mask, mid_mask, high_mask = _band_masks(power.shape[0])
    low = power[low_mask].sum(axis=(0, 2)) if low_mask.any() else np.zeros(n, dtype=np.float32)
    mid = power[mid_mask].sum(axis=(0, 2)) if mid_mask.any() else np.zeros(n, dtype=np.float32)
    high = power[high_mask].sum(axis=(0, 2)) if high_mask.any() else np.zeros(n, dtype=np.float32)

    band_power = np.stack([low, mid, high], axis=0)
    prob = band_power / (band_power.sum(axis=0, keepdims=True) + 1e-9)
    entropy = -(prob * np.log(prob + 1e-12)).sum(axis=0) / np.log(3.0)

    out = {}
    out.update(_stats(f"{prefix}_low_ratio", low / total))
    out.update(_stats(f"{prefix}_mid_ratio", mid / total))
    out.update(_stats(f"{prefix}_high_ratio", high / total))
    out.update(_stats(f"{prefix}_high_low_ratio", high / (low + 1e-9)))
    out.update(_stats(f"{prefix}_spectral_entropy", entropy))
    return out


def _difference_frequency_features(patch: np.ndarray, prefix: str) -> dict[str, float]:
    if patch.shape[0] < 5:
        return _frequency_features(patch[:0], prefix)
    delta = patch[1:] - patch[:-1]
    accel = patch[2:] - 2 * patch[1:-1] + patch[:-2]
    out = _frequency_features(delta, f"{prefix}_delta")
    out.update(_frequency_features(accel, f"{prefix}_accel"))
    return out


def _score_from_features(df: pd.DataFrame, mode: str) -> np.ndarray:
    """Realness scores. Higher should mean more real-like."""
    if mode == "anti_highfreq":
        return (
            -df["patch_high_ratio_mean"].to_numpy()
            -0.5 * df["patch_high_low_ratio_mean"].to_numpy()
            -0.25 * df["patch_delta_high_ratio_mean"].to_numpy()
        )
    if mode == "moderate_band":
        return (
            df["patch_mid_ratio_mean"].to_numpy()
            +0.25 * df["patch_spectral_entropy_mean"].to_numpy()
            -0.5 * np.abs(df["patch_high_ratio_mean"].to_numpy() - df["patch_high_ratio_mean"].median())
        )
    if mode == "anti_jitter_tail":
        return (
            -df["patch_high_ratio_p90"].to_numpy()
            -0.5 * df["patch_accel_high_ratio_p90"].to_numpy()
            -0.25 * df["patch_high_low_ratio_p90"].to_numpy()
        )
    if mode == "highfreq_anomaly":
        return (
            df["patch_high_ratio_mean"].to_numpy()
            +0.5 * df["patch_high_ratio_p90"].to_numpy()
            +0.25 * df["patch_accel_high_ratio_mean"].to_numpy()
        )
    raise ValueError(f"Unknown score mode: {mode}")


def _available_score_modes() -> list[str]:
    return ["anti_highfreq", "moderate_band", "anti_jitter_tail", "highfreq_anomaly"]


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
    parser.add_argument("--sample-patches", type=int, default=None)
    parser.add_argument("--score-mode", default="anti_jitter_tail")
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
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Patch frequency"):
        rowd = row.to_dict()
        if _is_missing(rowd.get(f"{args.duration}_sec_idxs")):
            continue
        path = _cache_path(cache_root, rowd["subset"], rowd["source_model"], rowd["video_path"], args.duration)
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=True)
        patch = _sample_patch_axis(payload["patch"].numpy(), args.sample_patches)
        global_emb = payload["global"].numpy()
        feats = {f"global_{k}": v for k, v in _frequency_features(global_emb, "freq").items()}
        feats.update(_frequency_features(patch, "patch"))
        feats.update(_difference_frequency_features(patch, "patch"))
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
