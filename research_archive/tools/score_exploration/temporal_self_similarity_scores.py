#!/usr/bin/env python3
"""Compute temporal self-similarity anomaly scores from cached STALL embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or str(value) == "nan"


def _cache_path(cache_root: Path, subset: str, source_model: str, video_path: str, duration: int) -> Path:
    return cache_root / subset / source_model / f"{Path(video_path).stem}_{duration}s.pt"


def _normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-9)


def _matrix_features(seq: np.ndarray) -> dict[str, float]:
    """Features from temporal cosine self-similarity matrix.

    seq: [T, D]
    """
    seq = _normalize(seq.astype(np.float32), axis=-1)
    sim = seq @ seq.T
    t = sim.shape[0]
    if t < 3:
        return {k: 0.0 for k in [
            "diag1_mean", "diag1_std", "diag1_min", "diag2_mean", "diag_curv_mean",
            "offdiag_mean", "offdiag_std", "spectral_entropy", "rank1_ratio",
        ]}

    diag1 = np.diag(sim, k=1)
    diag2 = np.diag(sim, k=2) if t > 2 else diag1
    curvature = np.abs(np.diff(diag1)) if len(diag1) > 1 else np.array([0.0])
    off = sim[~np.eye(t, dtype=bool)]
    eig = np.linalg.eigvalsh((sim + sim.T) * 0.5)
    eig = np.clip(eig, 0, None)
    prob = eig / (eig.sum() + 1e-9)
    entropy = float(-(prob * np.log(prob + 1e-12)).sum() / np.log(len(prob) + 1e-9))
    return {
        "diag1_mean": float(diag1.mean()),
        "diag1_std": float(diag1.std()),
        "diag1_min": float(diag1.min()),
        "diag2_mean": float(diag2.mean()),
        "diag_curv_mean": float(curvature.mean()),
        "offdiag_mean": float(off.mean()),
        "offdiag_std": float(off.std()),
        "spectral_entropy": entropy,
        "rank1_ratio": float(eig.max() / (eig.sum() + 1e-9)),
    }


def _patch_features(patch: np.ndarray, sample_patches: int | None = None) -> dict[str, float]:
    """Aggregate self-similarity features over patch trajectories.

    patch: [T, P, D]
    """
    t, p, _ = patch.shape
    if sample_patches is not None and p > sample_patches:
        idx = np.linspace(0, p - 1, sample_patches).round().astype(int)
        patch = patch[:, idx]
        p = patch.shape[1]
    vals = []
    for i in range(p):
        vals.append(_matrix_features(patch[:, i, :]))
    df = pd.DataFrame(vals)
    out = {}
    for col in df.columns:
        arr = df[col].to_numpy(np.float32)
        out[f"patch_{col}_mean"] = float(arr.mean())
        out[f"patch_{col}_std"] = float(arr.std())
        out[f"patch_{col}_p10"] = float(np.quantile(arr, 0.10))
        out[f"patch_{col}_p90"] = float(np.quantile(arr, 0.90))
    return out


def _score_from_features(df: pd.DataFrame, mode: str) -> np.ndarray:
    """Training-free realness score from self-similarity features."""
    if mode == "global_diag":
        return df["global_diag1_mean"].to_numpy()
    if mode == "patch_diag":
        return df["patch_diag1_mean_mean"].to_numpy()
    if mode == "patch_stability":
        return (
            df["patch_diag1_mean_mean"].to_numpy()
            - 0.5 * df["patch_diag_curv_mean_mean"].to_numpy()
            - 0.2 * df["patch_diag1_std_mean"].to_numpy()
        )
    if mode == "hybrid_stability":
        return (
            0.4 * df["global_diag1_mean"].to_numpy()
            + 0.6 * df["patch_diag1_mean_mean"].to_numpy()
            - 0.4 * df["patch_diag_curv_mean_mean"].to_numpy()
        )
    if mode == "anti_smooth":
        return (
            -0.4 * df["global_diag1_mean"].to_numpy()
            -0.4 * df["patch_diag1_mean_mean"].to_numpy()
            +0.2 * df["patch_diag_curv_mean_mean"].to_numpy()
            +0.1 * df["global_diag_curv_mean"].to_numpy()
        )
    if mode == "anti_smooth_global":
        return -df["global_diag1_mean"].to_numpy() + 0.5 * df["global_diag_curv_mean"].to_numpy()
    if mode == "anti_smooth_patch":
        return -df["patch_diag1_mean_mean"].to_numpy() + 0.5 * df["patch_diag_curv_mean_mean"].to_numpy()
    raise ValueError(f"Unknown score mode: {mode}")


def _metrics(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    real = df[df["subset"].astype(str).str.lower() == "real"]
    fake = df[df["subset"].astype(str).str.lower() != "real"]
    rows = []
    for model, group in fake.groupby("source_model"):
        y = np.concatenate([np.ones(len(real)), np.zeros(len(group))])
        s = np.concatenate([real[score_col].to_numpy(), group[score_col].to_numpy()])
        rows.append(
            {
                "source_model": model,
                "n_real": len(real),
                "n_fake": len(group),
                "auc": roc_auc_score(y, s),
                "ap": average_precision_score(y, s),
            }
        )
    out = pd.DataFrame(rows)
    out.loc[len(out)] = {
        "source_model": "Average",
        "n_real": int(round(out["n_real"].mean())),
        "n_fake": int(round(out["n_fake"].mean())),
        "auc": out["auc"].mean(),
        "ap": out["ap"].mean(),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--patch-cache", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--duration", type=int, default=2)
    parser.add_argument("--debug", type=int, default=None)
    parser.add_argument("--sample-patches", type=int, default=None)
    parser.add_argument("--score-mode", default="hybrid_stability")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.debug:
        df = (
            df.groupby(["subset", "source_model"], group_keys=False)
            .apply(lambda g: g.head(args.debug))
            .reset_index(drop=True)
        )
    cache_root = Path(args.patch_cache)
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Self-sim"):
        rowd = row.to_dict()
        window = rowd.get(f"{args.duration}_sec_idxs")
        if _is_missing(window):
            continue
        path = _cache_path(cache_root, rowd["subset"], rowd["source_model"], rowd["video_path"], args.duration)
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=True)
        global_emb = payload["global"].numpy()
        patch_emb = payload["patch"].numpy()
        feats = {f"global_{k}": v for k, v in _matrix_features(global_emb).items()}
        feats.update(_patch_features(patch_emb, sample_patches=args.sample_patches))
        feats.update(
            {
                "subset": rowd["subset"],
                "source_model": rowd["source_model"],
                "filename": Path(rowd["video_path"]).name,
            }
        )
        rows.append(feats)

    out = pd.DataFrame(rows)
    out["final_score"] = _score_from_features(out, args.score_mode)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    metrics = _metrics(out, "final_score")
    print(metrics.to_string(index=False, formatters={"auc": lambda x: f"{x:.4f}", "ap": lambda x: f"{x:.4f}"}))


if __name__ == "__main__":
    main()
