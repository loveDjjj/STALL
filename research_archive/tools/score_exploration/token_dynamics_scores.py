#!/usr/bin/env python3
"""Training-free token dynamics scores from cached Patch-STall embeddings."""

from __future__ import annotations

import argparse
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


def _safe_rank_ratio(x: np.ndarray) -> tuple[float, float]:
    """Return normalized spectral entropy and rank-1 energy ratio for [N, D].

    This is intended for small N. For patch trajectories, call it per patch
    over T frames instead of over T*P flattened tokens.
    """
    x = x.astype(np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    if x.shape[0] < 3:
        return 0.0, 1.0
    # Gram matrix is small because N is T or T*sampled_patches, avoiding D x D covariance.
    gram = x @ x.T / max(x.shape[1], 1)
    eig = np.linalg.eigvalsh((gram + gram.T) * 0.5)
    eig = np.clip(eig, 0, None)
    total = eig.sum() + 1e-9
    prob = eig / total
    entropy = float(-(prob * np.log(prob + 1e-12)).sum() / np.log(len(prob) + 1e-9))
    rank1 = float(eig.max() / total)
    return entropy, rank1


def _energy_stats(x: np.ndarray) -> dict[str, float]:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def _global_features(global_emb: np.ndarray) -> dict[str, float]:
    g = global_emb.astype(np.float32)
    delta = g[1:] - g[:-1] if len(g) > 1 else np.zeros((0, g.shape[-1]), np.float32)
    accel = g[2:] - 2 * g[1:-1] + g[:-2] if len(g) > 2 else np.zeros((0, g.shape[-1]), np.float32)
    entropy, rank1 = _safe_rank_ratio(g)
    out = {
        "global_temporal_entropy": entropy,
        "global_rank1_ratio": rank1,
        "global_var_mean": float(g.var(axis=0).mean()),
    }
    for prefix, vals in [
        ("global_delta_norm", np.linalg.norm(delta, axis=-1) if len(delta) else np.array([])),
        ("global_accel_norm", np.linalg.norm(accel, axis=-1) if len(accel) else np.array([])),
    ]:
        for k, v in _energy_stats(vals).items():
            out[f"{prefix}_{k}"] = v
    return out


def _patch_features(patch: np.ndarray, sample_patches: int | None = None) -> dict[str, float]:
    """Compute token dynamics features from patch trajectories [T, P, D]."""
    patch = patch.astype(np.float32)
    t, p, d = patch.shape
    if sample_patches is not None and p > sample_patches:
        idx = np.linspace(0, p - 1, sample_patches).round().astype(int)
        patch = patch[:, idx]
        p = patch.shape[1]

    delta = patch[1:] - patch[:-1] if t > 1 else np.zeros((0, p, d), np.float32)
    accel = patch[2:] - 2 * patch[1:-1] + patch[:-2] if t > 2 else np.zeros((0, p, d), np.float32)
    patch_var = patch.var(axis=0).mean(axis=-1)  # [P]
    patch_delta = np.linalg.norm(delta, axis=-1) if len(delta) else np.zeros((0, p), np.float32)
    patch_accel = np.linalg.norm(accel, axis=-1) if len(accel) else np.zeros((0, p), np.float32)

    # Estimate rank collapse per patch trajectory. This keeps the Gram matrix
    # at T x T (typically 16 x 16), avoiding a costly (T*P) x (T*P) eigendecomp.
    entropies = []
    rank1s = []
    for patch_id in range(p):
        entropy, rank1 = _safe_rank_ratio(patch[:, patch_id, :])
        entropies.append(entropy)
        rank1s.append(rank1)

    out = {
        "patch_temporal_entropy": float(np.mean(entropies) if entropies else 0.0),
        "patch_rank1_ratio": float(np.mean(rank1s) if rank1s else 1.0),
        "patch_temporal_entropy_p10": float(np.quantile(entropies, 0.10) if entropies else 0.0),
        "patch_rank1_ratio_p90": float(np.quantile(rank1s, 0.90) if rank1s else 1.0),
    }
    for prefix, vals in [
        ("patch_var", patch_var),
        ("patch_delta_norm", patch_delta),
        ("patch_accel_norm", patch_accel),
    ]:
        for k, v in _energy_stats(vals).items():
            out[f"{prefix}_{k}"] = v

    # Per-frame spatial diversity: fake videos can be overly homogeneous or unstable.
    frame_spatial_std = patch.std(axis=1).mean(axis=-1)  # [T]
    for k, v in _energy_stats(frame_spatial_std).items():
        out[f"frame_spatial_std_{k}"] = v
    return out


def _score_from_features(df: pd.DataFrame, mode: str) -> np.ndarray:
    """Training-free realness scores. Higher means more real-like."""
    if mode == "anti_collapse":
        return (
            df["patch_temporal_entropy"].to_numpy()
            - 0.5 * df["patch_rank1_ratio"].to_numpy()
            + 0.2 * df["patch_var_mean"].to_numpy()
        )
    if mode == "moderate_motion":
        return (
            -np.abs(df["patch_delta_norm_mean"].to_numpy() - df["patch_delta_norm_mean"].median())
            -0.5 * np.abs(df["patch_accel_norm_mean"].to_numpy() - df["patch_accel_norm_mean"].median())
        )
    if mode == "anti_smooth_dynamics":
        return (
            df["patch_delta_norm_p10"].to_numpy()
            + df["patch_accel_norm_p10"].to_numpy()
            + 0.5 * df["patch_temporal_entropy"].to_numpy()
            - 0.5 * df["patch_rank1_ratio"].to_numpy()
        )
    if mode == "stable_diverse":
        return (
            df["patch_temporal_entropy"].to_numpy()
            + 0.5 * df["frame_spatial_std_mean"].to_numpy()
            - 0.5 * df["patch_accel_norm_p90"].to_numpy()
        )
    raise ValueError(f"Unknown score mode: {mode}")


def _available_score_modes() -> list[str]:
    return ["anti_collapse", "moderate_motion", "anti_smooth_dynamics", "stable_diverse"]


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
                "auc_neg": roc_auc_score(y, -s),
            }
        )
    out = pd.DataFrame(rows)
    out.loc[len(out)] = {
        "source_model": "Average",
        "n_real": int(round(out["n_real"].mean())),
        "n_fake": int(round(out["n_fake"].mean())),
        "auc": out["auc"].mean(),
        "ap": out["ap"].mean(),
        "auc_neg": out["auc_neg"].mean(),
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
    parser.add_argument("--score-mode", default="anti_collapse")
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
    cache_root = Path(args.patch_cache)
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Token dynamics"):
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
        feats = _global_features(global_emb)
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
    modes = _available_score_modes() if args.score_all_modes else [args.score_mode]
    for mode in modes:
        out[f"score_{mode}"] = _score_from_features(out, mode)
    out["final_score"] = out[f"score_{args.score_mode}"]
    out["score_mode"] = args.score_mode
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    for mode in modes:
        print(f"\n== {mode} ==")
        metrics = _metrics(out, f"score_{mode}")
        print(metrics.to_string(index=False, formatters={"auc": lambda x: f"{x:.4f}", "ap": lambda x: f"{x:.4f}", "auc_neg": lambda x: f"{x:.4f}"}))


if __name__ == "__main__":
    main()
