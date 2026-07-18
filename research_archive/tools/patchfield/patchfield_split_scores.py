#!/usr/bin/env python3
"""SPLIT-inspired standalone patch-field scorer.

This intentionally avoids the current global/fallback/source-routing stack.
It computes two patch-token field diagnostics from cached patch embeddings:

  TTR  - two-step temporal roughness of patch trajectories
  LSMI - local spatial motion incoherence of the patch motion field

Scores are real-calibrated using only real videos from the evaluated set, then
fused multiplicatively. Higher final score means more real-like.
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


def _grid_hw(grid_size, patch_count: int) -> tuple[int, int]:
    if isinstance(grid_size, torch.Tensor):
        grid_size = grid_size.detach().cpu().tolist()
    if isinstance(grid_size, (list, tuple)) and len(grid_size) == 2:
        return int(grid_size[0]), int(grid_size[1])
    side = int(round(np.sqrt(patch_count)))
    if side * side == patch_count:
        return side, side
    return 1, patch_count


def _stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    keys = ("mean", "std", "p50", "p90", "p95", "top10_mean", "top20_mean")
    if arr.size == 0:
        return {f"{prefix}_{key}": 0.0 for key in keys}
    out = {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std()),
        f"{prefix}_p50": float(np.quantile(arr, 0.50)),
        f"{prefix}_p90": float(np.quantile(arr, 0.90)),
        f"{prefix}_p95": float(np.quantile(arr, 0.95)),
    }
    order = np.sort(arr)[::-1]
    for ratio in (0.10, 0.20):
        k = max(1, int(round(arr.size * ratio)))
        out[f"{prefix}_top{int(ratio * 100):02d}_mean"] = float(order[:k].mean())
    return out


def _neighbor_mean(x: np.ndarray) -> np.ndarray:
    # x: [T,H,W,C]
    total = np.zeros_like(x, dtype=np.float32)
    count = np.zeros(x.shape[:3], dtype=np.float32)
    total[:, :, 1:, :] += x[:, :, :-1, :]
    count[:, :, 1:] += 1
    total[:, :, :-1, :] += x[:, :, 1:, :]
    count[:, :, :-1] += 1
    total[:, 1:, :, :] += x[:, :-1, :, :]
    count[:, 1:, :] += 1
    total[:, :-1, :, :] += x[:, 1:, :, :]
    count[:, :-1, :] += 1
    return total / np.maximum(count[..., None], 1.0)


def _spatial_gradients(x: np.ndarray) -> np.ndarray:
    # x: [T,H,W], returns adjacent absolute differences.
    parts = []
    if x.shape[2] > 1:
        parts.append(np.abs(x[:, :, 1:] - x[:, :, :-1]).reshape(-1))
    if x.shape[1] > 1:
        parts.append(np.abs(x[:, 1:, :] - x[:, :-1, :]).reshape(-1))
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32)


def _feature_vector(patch: np.ndarray, grid_size) -> dict[str, float]:
    arr = patch.astype(np.float32, copy=False)
    if arr.ndim != 3 or arr.shape[0] < 3:
        feats: dict[str, float] = {}
        for prefix in ("ttr_accel", "ttr_contrast", "lsmi_residual", "lsmi_gradient", "motion_mag"):
            feats.update(_stats(prefix, np.zeros(0, dtype=np.float32)))
        feats["ttr_anomaly_raw"] = 0.0
        feats["lsmi_anomaly_raw"] = 0.0
        return feats

    t_count, p_count, dim = arr.shape
    height, width = _grid_hw(grid_size, p_count)
    delta = arr[1:] - arr[:-1]  # [T-1,P,D]
    delta_norm = np.linalg.norm(delta, axis=-1)
    accel = arr[2:] - 2.0 * arr[1:-1] + arr[:-2]
    accel_norm = np.linalg.norm(accel, axis=-1)

    # Two-step contrast: if two-step displacement is not explained by adjacent
    # one-step motion, the trajectory is locally rough.
    two_step = arr[2:] - arr[:-2]
    two_step_norm = np.linalg.norm(two_step, axis=-1)
    one_step_sum = delta_norm[1:] + delta_norm[:-1]
    ttr_contrast = accel_norm / (one_step_sum + 1e-6)
    ttr_accel_normed = accel_norm / (one_step_sum + 1e-6)

    motion = delta.reshape(t_count - 1, height, width, dim)
    neigh = _neighbor_mean(motion)
    residual = np.linalg.norm(motion - neigh, axis=-1) / (
        np.linalg.norm(motion, axis=-1) + np.linalg.norm(neigh, axis=-1) + 1e-6
    )
    motion_mag = np.linalg.norm(motion, axis=-1)
    gradients = _spatial_gradients(motion_mag)

    feats: dict[str, float] = {}
    feats.update(_stats("ttr_accel", ttr_accel_normed))
    feats.update(_stats("ttr_contrast", ttr_contrast))
    feats.update(_stats("lsmi_residual", residual))
    feats.update(_stats("lsmi_gradient", gradients))
    feats.update(_stats("motion_mag", motion_mag))
    feats["ttr_anomaly_raw"] = float(0.60 * feats["ttr_accel_top10_mean"] + 0.40 * feats["ttr_contrast_p95"])
    feats["lsmi_anomaly_raw"] = float(0.70 * feats["lsmi_residual_top10_mean"] + 0.30 * feats["lsmi_gradient_p95"])
    return feats


def _real_anomaly_rank(values: pd.Series, real_mask: pd.Series) -> np.ndarray:
    real_values = np.sort(values[real_mask].to_numpy(float))
    if len(real_values) == 0:
        raise ValueError("Need at least one real row for real-only calibration")
    ranks = np.searchsorted(real_values, values.to_numpy(float), side="right") / float(len(real_values))
    return ranks.astype(float)


def _auc_ap(real_scores: pd.Series, fake_scores: pd.Series) -> tuple[float, float]:
    y = np.r_[np.ones(len(real_scores), dtype=int), np.zeros(len(fake_scores), dtype=int)]
    s = np.r_[real_scores.to_numpy(float), fake_scores.to_numpy(float)]
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def _fake_recall_at_fpr(real_scores: pd.Series, fake_scores: pd.Series, fpr: float) -> float:
    # Higher score means more real-like. Reject as fake when score < threshold.
    real_values = np.sort(real_scores.to_numpy(float))
    if len(real_values) == 0 or len(fake_scores) == 0:
        return float("nan")
    idx = int(np.floor(fpr * len(real_values)))
    idx = min(max(idx, 0), len(real_values) - 1)
    threshold = real_values[idx]
    return float((fake_scores.to_numpy(float) < threshold).mean())


def _source_metrics(df: pd.DataFrame, score_col: str, fprs: list[float]) -> pd.DataFrame:
    real = df[df["subset"].astype(str).str.lower() == "real"]
    fake = df[df["subset"].astype(str).str.lower() != "real"]
    rows = []
    for source, group in fake.groupby("source_model", sort=True):
        auc, ap = _auc_ap(real[score_col], group[score_col])
        row = {
            "source_model": source,
            "n_real": int(len(real)),
            "n_fake": int(len(group)),
            "auc": auc,
            "ap": ap,
        }
        for fpr in fprs:
            tag = str(fpr).replace(".", "p")
            row[f"fake_recall_at_fpr_{tag}"] = _fake_recall_at_fpr(real[score_col], group[score_col], fpr)
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        avg = {
            "source_model": "Average",
            "n_real": int(round(out["n_real"].mean())),
            "n_fake": int(round(out["n_fake"].mean())),
            "auc": float(out["auc"].mean()),
            "ap": float(out["ap"].mean()),
        }
        for fpr in fprs:
            tag = str(fpr).replace(".", "p")
            avg[f"fake_recall_at_fpr_{tag}"] = float(out[f"fake_recall_at_fpr_{tag}"].mean())
        out.loc[len(out)] = avg
    return out


def _parse_fprs(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(args.csv)
    if args.shuffle:
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    if args.debug is not None:
        df = (
            df.groupby(["subset", "source_model"], group_keys=False)
            .apply(lambda g: g.head(args.debug))
            .reset_index(drop=True)
        )
    if args.max_total is not None:
        df = df.head(args.max_total).reset_index(drop=True)

    rows = []
    cache_root = Path(args.patch_cache)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="PF-SPLIT"):
        rowd = row.to_dict()
        if _is_missing(rowd.get(f"{args.duration}_sec_idxs")):
            continue
        path = _cache_path(cache_root, str(rowd["subset"]), str(rowd["source_model"]), str(rowd["video_path"]), args.duration)
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=True)
        patch = payload["patch"].numpy()
        grid_size = payload.get("grid_size", None)
        feats = _feature_vector(patch, grid_size)
        feats.update(
            {
                "subset": str(rowd["subset"]),
                "source_model": str(rowd["source_model"]),
                "filename": Path(str(rowd["video_path"])).name,
            }
        )
        rows.append(feats)

    scores = pd.DataFrame(rows)
    if len(scores) == 0:
        raise ValueError("No patch cache rows were scored")
    real_mask = scores["subset"].astype(str).str.lower() == "real"
    scores["ttr_anomaly_rank"] = _real_anomaly_rank(scores["ttr_anomaly_raw"], real_mask)
    scores["lsmi_anomaly_rank"] = _real_anomaly_rank(scores["lsmi_anomaly_raw"], real_mask)
    scores["pfsplit_anomaly"] = (
        np.power(scores["ttr_anomaly_rank"].to_numpy(float), args.gamma_t)
        * np.power(scores["lsmi_anomaly_rank"].to_numpy(float), args.gamma_s)
    )
    scores["score_pfsplit_real"] = -scores["pfsplit_anomaly"]
    scores["score_ttr_real"] = -scores["ttr_anomaly_rank"]
    scores["score_lsmi_real"] = -scores["lsmi_anomaly_rank"]
    scores["score_pfsplit_anti_real"] = scores["pfsplit_anomaly"]
    scores["score_ttr_anti_real"] = scores["ttr_anomaly_rank"]
    scores["score_lsmi_anti_real"] = scores["lsmi_anomaly_rank"]
    metrics = []
    for score_col in (
        "score_pfsplit_real",
        "score_ttr_real",
        "score_lsmi_real",
        "score_pfsplit_anti_real",
        "score_ttr_anti_real",
        "score_lsmi_anti_real",
    ):
        m = _source_metrics(scores, score_col, args.fprs)
        m.insert(0, "score_col", score_col)
        metrics.append(m)
    return scores, pd.concat(metrics, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--patch-cache", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-summary-csv", required=True)
    parser.add_argument("--duration", type=int, default=2)
    parser.add_argument("--debug", type=int, default=None)
    parser.add_argument("--max-total", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma-t", type=float, default=1.0)
    parser.add_argument("--gamma-s", type=float, default=1.0)
    parser.add_argument("--fprs", type=_parse_fprs, default=_parse_fprs("0.001,0.005,0.01"))
    args = parser.parse_args()

    scores, summary = run(args)
    out = Path(args.output_csv)
    summary_out = Path(args.output_summary_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out, index=False)
    summary.to_csv(summary_out, index=False)
    avg = summary[summary["source_model"] == "Average"]
    print(avg.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved scores -> {out}")
    print(f"Saved summary -> {summary_out}")


if __name__ == "__main__":
    main()
