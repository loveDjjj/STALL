#!/usr/bin/env python3
"""Extract morphology features from patch temporal likelihood maps.

This changes how patch evidence is used: instead of only averaging the lowest
likelihood cells, it describes the shape of the anomalous cells over time and
space. The features are training-free and are calibrated with real-video ranks
for downstream fusion probes.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
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

from dataset_utils import _is_missing_window, load_csv  # noqa: E402
from dataset_utils_patch import _get_patch_cache_path  # noqa: E402
from eval_patch_fast import FastPatchScorer, load_cache_batch  # noqa: E402


KEY_COLUMNS = ["subset", "source_model", "filename"]


def _jobs(
    csv_path: str,
    cache_root: str,
    duration: int,
    compact: bool,
    max_total: int | None,
    shuffle: bool,
    seed: int,
) -> list[dict[str, object]]:
    df = load_csv(csv_path)
    if shuffle:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if max_total is not None:
        df = df.head(max_total).reset_index(drop=True)

    root = Path(cache_root)
    window_col = f"{duration}_sec_idxs"
    jobs: list[dict[str, object]] = []
    for _, row in df.iterrows():
        if compact and _is_missing_window(row.get(window_col)):
            continue
        video_path = str(row["video_path"])
        cache_path = _get_patch_cache_path(
            root,
            str(row["subset"]),
            str(row["source_model"]),
            Path(video_path).stem,
            duration,
            compact,
        )
        if not cache_path.exists():
            continue
        jobs.append(
            {
                "subset": str(row["subset"]),
                "source_model": str(row["source_model"]),
                "filename": Path(video_path).name,
                "cache_path": cache_path,
            }
        )
    return jobs


def _rank01_from_real(df: pd.DataFrame, col: str, real_mask: np.ndarray, high: bool) -> np.ndarray:
    real = np.sort(df.loc[real_mask, col].to_numpy(dtype=np.float64))
    if len(real) == 0:
        raise ValueError("No real rows available for rank calibration")
    vals = df[col].to_numpy(dtype=np.float64)
    pct = np.searchsorted(real, vals, side="right") / float(len(real))
    return pct if high else 1.0 - pct


@torch.inference_mode()
def _temporal_ll_batch(
    scorer: FastPatchScorer,
    patch_batch: torch.Tensor,
    patch_temp_mode: str,
    patch_region_size: int,
) -> torch.Tensor:
    patch = patch_batch.float().to(scorer.device, non_blocking=True)
    _, _, mu_temp, W_temp = scorer._params_for_device(scorer.device)
    temp = scorer.temporal_features(patch, patch_temp_mode, patch_region_size)
    white = torch.matmul(temp - mu_temp, W_temp)
    return scorer.log_likelihood_from_white(white).detach().cpu()


def _entropy_norm(counts: np.ndarray) -> float:
    counts = counts.astype(np.float64)
    total = counts.sum()
    if total <= 0 or len(counts) <= 1:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p + 1e-12)).sum() / np.log(len(counts)))


def _largest_component(mask: np.ndarray) -> tuple[int, int]:
    """Return largest and number of 6-connected components in [T,H,W]."""
    if mask.ndim != 3:
        raise ValueError(f"Expected [T,H,W] mask, got {mask.shape}")
    visited = np.zeros(mask.shape, dtype=bool)
    largest = 0
    components = 0
    t_count, h, w = mask.shape
    neighbors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    starts = np.argwhere(mask)
    for t, y, x in starts:
        if visited[t, y, x]:
            continue
        components += 1
        size = 0
        q: deque[tuple[int, int, int]] = deque([(int(t), int(y), int(x))])
        visited[t, y, x] = True
        while q:
            ct, cy, cx = q.popleft()
            size += 1
            for dt, dy, dx in neighbors:
                nt, ny, nx = ct + dt, cy + dy, cx + dx
                if 0 <= nt < t_count and 0 <= ny < h and 0 <= nx < w and mask[nt, ny, nx] and not visited[nt, ny, nx]:
                    visited[nt, ny, nx] = True
                    q.append((nt, ny, nx))
        largest = max(largest, size)
    return largest, components


def _bbox_area(mask2d: np.ndarray) -> float:
    ys, xs = np.where(mask2d)
    if len(ys) == 0:
        return 0.0
    return float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1) / mask2d.size)


def _features_for_one(
    ll: np.ndarray,
    grid_size: tuple[int, int],
    ratios: list[float],
    fast_shape_only: bool = False,
) -> dict[str, float]:
    if ll.ndim != 2:
        raise ValueError(f"Expected [T,P] ll, got {ll.shape}")
    t_count, p_count = ll.shape
    gh, gw = grid_size
    if gh * gw != p_count:
        # Region-pooled params may have a smaller grid than the raw cache.
        side = int(round(p_count ** 0.5))
        if side * side != p_count:
            raise ValueError(f"Cannot infer grid for P={p_count}")
        gh, gw = side, side

    out: dict[str, float] = {
        "ll_mean": float(ll.mean()),
        "ll_std": float(ll.std()),
        "ll_min": float(ll.min()),
        "ll_q05": float(np.quantile(ll, 0.05)),
        "ll_q20": float(np.quantile(ll, 0.20)),
        "frame_min_std": float(ll.min(axis=1).std()),
        "patch_min_std": float(ll.min(axis=0).std()),
    }
    flat = ll.reshape(-1)
    for ratio in ratios:
        k = max(1, int(np.ceil(flat.size * ratio)))
        idx = np.argpartition(flat, kth=k - 1)[:k]
        mask_flat = np.zeros(flat.size, dtype=bool)
        mask_flat[idx] = True
        mask = mask_flat.reshape(t_count, gh, gw)
        frame_counts = mask.reshape(t_count, -1).sum(axis=1)
        patch_counts = mask.sum(axis=0).reshape(-1)
        active_frames = frame_counts > 0
        active_patches = patch_counts > 0
        bbox_areas = [_bbox_area(mask[t]) for t in range(t_count) if frame_counts[t] > 0]
        tag = f"r{str(ratio).replace('.', 'p')}"
        out[f"selected_ll_mean_{tag}"] = float(flat[idx].mean())
        out[f"active_frame_frac_{tag}"] = float(active_frames.mean())
        out[f"active_patch_frac_{tag}"] = float(active_patches.mean())
        out[f"frame_entropy_{tag}"] = _entropy_norm(frame_counts)
        out[f"patch_entropy_{tag}"] = _entropy_norm(patch_counts)
        out[f"max_frame_mass_{tag}"] = float(frame_counts.max() / max(k, 1))
        out[f"max_patch_mass_{tag}"] = float(patch_counts.max() / max(k, 1))
        out[f"bbox_area_mean_{tag}"] = float(np.mean(bbox_areas) if bbox_areas else 0.0)
        largest_component_frac = 0.0
        if not fast_shape_only:
            largest, components = _largest_component(mask)
            largest_component_frac = float(largest / max(k, 1))
            out[f"largest_component_frac_{tag}"] = largest_component_frac
            out[f"component_count_norm_{tag}"] = float(components / max(k, 1))
        # Two signed, hypothesis-level realness scores. They are deliberately
        # simple; the detailed columns above are used to inspect direction.
        out[f"score_compact_anomaly_real_{tag}"] = float(
            -out[f"active_patch_frac_{tag}"]
            -0.5 * out[f"active_frame_frac_{tag}"]
            +0.5 * largest_component_frac
        )
        out[f"score_diffuse_natural_real_{tag}"] = float(
            out[f"active_patch_frac_{tag}"]
            +0.5 * out[f"active_frame_frac_{tag}"]
            +0.25 * out[f"patch_entropy_{tag}"]
            +0.25 * out[f"frame_entropy_{tag}"]
        )
    return out


def _metrics(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    real = df[df["subset"].str.lower() == "real"]
    fake = df[df["subset"].str.lower() != "real"]
    rows = []
    for model, group in fake.groupby("source_model", sort=True):
        y = np.concatenate([np.ones(len(real)), np.zeros(len(group))])
        s = np.concatenate([real[score_col].to_numpy(float), group[score_col].to_numpy(float)])
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
    if len(out):
        out.loc[len(out)] = {
            "source_model": "Average",
            "n_real": int(round(out["n_real"].mean())),
            "n_fake": int(round(out["n_fake"].mean())),
            "auc": out["auc"].mean(),
            "ap": out["ap"].mean(),
            "auc_neg": out["auc_neg"].mean(),
        }
    return out


def run(args: argparse.Namespace) -> pd.DataFrame:
    ratios = [float(x) for x in args.ratios.split(",") if x.strip()]
    scorer = FastPatchScorer(args.patch_params, device=args.score_device)
    scorer.validate(args.patch_temp_mode)
    region_size = args.patch_region_size or scorer.params_patch_region_size
    jobs = _jobs(args.csv, args.patch_emb_cache, args.duration, args.compact, args.max_total, args.shuffle, args.seed)
    rows = []
    print(
        f"Patch anomaly morphology: videos={len(jobs)}, device={scorer.device}, batch={args.score_batch_size}, "
        f"mode={args.patch_temp_mode}, region={region_size}, ratios={ratios}, "
        f"fast_shape_only={args.fast_shape_only}",
        flush=True,
    )
    for start in tqdm(range(0, len(jobs), args.score_batch_size), desc="Morphology", unit="batch"):
        batch_jobs = jobs[start : start + args.score_batch_size]
        patch_batch, _ = load_cache_batch(batch_jobs)
        ll_batch = _temporal_ll_batch(scorer, patch_batch, args.patch_temp_mode, region_size).numpy()
        p_count = ll_batch.shape[2]
        if region_size > 1:
            gh = scorer.patch_grid_size[0] // region_size
            gw = scorer.patch_grid_size[1] // region_size
        else:
            gh, gw = scorer.patch_grid_size
        if gh * gw != p_count:
            side = int(round(p_count ** 0.5))
            gh, gw = side, side
        for i, job in enumerate(batch_jobs):
            feats = _features_for_one(ll_batch[i], (gh, gw), ratios, fast_shape_only=args.fast_shape_only)
            feats.update({col: job[col] for col in KEY_COLUMNS})
            rows.append(feats)
    out = pd.DataFrame(rows)
    real_mask = out["subset"].str.lower().eq("real").to_numpy()
    feature_cols = [c for c in out.columns if c not in KEY_COLUMNS]
    for col in feature_cols:
        out[f"{col}_pct_high"] = _rank01_from_real(out, col, real_mask, high=True).astype(np.float32)
        out[f"{col}_pct_low"] = _rank01_from_real(out, col, real_mask, high=False).astype(np.float32)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--patch-emb-cache", required=True)
    parser.add_argument("--patch-params", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--duration", type=int, default=2)
    parser.add_argument("--compact", action="store_true", default=False)
    parser.add_argument("--patch-temp-mode", default="same_grid_second_order")
    parser.add_argument("--patch-region-size", type=int, default=None)
    parser.add_argument("--score-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--score-batch-size", type=int, default=32)
    parser.add_argument("--ratios", default="0.05,0.10,0.20")
    parser.add_argument("--max-total", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fast-shape-only",
        action="store_true",
        help="Skip connected-component features and keep vectorized shape statistics.",
    )
    args = parser.parse_args()

    out = run(args)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"Saved morphology scores -> {output}")
    score_cols = [c for c in out.columns if c.startswith("score_") and not c.endswith(("_pct_high", "_pct_low"))]
    score_cols += [c for c in out.columns if c.endswith("_pct_high") or c.endswith("_pct_low")]
    metric_rows = []
    for col in score_cols:
        m = _metrics(out, col)
        if len(m) == 0:
            continue
        avg = m[m["source_model"] == "Average"].iloc[0]
        metric_rows.append({"feature_score": col, "auc": avg["auc"], "ap": avg["ap"], "auc_neg": avg["auc_neg"]})
    metrics = pd.DataFrame(metric_rows).sort_values(["auc", "ap"], ascending=False)
    metrics_path = output.with_name(output.stem + "_metrics.csv")
    metrics.to_csv(metrics_path, index=False)
    print(metrics.head(20).to_string(index=False, formatters={"auc": lambda x: f"{x:.4f}", "ap": lambda x: f"{x:.4f}", "auc_neg": lambda x: f"{x:.4f}"}))
    print(f"Saved metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
