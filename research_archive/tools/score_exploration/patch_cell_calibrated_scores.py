#!/usr/bin/env python3
"""Patch-cell calibrated scoring probe.

The existing PatchSTALL score first aggregates raw patch likelihoods over a
video, then calibrates the video-level aggregate against real videos. This
probe changes the patch evidence itself: each patch-time likelihood cell is
converted to a real-only, position-conditioned percentile before video-level
aggregation. The goal is to reduce bias from naturally hard patch positions and
let bottom-k focus on cells that are rare for their own position.
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
    out: list[dict[str, object]] = []
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
        out.append(
            {
                "subset": str(row["subset"]),
                "source_model": str(row["source_model"]),
                "filename": Path(video_path).name,
                "cache_path": cache_path,
            }
        )
    return out


@torch.inference_mode()
def _ll_maps(
    scorer: FastPatchScorer,
    patch_batch: torch.Tensor,
    patch_temp_mode: str,
    patch_region_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    patch = patch_batch.float().to(scorer.device, non_blocking=True)
    mu_spat, W_spat, mu_temp, W_temp = scorer._params_for_device(scorer.device)

    spat_white = torch.matmul(patch - mu_spat, W_spat)
    spat_ll = scorer.log_likelihood_from_white(spat_white).detach().cpu().numpy()

    temp = scorer.temporal_features(patch, patch_temp_mode, patch_region_size)
    temp_white = torch.matmul(temp - mu_temp, W_temp)
    temp_ll = scorer.log_likelihood_from_white(temp_white).detach().cpu().numpy()
    return spat_ll.astype(np.float32), temp_ll.astype(np.float32)


def _append_real_cells(
    store: list[list[np.ndarray]] | None,
    ll: np.ndarray,
    max_cells_per_position: int | None,
    rng: np.random.RandomState,
) -> list[list[np.ndarray]]:
    if ll.ndim != 3:
        raise ValueError(f"Expected [N,T,P] likelihood map, got {ll.shape}")
    if store is None:
        store = [[] for _ in range(ll.shape[2])]
    for p in range(ll.shape[2]):
        vals = ll[:, :, p].reshape(-1)
        if max_cells_per_position is not None and len(vals) > max_cells_per_position:
            vals = rng.choice(vals, size=max_cells_per_position, replace=False)
        store[p].append(vals.astype(np.float32, copy=False))
    return store


def _finalize_store(store: list[list[np.ndarray]], max_cells_per_position: int | None, seed: int) -> list[np.ndarray]:
    rng = np.random.RandomState(seed)
    out: list[np.ndarray] = []
    for parts in store:
        vals = np.concatenate(parts).astype(np.float32, copy=False)
        if max_cells_per_position is not None and len(vals) > max_cells_per_position:
            vals = rng.choice(vals, size=max_cells_per_position, replace=False)
        out.append(np.sort(vals))
    return out


def _percentile_by_position(ll: np.ndarray, calib: list[np.ndarray]) -> np.ndarray:
    out = np.empty(ll.shape, dtype=np.float32)
    if ll.shape[2] != len(calib):
        raise ValueError(f"Likelihood P={ll.shape[2]} but calibration P={len(calib)}")
    for p, real_vals in enumerate(calib):
        out[:, :, p] = np.searchsorted(real_vals, ll[:, :, p], side="right") / float(len(real_vals))
    return out


def _bottomk_mean(arr: np.ndarray, ratio: float) -> np.ndarray:
    flat = arr.reshape(arr.shape[0], -1)
    k = max(1, int(np.ceil(flat.shape[1] * ratio)))
    part = np.partition(flat, kth=k - 1, axis=1)[:, :k]
    return part.mean(axis=1)


def _temporal_run_bottomk_mean(arr: np.ndarray, ratio: float, run_length: int) -> np.ndarray:
    if run_length <= 1 or arr.shape[1] < run_length:
        return _bottomk_mean(arr, ratio)
    windows = []
    for start in range(0, arr.shape[1] - run_length + 1):
        windows.append(arr[:, start : start + run_length, :].mean(axis=1))
    run_scores = np.stack(windows, axis=1)
    return _bottomk_mean(run_scores, ratio)


def _aggregate_percentiles(arr: np.ndarray, mode: str, ratio: float, run_length: int) -> np.ndarray:
    if mode == "mean":
        return arr.reshape(arr.shape[0], -1).mean(axis=1)
    if mode == "q05":
        return np.quantile(arr.reshape(arr.shape[0], -1), 0.05, axis=1)
    if mode == "bottomk_mean":
        return _bottomk_mean(arr, ratio)
    if mode == "temporal_run_bottomk_mean":
        return _temporal_run_bottomk_mean(arr, ratio, run_length)
    raise ValueError(f"Unknown aggregation mode: {mode}")


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
        }
    return out


def run(args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.RandomState(args.seed)
    scorer = FastPatchScorer(args.patch_params, device=args.score_device)
    scorer.validate(args.patch_temp_mode)
    patch_region_size = args.patch_region_size or scorer.params_patch_region_size
    if patch_region_size != scorer.params_patch_region_size:
        raise ValueError(
            f"Patch params expect patch_region_size={scorer.params_patch_region_size}, got {patch_region_size}"
        )

    jobs = _jobs(args.csv, args.patch_emb_cache, args.duration, args.compact, args.max_total, args.shuffle, args.seed)
    real_jobs = [job for job in jobs if str(job["subset"]).lower() == "real"]
    if args.max_real_calib is not None:
        real_jobs = real_jobs[: args.max_real_calib]
    if not real_jobs:
        raise ValueError("No real cached jobs available for cell calibration")

    print(
        f"Patch-cell calibration fit: real_videos={len(real_jobs)}, device={scorer.device}, "
        f"batch={args.score_batch_size}, mode={args.patch_temp_mode}, region={patch_region_size}",
        flush=True,
    )
    spat_store = None
    temp_store = None
    for start in tqdm(range(0, len(real_jobs), args.score_batch_size), desc="Fit cell calib", unit="batch"):
        batch_jobs = real_jobs[start : start + args.score_batch_size]
        patch_batch, grid_size = load_cache_batch(batch_jobs)
        if grid_size != scorer.patch_grid_size:
            raise ValueError(f"Params grid_size={scorer.patch_grid_size}, cache grid_size={grid_size}")
        spat_ll, temp_ll = _ll_maps(scorer, patch_batch, args.patch_temp_mode, patch_region_size)
        spat_store = _append_real_cells(spat_store, spat_ll, args.max_cells_per_position, rng)
        temp_store = _append_real_cells(temp_store, temp_ll, args.max_cells_per_position, rng)

    calib_spat = _finalize_store(spat_store, args.max_cells_per_position, args.seed + 1000)
    calib_temp = _finalize_store(temp_store, args.max_cells_per_position, args.seed + 2000)

    ratios = [float(x) for x in args.ratios.split(",") if x.strip()]
    modes = [x.strip() for x in args.aggregation_modes.split(",") if x.strip()]
    weight_pairs = []
    for item in args.weight_pairs.split(","):
        if not item.strip():
            continue
        a, b = item.split(":")
        weight_pairs.append((float(a), float(b)))

    rows = []
    print(
        f"Patch-cell calibrated scoring: videos={len(jobs)}, ratios={ratios}, modes={modes}, weights={weight_pairs}",
        flush=True,
    )
    for start in tqdm(range(0, len(jobs), args.score_batch_size), desc="Score cell-calib", unit="batch"):
        batch_jobs = jobs[start : start + args.score_batch_size]
        patch_batch, grid_size = load_cache_batch(batch_jobs)
        if grid_size != scorer.patch_grid_size:
            raise ValueError(f"Params grid_size={scorer.patch_grid_size}, cache grid_size={grid_size}")
        spat_ll, temp_ll = _ll_maps(scorer, patch_batch, args.patch_temp_mode, patch_region_size)
        spat_pct = _percentile_by_position(spat_ll, calib_spat)
        temp_pct = _percentile_by_position(temp_ll, calib_temp)

        score_values: dict[str, np.ndarray] = {}
        for mode in modes:
            for ratio in ratios:
                tag = f"{mode}_r{str(ratio).replace('.', 'p')}"
                spat_score = _aggregate_percentiles(spat_pct, mode, ratio, args.temporal_run_length)
                temp_score = _aggregate_percentiles(temp_pct, mode, ratio, args.temporal_run_length)
                score_values[f"spat_{tag}"] = spat_score.astype(np.float32)
                score_values[f"temp_{tag}"] = temp_score.astype(np.float32)
                for sw, tw in weight_pairs:
                    denom = max(sw + tw, 1e-8)
                    score_values[f"cell_calib_sw{sw:g}_tw{tw:g}_{tag}"] = (
                        (sw * spat_score + tw * temp_score) / denom
                    ).astype(np.float32)

        for i, job in enumerate(batch_jobs):
            row = {col: job[col] for col in KEY_COLUMNS}
            for col, values in score_values.items():
                row[col] = float(values[i])
            rows.append(row)
    return pd.DataFrame(rows)


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
    parser.add_argument("--ratios", default="0.05,0.10,0.20,0.50")
    parser.add_argument("--aggregation-modes", default="bottomk_mean,temporal_run_bottomk_mean,q05,mean")
    parser.add_argument("--temporal-run-length", type=int, default=3)
    parser.add_argument("--weight-pairs", default="0.7:0.3,0.5:0.5,0.3:0.7")
    parser.add_argument("--max-real-calib", type=int, default=None)
    parser.add_argument("--max-cells-per-position", type=int, default=20000)
    parser.add_argument("--max-total", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = run(args)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"Saved patch-cell calibrated scores -> {output}")

    metric_rows = []
    score_cols = [c for c in out.columns if c not in KEY_COLUMNS]
    for col in score_cols:
        metrics = _metrics(out, col)
        if metrics.empty:
            continue
        avg = metrics[metrics["source_model"] == "Average"].iloc[0]
        metric_rows.append({"score_col": col, "auc": avg["auc"], "ap": avg["ap"]})
    metric_df = pd.DataFrame(metric_rows).sort_values(["auc", "ap"], ascending=False)
    metrics_path = output.with_name(output.stem + "_metrics.csv")
    metric_df.to_csv(metrics_path, index=False)
    print(metric_df.head(30).to_string(index=False, formatters={"auc": lambda x: f"{x:.4f}", "ap": lambda x: f"{x:.4f}"}))
    print(f"Saved metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
