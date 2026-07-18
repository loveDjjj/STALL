#!/usr/bin/env python3
"""Extract likelihood-level tail-shape features from patch temporal scores."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
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

from eval_patch_fast import FastPatchScorer, iter_cache_jobs, load_cache_batch


def _percentile(scores: np.ndarray, real_scores: np.ndarray) -> np.ndarray:
    sorted_real = np.sort(real_scores.astype(np.float64))
    return np.searchsorted(sorted_real, scores.astype(np.float64), side="right") / float(len(sorted_real))


def _tail_features_from_ll(ll: torch.Tensor, real_thresholds: dict[str, float] | None = None) -> dict[str, np.ndarray]:
    flat = ll.reshape(ll.shape[0], -1).detach().cpu().numpy().astype(np.float64)
    qs = np.quantile(flat, [0.01, 0.05, 0.10, 0.20, 0.50], axis=1).T
    q01, q05, q10, q20, q50 = [qs[:, i] for i in range(qs.shape[1])]

    sorted_flat = np.sort(flat, axis=1)
    n = sorted_flat.shape[1]
    k05 = max(1, int(np.ceil(n * 0.05)))
    k10 = max(1, int(np.ceil(n * 0.10)))
    k20 = max(1, int(np.ceil(n * 0.20)))
    k50 = max(1, int(np.ceil(n * 0.50)))
    mean_b05 = sorted_flat[:, :k05].mean(axis=1)
    mean_b10 = sorted_flat[:, :k10].mean(axis=1)
    mean_b20 = sorted_flat[:, :k20].mean(axis=1)
    mean_b50 = sorted_flat[:, :k50].mean(axis=1)

    out = {
        "ll_q01": q01,
        "ll_q05": q05,
        "ll_q10": q10,
        "ll_q20": q20,
        "ll_q50": q50,
        "ll_tail_gap_q20_q01": q20 - q01,
        "ll_tail_gap_q50_q05": q50 - q05,
        "ll_extreme_to_bulk_b05_b50": mean_b05 - mean_b50,
        "ll_extreme_to_bulk_b10_b50": mean_b10 - mean_b50,
        "ll_mean_bottom05": mean_b05,
        "ll_mean_bottom10": mean_b10,
        "ll_mean_bottom20": mean_b20,
        "ll_mean_bottom50": mean_b50,
        "ll_std": flat.std(axis=1),
    }
    if real_thresholds:
        for name, threshold in real_thresholds.items():
            out[f"ll_tail_mass_below_{name}"] = (flat < threshold).mean(axis=1)
    return out


def _tail_features_without_global_thresholds(ll: torch.Tensor) -> dict[str, np.ndarray]:
    return _tail_features_from_ll(ll, real_thresholds=None)


@torch.inference_mode()
def _temp_ll_batch(
    scorer: FastPatchScorer,
    patch_batch: torch.Tensor,
    patch_temp_mode: str,
    patch_region_size: int,
) -> torch.Tensor:
    patch = patch_batch
    if patch.dtype != torch.float32:
        patch = patch.float()
    patch = patch.to(scorer.device, non_blocking=True)
    _, _, mu_temp, W_temp = scorer._params_for_device(scorer.device)
    temp = scorer.temporal_features(patch, patch_temp_mode, patch_region_size)
    temp_white = torch.matmul(temp - mu_temp, W_temp)
    return scorer.log_likelihood_from_white(temp_white)


def _load_one_patch(job: dict) -> tuple[dict, torch.Tensor, tuple[int, int]]:
    payload = torch.load(job["cache_path"], weights_only=True, map_location="cpu")
    patch = payload["patch"]
    if patch.dtype != torch.float32:
        patch = patch.float()
    return job, patch, tuple(int(x) for x in payload["grid_size"])


def _load_cache_batch_parallel(jobs: list[dict], workers: int) -> tuple[list[dict], torch.Tensor, tuple[int, int]]:
    if workers <= 1:
        patch_batch, grid_size = load_cache_batch(jobs)
        return jobs, patch_batch, grid_size
    with ThreadPoolExecutor(max_workers=workers) as ex:
        loaded = list(ex.map(_load_one_patch, jobs))
    ordered_jobs = []
    patches = []
    grid_size = None
    shape = None
    for job, patch, this_grid in loaded:
        if grid_size is None:
            grid_size = this_grid
            shape = patch.shape
        elif grid_size != this_grid or patch.shape != shape:
            raise ValueError(
                f"Fast batching requires fixed grid/shape. Got {patch.shape}/{this_grid}, "
                f"expected {shape}/{grid_size} for {job['cache_path']}"
            )
        ordered_jobs.append(job)
        patches.append(patch)
    return ordered_jobs, torch.stack(patches, dim=0), grid_size


def _score_feature_directions(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, np.ndarray]:
    real_mask = df["subset"].str.lower().to_numpy() == "real"
    scores: dict[str, np.ndarray] = {}
    for col in feature_cols:
        values = df[col].to_numpy(dtype=np.float64)
        real_values = values[real_mask]
        pct = _percentile(values, real_values)
        scores[f"{col}_pct_high"] = pct
        scores[f"{col}_pct_low"] = 1.0 - pct
    return scores


def _average_metrics(df: pd.DataFrame, score_col: str) -> tuple[float, float]:
    real = df[df["subset"].str.lower() == "real"]
    fake = df[df["subset"].str.lower() != "real"]
    aucs = []
    aps = []
    for _, fake_group in fake.groupby("source_model", sort=True):
        y_true = np.concatenate([np.ones(len(real)), np.zeros(len(fake_group))])
        y_score = np.concatenate(
            [
                real[score_col].to_numpy(dtype=np.float64),
                fake_group[score_col].to_numpy(dtype=np.float64),
            ]
        )
        aucs.append(roc_auc_score(y_true, y_score))
        aps.append(average_precision_score(y_true, y_score))
    if not aucs:
        raise ValueError("No fake rows found")
    return float(np.mean(aucs)), float(np.mean(aps))


def run(args: argparse.Namespace) -> pd.DataFrame:
    scorer = FastPatchScorer(args.patch_params, device=args.score_device)
    scorer.validate(args.patch_temp_mode)
    patch_region_size = args.patch_region_size or scorer.params_patch_region_size
    if patch_region_size != scorer.params_patch_region_size:
        raise ValueError(
            f"Params expect patch_region_size={scorer.params_patch_region_size}, got {patch_region_size}"
        )

    jobs = list(iter_cache_jobs(args.csv, args.patch_emb_cache, args.duration, args.compact, args.debug_n))
    rows: list[dict[str, object]] = []
    real_flat_samples: list[np.ndarray] = []

    print(
        f"Extracting likelihood tail features: videos={len(jobs)}, batch={args.score_batch_size}, "
        f"mode={args.patch_temp_mode}, region={patch_region_size}",
        flush=True,
    )

    real_thresholds = None
    if not args.single_pass:
        # First pass: collect global real thresholds for tail-mass features.
        for start in tqdm(range(0, len(jobs), args.score_batch_size), desc="Tail threshold pass", unit="batch"):
            batch_jobs = jobs[start : start + args.score_batch_size]
            batch_jobs, patch_batch, grid_size = _load_cache_batch_parallel(batch_jobs, args.cache_workers)
            if grid_size != scorer.patch_grid_size:
                raise ValueError(f"Params grid_size={scorer.patch_grid_size}, cache grid_size={grid_size}")
            ll = _temp_ll_batch(scorer, patch_batch, args.patch_temp_mode, patch_region_size)
            flat = ll.reshape(ll.shape[0], -1).detach().cpu().numpy().astype(np.float64)
            for i, job in enumerate(batch_jobs):
                if str(job["subset"]).lower() == "real":
                    real_flat_samples.append(flat[i])

        if not real_flat_samples:
            raise ValueError("No real rows available for tail thresholds")
        real_all = np.concatenate(real_flat_samples, axis=0)
        real_thresholds = {
            "real_q01": float(np.quantile(real_all, 0.01)),
            "real_q05": float(np.quantile(real_all, 0.05)),
            "real_q10": float(np.quantile(real_all, 0.10)),
        }

    for start in tqdm(range(0, len(jobs), args.score_batch_size), desc="Tail feature pass", unit="batch"):
        batch_jobs = jobs[start : start + args.score_batch_size]
        batch_jobs, patch_batch, grid_size = _load_cache_batch_parallel(batch_jobs, args.cache_workers)
        if grid_size != scorer.patch_grid_size:
            raise ValueError(f"Params grid_size={scorer.patch_grid_size}, cache grid_size={grid_size}")
        ll = _temp_ll_batch(scorer, patch_batch, args.patch_temp_mode, patch_region_size)
        features = _tail_features_from_ll(ll, real_thresholds=real_thresholds)
        for i, job in enumerate(batch_jobs):
            row = {
                "subset": job["subset"],
                "source_model": job["source_model"],
                "filename": job["filename"],
                "patch_temp_mode": args.patch_temp_mode,
                "patch_region_size": patch_region_size,
            }
            for key, values in features.items():
                row[key] = float(values[i])
            rows.append(row)

    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c.startswith("ll_")]
    score_cols = _score_feature_directions(df, feature_cols)
    for col, values in score_cols.items():
        df[col] = values.astype(np.float32)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--patch-emb-cache", required=True)
    parser.add_argument("--patch-params", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--duration", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("--compact", action="store_true", default=False)
    parser.add_argument("--debug-n", type=int, default=None)
    parser.add_argument("--score-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--score-batch-size", type=int, default=64)
    parser.add_argument("--cache-workers", type=int, default=1)
    parser.add_argument("--single-pass", action="store_true", help="Skip global real tail-mass thresholds.")
    parser.add_argument("--patch-temp-mode", required=True)
    parser.add_argument("--patch-region-size", type=int, default=None)
    args = parser.parse_args()

    df = run(args)
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved tail features -> {out}", flush=True)

    score_cols = [c for c in df.columns if c.endswith("_pct_high") or c.endswith("_pct_low")]
    metric_rows = []
    for col in score_cols:
        auc, ap = _average_metrics(df, col)
        metric_rows.append({"feature_score": col, "auc": auc, "ap": ap})
    metric_df = pd.DataFrame(metric_rows).sort_values(["auc", "ap"], ascending=False)
    metric_path = out.with_name(out.stem + "_metrics.csv")
    metric_df.to_csv(metric_path, index=False)
    print(metric_df.head(12).to_string(index=False), flush=True)
    print(f"Saved feature metrics -> {metric_path}", flush=True)


if __name__ == "__main__":
    main()
