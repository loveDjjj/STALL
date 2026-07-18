#!/usr/bin/env python3
"""Persistence features on real-calibrated patch likelihood maps.

This probe treats the calibrated patch map as a structured anomaly field rather
than a bag of low likelihood cells. After per-position real calibration, it
measures whether low-percentile cells persist through time, concentrate in
frames, or recur at the same patch positions.
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


def _jobs(csv_path: str, cache_root: str, duration: int, compact: bool) -> list[dict[str, object]]:
    df = load_csv(csv_path)
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


@torch.inference_mode()
def _temporal_ll(
    scorer: FastPatchScorer,
    patch_batch: torch.Tensor,
    patch_temp_mode: str,
    patch_region_size: int,
) -> np.ndarray:
    patch = patch_batch.float().to(scorer.device, non_blocking=True)
    _, _, mu_temp, W_temp = scorer._params_for_device(scorer.device)
    temp = scorer.temporal_features(patch, patch_temp_mode, patch_region_size)
    white = torch.matmul(temp - mu_temp, W_temp)
    return scorer.log_likelihood_from_white(white).detach().cpu().numpy().astype(np.float32)


def _append_real_cells(store: list[list[np.ndarray]] | None, ll: np.ndarray) -> list[list[np.ndarray]]:
    if store is None:
        store = [[] for _ in range(ll.shape[2])]
    for p in range(ll.shape[2]):
        store[p].append(ll[:, :, p].reshape(-1).astype(np.float32, copy=False))
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
    for p, real_vals in enumerate(calib):
        out[:, :, p] = np.searchsorted(real_vals, ll[:, :, p], side="right") / float(len(real_vals))
    return out


def _entropy_norm(counts: np.ndarray) -> float:
    counts = counts.astype(np.float64)
    total = counts.sum()
    if total <= 0 or len(counts) <= 1:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p + 1e-12)).sum() / np.log(len(counts)))


def _longest_true_run(mask: np.ndarray) -> int:
    # mask: [T, P]. Returns longest consecutive true run over all patches.
    if mask.shape[0] == 0:
        return 0
    cur = np.zeros(mask.shape[1], dtype=np.int32)
    best = 0
    for t in range(mask.shape[0]):
        cur = (cur + 1) * mask[t].astype(np.int32)
        best = max(best, int(cur.max(initial=0)))
    return best


def _features_for_map(pct: np.ndarray, thresholds: list[float]) -> dict[str, float]:
    # pct low means unusual low likelihood under real reference. Higher output
    # scores should mean more real, so anomaly mass features are negated.
    if pct.ndim != 2:
        raise ValueError(f"Expected [T,P] percentile map, got {pct.shape}")
    t_count, p_count = pct.shape
    out: dict[str, float] = {
        "temp_pct_mean_real": float(pct.mean()),
        "temp_pct_q05_real": float(np.quantile(pct, 0.05)),
        "temp_pct_q10_real": float(np.quantile(pct, 0.10)),
        "temp_pct_q20_real": float(np.quantile(pct, 0.20)),
    }
    for threshold in thresholds:
        mask = pct <= threshold
        tag = f"thr{str(threshold).replace('.', 'p')}"
        frame_counts = mask.sum(axis=1)
        patch_counts = mask.sum(axis=0)
        total = int(mask.sum())
        out[f"anom_mass_{tag}_real"] = float(-mask.mean())
        out[f"active_frame_frac_{tag}_real"] = float(-(frame_counts > 0).mean())
        out[f"active_patch_frac_{tag}_real"] = float(-(patch_counts > 0).mean())
        out[f"max_frame_mass_{tag}_real"] = float(-(frame_counts.max(initial=0) / max(p_count, 1)))
        out[f"max_patch_mass_{tag}_real"] = float(-(patch_counts.max(initial=0) / max(t_count, 1)))
        out[f"frame_entropy_{tag}_real"] = float(_entropy_norm(frame_counts))
        out[f"patch_entropy_{tag}_real"] = float(_entropy_norm(patch_counts))
        longest = _longest_true_run(mask)
        out[f"longest_patch_run_{tag}_real"] = float(-(longest / max(t_count, 1)))
        if total > 0:
            out[f"max_frame_share_{tag}_real"] = float(-(frame_counts.max(initial=0) / total))
            out[f"max_patch_share_{tag}_real"] = float(-(patch_counts.max(initial=0) / total))
        else:
            out[f"max_frame_share_{tag}_real"] = 0.0
            out[f"max_patch_share_{tag}_real"] = 0.0
    return out


def _rank01_from_real(df: pd.DataFrame, col: str) -> np.ndarray:
    real_mask = df["subset"].str.lower().eq("real").to_numpy()
    real = np.sort(df.loc[real_mask, col].to_numpy(dtype=np.float64))
    if len(real) == 0:
        raise ValueError("No real rows for score calibration")
    vals = df[col].to_numpy(dtype=np.float64)
    return np.searchsorted(real, vals, side="right") / float(len(real))


def _rank01_from_reference(df: pd.DataFrame, col: str, reference: np.ndarray) -> np.ndarray:
    if len(reference) == 0:
        raise ValueError(f"No reference rows for score calibration: {col}")
    ref = np.sort(reference.astype(np.float64, copy=False))
    vals = df[col].to_numpy(dtype=np.float64)
    return np.searchsorted(ref, vals, side="right") / float(len(ref))


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
    scorer = FastPatchScorer(args.patch_params, device=args.score_device)
    scorer.validate(args.patch_temp_mode)
    patch_region_size = args.patch_region_size or scorer.params_patch_region_size
    if patch_region_size != scorer.params_patch_region_size:
        raise ValueError(
            f"Patch params expect patch_region_size={scorer.params_patch_region_size}, got {patch_region_size}"
        )
    jobs = _jobs(args.csv, args.patch_emb_cache, args.duration, args.compact)
    calibration_duration = args.calibration_duration or args.duration
    calibration_cache = args.calibration_patch_emb_cache or args.patch_emb_cache
    calib_jobs = _jobs(args.calibration_csv or args.csv, calibration_cache, calibration_duration, args.compact)
    real_jobs = [job for job in calib_jobs if str(job["subset"]).lower() == "real"]
    if args.max_real_calib is not None:
        real_jobs = real_jobs[: args.max_real_calib]
    if not real_jobs:
        raise ValueError(
            "No real rows available for persistence calibration. "
            "Pass --calibration-csv with cached real rows when scoring an annotated-only target CSV."
        )
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]

    print(
        f"Calibrated persistence fit: real_videos={len(real_jobs)}, total_videos={len(jobs)}, "
        f"calibration_csv={args.calibration_csv or args.csv}, "
        f"calibration_patch_emb_cache={calibration_cache}, "
        f"target_duration={args.duration}, calibration_duration={calibration_duration}, "
        f"device={scorer.device}, batch={args.score_batch_size}, thresholds={thresholds}",
        flush=True,
    )
    store = None
    for start in tqdm(range(0, len(real_jobs), args.score_batch_size), desc="Fit temporal cell calib", unit="batch"):
        batch_jobs = real_jobs[start : start + args.score_batch_size]
        patch_batch, grid_size = load_cache_batch(batch_jobs)
        if grid_size != scorer.patch_grid_size:
            raise ValueError(f"Params grid_size={scorer.patch_grid_size}, cache grid_size={grid_size}")
        ll = _temporal_ll(scorer, patch_batch, args.patch_temp_mode, patch_region_size)
        store = _append_real_cells(store, ll)
    calib = _finalize_store(store, args.max_cells_per_position, args.seed)

    rows = []
    reference_rows = []
    for start in tqdm(range(0, len(jobs), args.score_batch_size), desc="Score persistence", unit="batch"):
        batch_jobs = jobs[start : start + args.score_batch_size]
        patch_batch, grid_size = load_cache_batch(batch_jobs)
        if grid_size != scorer.patch_grid_size:
            raise ValueError(f"Params grid_size={scorer.patch_grid_size}, cache grid_size={grid_size}")
        ll = _temporal_ll(scorer, patch_batch, args.patch_temp_mode, patch_region_size)
        pct = _percentile_by_position(ll, calib)
        for i, job in enumerate(batch_jobs):
            feats = _features_for_map(pct[i], thresholds)
            feats.update({col: job[col] for col in KEY_COLUMNS})
            rows.append(feats)

    if args.calibration_csv:
        for start in tqdm(range(0, len(real_jobs), args.score_batch_size), desc="Score real reference", unit="batch"):
            batch_jobs = real_jobs[start : start + args.score_batch_size]
            patch_batch, grid_size = load_cache_batch(batch_jobs)
            if grid_size != scorer.patch_grid_size:
                raise ValueError(f"Params grid_size={scorer.patch_grid_size}, cache grid_size={grid_size}")
            ll = _temporal_ll(scorer, patch_batch, args.patch_temp_mode, patch_region_size)
            pct = _percentile_by_position(ll, calib)
            for i, job in enumerate(batch_jobs):
                feats = _features_for_map(pct[i], thresholds)
                reference_rows.append(feats)

    out = pd.DataFrame(rows)
    feature_cols = [c for c in out.columns if c not in KEY_COLUMNS]
    for col in feature_cols:
        if args.calibration_csv:
            reference = np.array([row[col] for row in reference_rows], dtype=np.float64)
            out[f"{col}_pct_real"] = _rank01_from_reference(out, col, reference).astype(np.float32)
        else:
            out[f"{col}_pct_real"] = _rank01_from_real(out, col).astype(np.float32)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--calibration-csv", default=None)
    parser.add_argument("--calibration-duration", type=int, default=None)
    parser.add_argument("--patch-emb-cache", required=True)
    parser.add_argument("--calibration-patch-emb-cache", default=None)
    parser.add_argument("--patch-params", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--duration", type=int, default=2)
    parser.add_argument("--compact", action="store_true", default=False)
    parser.add_argument("--patch-temp-mode", default="same_grid_second_order")
    parser.add_argument("--patch-region-size", type=int, default=None)
    parser.add_argument("--score-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--score-batch-size", type=int, default=64)
    parser.add_argument("--thresholds", default="0.05,0.10,0.20")
    parser.add_argument("--max-real-calib", type=int, default=None)
    parser.add_argument("--max-cells-per-position", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = run(args)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"Saved calibrated persistence scores -> {output}")

    metric_rows = []
    if out["subset"].str.lower().nunique() >= 2:
        for col in [c for c in out.columns if c not in KEY_COLUMNS]:
            metrics = _metrics(out, col)
            if metrics.empty:
                continue
            avg = metrics[metrics["source_model"] == "Average"].iloc[0]
            metric_rows.append({"score_col": col, "auc": avg["auc"], "ap": avg["ap"]})
    metric_df = pd.DataFrame(metric_rows)
    if not metric_df.empty:
        metric_df = metric_df.sort_values(["auc", "ap"], ascending=False)
    metrics_path = output.with_name(output.stem + "_metrics.csv")
    metric_df.to_csv(metrics_path, index=False)
    if metric_df.empty:
        print("Skipped metrics: output contains fewer than two subset classes.")
    else:
        print(metric_df.head(30).to_string(index=False, formatters={"auc": lambda x: f"{x:.4f}", "ap": lambda x: f"{x:.4f}"}))
    print(f"Saved metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
