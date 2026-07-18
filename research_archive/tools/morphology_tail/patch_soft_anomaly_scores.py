#!/usr/bin/env python3
"""Soft anomaly-field features from real-calibrated patch temporal likelihoods."""

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

from patch_calibrated_persistence_scores import (  # noqa: E402
    KEY_COLUMNS,
    _finalize_store,
    _jobs,
    _percentile_by_position,
    _rank01_from_real,
    _rank01_from_reference,
    _temporal_ll,
)
from eval_patch_fast import FastPatchScorer, load_cache_batch  # noqa: E402


def _entropy_norm(weights: np.ndarray) -> float:
    weights = weights.astype(np.float64, copy=False)
    total = float(weights.sum())
    if total <= 1e-12 or len(weights) <= 1:
        return 0.0
    p = weights[weights > 0] / total
    return float(-(p * np.log(p + 1e-12)).sum() / np.log(len(weights)))


def _soft_longest_run(soft: np.ndarray, threshold: float = 0.5) -> float:
    """Soft run score: sum soft strength over longest above-threshold patch run."""
    if soft.shape[0] == 0:
        return 0.0
    active = soft >= threshold
    cur_len = np.zeros(soft.shape[1], dtype=np.int32)
    cur_sum = np.zeros(soft.shape[1], dtype=np.float32)
    best = 0.0
    for t in range(soft.shape[0]):
        cur_len = (cur_len + 1) * active[t].astype(np.int32)
        cur_sum = (cur_sum + soft[t]) * active[t].astype(np.float32)
        if cur_len.max(initial=0) > 0:
            best = max(best, float(cur_sum.max(initial=0)))
    return best / max(soft.shape[0], 1)


def _features_for_soft_map(pct: np.ndarray, taus: list[float], powers: list[float]) -> dict[str, float]:
    if pct.ndim != 2:
        raise ValueError(f"Expected [T,P] percentile map, got {pct.shape}")
    t_count, p_count = pct.shape
    eps = 1e-6
    surprise = -np.log(np.clip(pct, eps, 1.0))
    out: dict[str, float] = {
        "soft_surprise_mean_real": float(-surprise.mean()),
        "soft_surprise_q90_real": float(-np.quantile(surprise, 0.90)),
        "soft_surprise_q95_real": float(-np.quantile(surprise, 0.95)),
        "soft_surprise_max_real": float(-surprise.max(initial=0.0)),
    }
    for tau in taus:
        tau_tag = f"tau{str(tau).replace('.', 'p')}"
        base = np.clip((tau - pct) / max(tau, eps), 0.0, 1.0).astype(np.float32)
        for power in powers:
            soft = np.power(base, power).astype(np.float32)
            tag = f"{tau_tag}_p{str(power).replace('.', 'p')}"
            frame_mass = soft.sum(axis=1) / max(p_count, 1)
            patch_mass = soft.sum(axis=0) / max(t_count, 1)
            total = float(soft.sum())
            out[f"soft_mass_{tag}_real"] = float(-soft.mean())
            out[f"soft_frame_top1_{tag}_real"] = float(-frame_mass.max(initial=0.0))
            out[f"soft_patch_top1_{tag}_real"] = float(-patch_mass.max(initial=0.0))
            out[f"soft_frame_top20pct_{tag}_real"] = float(-np.quantile(frame_mass, 0.80))
            out[f"soft_patch_top20pct_{tag}_real"] = float(-np.quantile(patch_mass, 0.80))
            out[f"soft_frame_l2_{tag}_real"] = float(-np.sqrt(np.mean(frame_mass * frame_mass)))
            out[f"soft_patch_l2_{tag}_real"] = float(-np.sqrt(np.mean(patch_mass * patch_mass)))
            out[f"soft_frame_entropy_{tag}_real"] = float(_entropy_norm(frame_mass))
            out[f"soft_patch_entropy_{tag}_real"] = float(_entropy_norm(patch_mass))
            out[f"soft_longest_run_{tag}_real"] = float(-_soft_longest_run(soft))
            if total > 0:
                out[f"soft_frame_share_{tag}_real"] = float(-(frame_mass.max(initial=0.0) / total))
                out[f"soft_patch_share_{tag}_real"] = float(-(patch_mass.max(initial=0.0) / total))
            else:
                out[f"soft_frame_share_{tag}_real"] = 0.0
                out[f"soft_patch_share_{tag}_real"] = 0.0
    return out


def _features_for_soft_map_lite(pct: np.ndarray, taus: list[float]) -> dict[str, float]:
    if pct.ndim != 2:
        raise ValueError(f"Expected [T,P] percentile map, got {pct.shape}")
    eps = 1e-6
    surprise = -np.log(np.clip(pct, eps, 1.0))
    out: dict[str, float] = {
        "soft_lite_surprise_mean_real": float(-surprise.mean()),
        "soft_lite_surprise_q90_real": float(-np.quantile(surprise, 0.90)),
        "soft_lite_surprise_q95_real": float(-np.quantile(surprise, 0.95)),
        "soft_lite_surprise_max_real": float(-surprise.max(initial=0.0)),
    }
    for tau in taus:
        tag = f"tau{str(tau).replace('.', 'p')}"
        soft = np.clip((tau - pct) / max(tau, eps), 0.0, 1.0).astype(np.float32)
        frame_mass = soft.mean(axis=1)
        patch_mass = soft.mean(axis=0)
        out[f"soft_lite_mass_{tag}_real"] = float(-soft.mean())
        out[f"soft_lite_frame_top1_{tag}_real"] = float(-frame_mass.max(initial=0.0))
        out[f"soft_lite_patch_top1_{tag}_real"] = float(-patch_mass.max(initial=0.0))
        out[f"soft_lite_frame_q80_{tag}_real"] = float(-np.quantile(frame_mass, 0.80))
        out[f"soft_lite_patch_q80_{tag}_real"] = float(-np.quantile(patch_mass, 0.80))
        out[f"soft_lite_frame_l2_{tag}_real"] = float(-np.sqrt(np.mean(frame_mass * frame_mass)))
        out[f"soft_lite_patch_l2_{tag}_real"] = float(-np.sqrt(np.mean(patch_mass * patch_mass)))
    return out


def _append_real_cells(store: list[list[np.ndarray]] | None, ll: np.ndarray) -> list[list[np.ndarray]]:
    if store is None:
        store = [[] for _ in range(ll.shape[2])]
    for p in range(ll.shape[2]):
        store[p].append(ll[:, :, p].reshape(-1).astype(np.float32, copy=False))
    return store


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
                "auc": float(roc_auc_score(y, s)),
                "ap": float(average_precision_score(y, s)),
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
        }
    return out


def run(args: argparse.Namespace) -> pd.DataFrame:
    scorer = FastPatchScorer(args.patch_params, device=args.score_device)
    scorer.validate(args.patch_temp_mode)
    patch_region_size = args.patch_region_size or scorer.params_patch_region_size
    if patch_region_size != scorer.params_patch_region_size:
        raise ValueError(f"Patch params expect patch_region_size={scorer.params_patch_region_size}, got {patch_region_size}")

    target_jobs = _jobs(args.csv, args.patch_emb_cache, args.duration, args.compact)
    if args.debug_n is not None:
        grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
        for job in target_jobs:
            key = (str(job["subset"]), str(job["source_model"]))
            grouped.setdefault(key, []).append(job)
        target_jobs = [
            job
            for key in sorted(grouped)
            for job in grouped[key][: args.debug_n]
        ]
    calibration_duration = args.calibration_duration or args.duration
    calibration_cache = args.calibration_patch_emb_cache or args.patch_emb_cache
    calibration_jobs = _jobs(args.calibration_csv or args.csv, calibration_cache, calibration_duration, args.compact)
    real_jobs = [job for job in calibration_jobs if str(job["subset"]).lower() == "real"]
    if args.max_real_calib is not None:
        real_jobs = real_jobs[: args.max_real_calib]
    if not real_jobs:
        raise ValueError("No real rows available for soft anomaly calibration")

    print(
        f"Soft anomaly fit: real_videos={len(real_jobs)}, total_videos={len(target_jobs)}, "
        f"calibration_cache={calibration_cache}, device={scorer.device}, batch={args.score_batch_size}",
        flush=True,
    )
    store = None
    for start in tqdm(range(0, len(real_jobs), args.score_batch_size), desc="Fit soft calib", unit="batch"):
        batch_jobs = real_jobs[start : start + args.score_batch_size]
        patch_batch, grid_size = load_cache_batch(batch_jobs)
        if grid_size != scorer.patch_grid_size:
            raise ValueError(f"Params grid_size={scorer.patch_grid_size}, cache grid_size={grid_size}")
        ll = _temporal_ll(scorer, patch_batch, args.patch_temp_mode, patch_region_size)
        store = _append_real_cells(store, ll)
    calib = _finalize_store(store, args.max_cells_per_position, args.seed)

    taus = [float(x) for x in args.taus.split(",") if x.strip()]
    powers = [float(x) for x in args.powers.split(",") if x.strip()]
    rows = []
    reference_rows = []
    for start in tqdm(range(0, len(target_jobs), args.score_batch_size), desc="Score soft anomaly", unit="batch"):
        batch_jobs = target_jobs[start : start + args.score_batch_size]
        patch_batch, grid_size = load_cache_batch(batch_jobs)
        if grid_size != scorer.patch_grid_size:
            raise ValueError(f"Params grid_size={scorer.patch_grid_size}, cache grid_size={grid_size}")
        ll = _temporal_ll(scorer, patch_batch, args.patch_temp_mode, patch_region_size)
        pct = _percentile_by_position(ll, calib)
        for i, job in enumerate(batch_jobs):
            feats = (
                _features_for_soft_map_lite(pct[i], taus)
                if args.feature_mode == "lite"
                else _features_for_soft_map(pct[i], taus, powers)
            )
            feats.update({col: job[col] for col in KEY_COLUMNS})
            rows.append(feats)

    if args.calibration_csv:
        for start in tqdm(range(0, len(real_jobs), args.score_batch_size), desc="Reference soft anomaly", unit="batch"):
            batch_jobs = real_jobs[start : start + args.score_batch_size]
            patch_batch, grid_size = load_cache_batch(batch_jobs)
            if grid_size != scorer.patch_grid_size:
                raise ValueError(f"Params grid_size={scorer.patch_grid_size}, cache grid_size={grid_size}")
            ll = _temporal_ll(scorer, patch_batch, args.patch_temp_mode, patch_region_size)
            pct = _percentile_by_position(ll, calib)
            for i in range(len(batch_jobs)):
                reference_rows.append(
                    _features_for_soft_map_lite(pct[i], taus)
                    if args.feature_mode == "lite"
                    else _features_for_soft_map(pct[i], taus, powers)
                )

    out = pd.DataFrame(rows)
    feature_cols = [c for c in out.columns if c not in KEY_COLUMNS]
    calibrated_cols: dict[str, np.ndarray] = {}
    for col in feature_cols:
        if args.calibration_csv:
            ref = np.array([row[col] for row in reference_rows], dtype=np.float64)
            calibrated_cols[f"{col}_pct_real"] = _rank01_from_reference(out, col, ref).astype(np.float32)
        else:
            calibrated_cols[f"{col}_pct_real"] = _rank01_from_real(out, col).astype(np.float32)
    return pd.concat([out, pd.DataFrame(calibrated_cols)], axis=1)


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
    parser.add_argument("--debug-n", type=int, default=None)
    parser.add_argument("--taus", default="0.05,0.10,0.20,0.40")
    parser.add_argument("--powers", default="0.5,1.0,2.0")
    parser.add_argument("--feature-mode", choices=["lite", "full"], default="lite")
    parser.add_argument("--max-real-calib", type=int, default=None)
    parser.add_argument("--max-cells-per-position", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = run(args)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"Saved soft anomaly scores -> {output}")

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
