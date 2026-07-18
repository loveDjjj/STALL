#!/usr/bin/env python3
"""Patch scoring with reliability-weighted temporal aggregation.

This reuses existing patch whitening params and patch embedding caches. It
does not rerun DINO and does not refit whitening. The only changed component is
the aggregation of per-patch temporal likelihoods.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval_patch_fast import FastPatchScorer, iter_cache_jobs, load_cache_batch
from metrics import Score, ScoreDirection, get_results_df, print_results


def _weighted_bottomk_mean(
    ll: torch.Tensor,
    ratio: float,
    temporal_weight: float,
    spatial_weight: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reliability-weighted bottom-k over temporal likelihood tensor [N,T,P]."""
    if ll.ndim != 3:
        raise ValueError(f"Expected [N,T,P] likelihood tensor, got {tuple(ll.shape)}")

    # Anomaly strength: low likelihood is suspicious. Normalize per video so
    # reliability weights are scale-stable across datasets and configs.
    anomaly = -ll
    center = anomaly.mean(dim=(1, 2), keepdim=True)
    scale = anomaly.std(dim=(1, 2), keepdim=True).clamp_min(eps)
    z = (anomaly - center) / scale

    # Temporal stability: high mean anomaly with lower temporal variance is
    # treated as more reliable than an isolated spike.
    patch_mean = z.mean(dim=1)  # [N,P]
    patch_std = z.std(dim=1).clamp_min(eps)
    temporal_rel = torch.sigmoid(temporal_weight * patch_mean / (1.0 + patch_std))

    # Spatial coherence: reliable anomalous regions should agree with their
    # local neighborhood more than single noisy tokens do.
    if spatial_weight > 0.0:
        patch_count = ll.shape[2]
        side = int(round(patch_count ** 0.5))
        if side * side == patch_count:
            x = patch_mean.reshape(ll.shape[0], 1, side, side)
            neigh = torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
            neigh = neigh.reshape(ll.shape[0], patch_count)
            spatial_rel = torch.sigmoid(spatial_weight * neigh)
        else:
            spatial_rel = torch.ones_like(temporal_rel)
    else:
        spatial_rel = torch.ones_like(temporal_rel)

    reliability = (temporal_rel * spatial_rel).clamp_min(eps)  # [N,P]

    # Use reliability only to rank which low-likelihood patch/time cells are
    # trustworthy. Average the original likelihood values so the calibration
    # scale remains comparable to the existing patch params.
    rank_key = ll / reliability[:, None, :]
    flat_key = rank_key.reshape(ll.shape[0], -1)
    flat_ll = ll.reshape(ll.shape[0], -1)
    k = max(1, int(np.ceil(flat_key.shape[1] * ratio)))
    idx = torch.topk(flat_key, k=k, largest=False, dim=1).indices
    return torch.gather(flat_ll, dim=1, index=idx).mean(dim=1)


@torch.inference_mode()
def _raw_scores_for_batch(
    scorer: FastPatchScorer,
    patch_batch: torch.Tensor,
    patch_temp_mode: str,
    patch_region_size: int,
    bottomk_ratio: float,
    temporal_weight: float,
    spatial_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    patch = patch_batch
    if patch.dtype != torch.float32:
        patch = patch.float()
    patch = patch.to(scorer.device, non_blocking=True)
    mu_spat, W_spat, mu_temp, W_temp = scorer._params_for_device(scorer.device)

    spat_white = torch.matmul(patch - mu_spat, W_spat)
    spat_ll = scorer.log_likelihood_from_white(spat_white)
    # Keep spatial branch aggregation identical to the original config. The
    # experiment isolates whether temporal reliability improves patch evidence.
    spat_agg = scorer._aggregate(
        spat_ll,
        scorer.aggregation_config.get("mode", "bottomk_mean"),
        scorer.params_bottomk_ratio,
        scorer.params_temporal_run_length,
    )

    temp = scorer.temporal_features(patch, patch_temp_mode, patch_region_size)
    temp_white = torch.matmul(temp - mu_temp, W_temp)
    temp_ll = scorer.log_likelihood_from_white(temp_white)
    temp_agg = _weighted_bottomk_mean(
        temp_ll,
        ratio=bottomk_ratio,
        temporal_weight=temporal_weight,
        spatial_weight=spatial_weight,
    )
    return spat_agg.cpu().numpy(), temp_agg.cpu().numpy()


def _percentile(scores: np.ndarray, calib_sorted: np.ndarray) -> np.ndarray:
    return np.searchsorted(calib_sorted, scores, side="right") / float(len(calib_sorted))


def run(args: argparse.Namespace) -> pd.DataFrame:
    scorer = FastPatchScorer(args.patch_params, device=args.score_device)
    scorer.validate(args.patch_temp_mode)
    patch_region_size = args.patch_region_size or scorer.params_patch_region_size
    if patch_region_size != scorer.params_patch_region_size:
        raise ValueError(
            f"Params expect patch_region_size={scorer.params_patch_region_size}, got {patch_region_size}"
        )
    bottomk_ratio = args.bottomk_ratio if args.bottomk_ratio is not None else scorer.params_bottomk_ratio

    jobs = list(iter_cache_jobs(args.csv, args.patch_emb_cache, args.duration, args.compact, args.debug_n))
    raw_rows: list[dict[str, object]] = []
    real_spat: list[float] = []
    real_temp: list[float] = []

    print(
        f"Reliability-weighted patch scoring: videos={len(jobs)}, batch={args.score_batch_size}, "
        f"mode={args.patch_temp_mode}, region={patch_region_size}, bottomk={bottomk_ratio}, "
        f"temporal_weight={args.temporal_weight}, spatial_weight={args.spatial_weight}",
        flush=True,
    )

    for start in tqdm(range(0, len(jobs), args.score_batch_size), desc="Weighted patch scoring", unit="batch"):
        batch_jobs = jobs[start : start + args.score_batch_size]
        patch_batch, grid_size = load_cache_batch(batch_jobs)
        if grid_size != scorer.patch_grid_size:
            raise ValueError(f"Params grid_size={scorer.patch_grid_size}, cache grid_size={grid_size}")
        spat_raw, temp_raw = _raw_scores_for_batch(
            scorer=scorer,
            patch_batch=patch_batch,
            patch_temp_mode=args.patch_temp_mode,
            patch_region_size=patch_region_size,
            bottomk_ratio=bottomk_ratio,
            temporal_weight=args.temporal_weight,
            spatial_weight=args.spatial_weight,
        )
        for i, job in enumerate(batch_jobs):
            row = {
                "subset": job["subset"],
                "source_model": job["source_model"],
                "filename": job["filename"],
                "patch_spat_raw": float(spat_raw[i]),
                "patch_temp_raw": float(temp_raw[i]),
            }
            raw_rows.append(row)
            if str(job["subset"]).lower() == "real":
                real_spat.append(float(spat_raw[i]))
                real_temp.append(float(temp_raw[i]))

    if not real_spat or not real_temp:
        raise ValueError("No real rows found for weighted calibration")

    df = pd.DataFrame(raw_rows)
    calib_spat = np.sort(np.asarray(real_spat, dtype=np.float64))
    calib_temp = np.sort(np.asarray(real_temp, dtype=np.float64))
    spat_pct = _percentile(df["patch_spat_raw"].to_numpy(np.float64), calib_spat)
    temp_pct = _percentile(df["patch_temp_raw"].to_numpy(np.float64), calib_temp)
    denom = max(args.patch_spat_weight + args.patch_temp_weight, 1e-8)
    final = (args.patch_spat_weight * spat_pct + args.patch_temp_weight * temp_pct) / denom

    df["fusion"] = "patch_reliability_weighted"
    df["patch_spat_weight"] = args.patch_spat_weight
    df["patch_temp_weight"] = args.patch_temp_weight
    df["patch_temp_mode"] = args.patch_temp_mode
    df["aggregation"] = "reliability_weighted_bottomk_mean"
    df["bottomk_ratio"] = bottomk_ratio
    df["patch_region_size"] = patch_region_size
    df["temporal_reliability_weight"] = args.temporal_weight
    df["spatial_reliability_weight"] = args.spatial_weight
    df["patch_spat_percentile"] = spat_pct.astype(np.float32)
    df["patch_temp_percentile"] = temp_pct.astype(np.float32)
    df["patch_final_score"] = final.astype(np.float32)
    df["final_score"] = final.astype(np.float32)
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
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--patch-temp-mode", required=True)
    parser.add_argument("--patch-region-size", type=int, default=None)
    parser.add_argument("--bottomk-ratio", type=float, default=None)
    parser.add_argument("--patch-spat-weight", type=float, default=0.10)
    parser.add_argument("--patch-temp-weight", type=float, default=0.90)
    parser.add_argument("--temporal-weight", type=float, default=1.0)
    parser.add_argument("--spatial-weight", type=float, default=1.0)
    args = parser.parse_args()

    df = run(args)
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved per-video scores -> {out}", flush=True)
    results_df = get_results_df(
        df[["subset", "source_model"]],
        {
            "final_score": Score(
                value=df["final_score"].to_numpy(),
                direction=ScoreDirection.HIGHER_IS_REAL,
            )
        },
    )
    print_results(results_df)


if __name__ == "__main__":
    main()
