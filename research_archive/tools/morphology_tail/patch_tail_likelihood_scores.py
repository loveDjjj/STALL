#!/usr/bin/env python3
"""Fast spatial+temporal patch likelihood-tail scores."""

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

from eval_patch_fast import FastPatchScorer, iter_cache_jobs, load_cache_batch  # noqa: E402


KEY_COLUMNS = ["subset", "source_model", "filename"]


def _tail_stats(ll: torch.Tensor, ratios: list[float], prefix: str) -> dict[str, np.ndarray]:
    flat = ll.reshape(ll.shape[0], -1)
    out = {
        f"{prefix}_mean": flat.mean(dim=1).detach().cpu().numpy(),
        f"{prefix}_q05": torch.quantile(flat, 0.05, dim=1).detach().cpu().numpy(),
        f"{prefix}_q20": torch.quantile(flat, 0.20, dim=1).detach().cpu().numpy(),
        f"{prefix}_min": flat.amin(dim=1).detach().cpu().numpy(),
    }
    for ratio in ratios:
        k = max(1, int(np.ceil(flat.shape[1] * ratio)))
        vals = torch.topk(flat, k=k, largest=False, dim=1).values
        tag = f"bottom{int(round(ratio * 100)):02d}"
        out[f"{prefix}_{tag}_mean"] = vals.mean(dim=1).detach().cpu().numpy()
        out[f"{prefix}_{tag}_std"] = vals.std(dim=1).detach().cpu().numpy()
    return out


def _rank01(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref = np.sort(reference.astype(np.float64, copy=False))
    return np.searchsorted(ref, values.astype(np.float64, copy=False), side="right") / float(len(ref))


@torch.inference_mode()
def _score_batch(
    scorer: FastPatchScorer,
    patch_batch: torch.Tensor,
    patch_temp_mode: str,
    patch_region_size: int,
    ratios: list[float],
) -> dict[str, np.ndarray]:
    patch = patch_batch.float().to(scorer.device, non_blocking=True)
    mu_spat, W_spat, mu_temp, W_temp = scorer._params_for_device(scorer.device)
    spat_white = torch.matmul(patch - mu_spat, W_spat)
    spat_ll = scorer.log_likelihood_from_white(spat_white)
    temp = scorer.temporal_features(patch, patch_temp_mode, patch_region_size)
    temp_white = torch.matmul(temp - mu_temp, W_temp)
    temp_ll = scorer.log_likelihood_from_white(temp_white)
    out = {}
    out.update(_tail_stats(spat_ll, ratios, "spat_ll"))
    out.update(_tail_stats(temp_ll, ratios, "temp_ll"))
    for ratio in ratios:
        tag = f"bottom{int(round(ratio * 100)):02d}"
        out[f"joint_ll_{tag}_mean_10s90t"] = 0.1 * out[f"spat_ll_{tag}_mean"] + 0.9 * out[f"temp_ll_{tag}_mean"]
        out[f"joint_ll_{tag}_mean_30s70t"] = 0.3 * out[f"spat_ll_{tag}_mean"] + 0.7 * out[f"temp_ll_{tag}_mean"]
        out[f"joint_ll_{tag}_mean_50s50t"] = 0.5 * out[f"spat_ll_{tag}_mean"] + 0.5 * out[f"temp_ll_{tag}_mean"]
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
    ratios = [float(x) for x in args.ratios.split(",") if x.strip()]
    scorer = FastPatchScorer(args.patch_params, device=args.score_device)
    scorer.validate(args.patch_temp_mode)
    patch_region_size = args.patch_region_size or scorer.params_patch_region_size
    if patch_region_size != scorer.params_patch_region_size:
        raise ValueError(f"Patch params expect patch_region_size={scorer.params_patch_region_size}, got {patch_region_size}")

    jobs = list(iter_cache_jobs(args.csv, args.patch_emb_cache, args.duration, args.compact, args.debug_n))
    rows = []
    print(
        f"Patch tail likelihood scoring: videos={len(jobs)}, device={scorer.device}, "
        f"batch={args.score_batch_size}, ratios={ratios}",
        flush=True,
    )
    for start in tqdm(range(0, len(jobs), args.score_batch_size), desc="Patch tail", unit="batch"):
        batch_jobs = jobs[start : start + args.score_batch_size]
        patch_batch, grid_size = load_cache_batch(batch_jobs)
        if grid_size != scorer.patch_grid_size:
            raise ValueError(f"Params grid_size={scorer.patch_grid_size}, cache grid_size={grid_size}")
        scores = _score_batch(scorer, patch_batch, args.patch_temp_mode, patch_region_size, ratios)
        for i, job in enumerate(batch_jobs):
            row = {col: job[col] for col in KEY_COLUMNS}
            for key, vals in scores.items():
                row[key] = float(vals[i])
            rows.append(row)
    out = pd.DataFrame(rows)
    real_mask = out["subset"].str.lower().eq("real").to_numpy()
    feature_cols = [c for c in out.columns if c not in KEY_COLUMNS]
    rank_cols = {}
    for col in feature_cols:
        rank_cols[f"{col}_pct_real"] = _rank01(out[col].to_numpy(float), out.loc[real_mask, col].to_numpy(float)).astype(np.float32)
    out = pd.concat([out, pd.DataFrame(rank_cols)], axis=1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--patch-emb-cache", required=True)
    parser.add_argument("--patch-params", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--duration", type=int, default=2)
    parser.add_argument("--compact", action="store_true", default=False)
    parser.add_argument("--debug-n", type=int, default=None)
    parser.add_argument("--score-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--score-batch-size", type=int, default=128)
    parser.add_argument("--ratios", default="0.05,0.10,0.20,0.50")
    parser.add_argument("--patch-temp-mode", default="same_grid_second_order")
    parser.add_argument("--patch-region-size", type=int, default=None)
    args = parser.parse_args()

    out = run(args)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"Saved patch tail likelihood scores -> {output}")

    metric_rows = []
    for col in [c for c in out.columns if c not in KEY_COLUMNS]:
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
