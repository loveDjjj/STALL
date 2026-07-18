#!/usr/bin/env python3
"""LSMI-only PatchField scorer.

This is the spatial-motion part of `patchfield_split_scores.py`, kept separate
for full-scope sweeps where TTR is not needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from patchfield_split_scores import (
    _cache_path,
    _grid_hw,
    _is_missing,
    _neighbor_mean,
    _parse_fprs,
    _real_anomaly_rank,
    _source_metrics,
    _spatial_gradients,
    _stats,
)


def _lsmi_features(patch: np.ndarray, grid_size) -> dict[str, float]:
    arr = patch.astype(np.float32, copy=False)
    if arr.ndim != 3 or arr.shape[0] < 2:
        feats: dict[str, float] = {}
        for prefix in ("lsmi_residual", "lsmi_gradient", "motion_mag"):
            feats.update(_stats(prefix, np.zeros(0, dtype=np.float32)))
        feats["lsmi_anomaly_raw"] = 0.0
        return feats

    t_count, p_count, dim = arr.shape
    height, width = _grid_hw(grid_size, p_count)
    delta = arr[1:] - arr[:-1]
    motion = delta.reshape(t_count - 1, height, width, dim)
    neigh = _neighbor_mean(motion)
    residual = np.linalg.norm(motion - neigh, axis=-1) / (
        np.linalg.norm(motion, axis=-1) + np.linalg.norm(neigh, axis=-1) + 1e-6
    )
    motion_mag = np.linalg.norm(motion, axis=-1)
    gradients = _spatial_gradients(motion_mag)

    feats: dict[str, float] = {}
    feats.update(_stats("lsmi_residual", residual))
    feats.update(_stats("lsmi_gradient", gradients))
    feats.update(_stats("motion_mag", motion_mag))
    feats["lsmi_anomaly_raw"] = float(0.70 * feats["lsmi_residual_top10_mean"] + 0.30 * feats["lsmi_gradient_p95"])
    return feats


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
    for _, row in tqdm(df.iterrows(), total=len(df), desc="PF-LSMI"):
        rowd = row.to_dict()
        if _is_missing(rowd.get(f"{args.duration}_sec_idxs")):
            continue
        path = _cache_path(cache_root, str(rowd["subset"]), str(rowd["source_model"]), str(rowd["video_path"]), args.duration)
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=True)
        feats = _lsmi_features(payload["patch"].numpy(), payload.get("grid_size", None))
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
    scores["lsmi_anomaly_rank"] = _real_anomaly_rank(scores["lsmi_anomaly_raw"], real_mask)
    scores["score_lsmi_real"] = -scores["lsmi_anomaly_rank"]
    scores["score_lsmi_anti_real"] = scores["lsmi_anomaly_rank"]

    metrics = []
    for score_col in ("score_lsmi_real", "score_lsmi_anti_real"):
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
