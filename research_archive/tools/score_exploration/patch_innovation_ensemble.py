#!/usr/bin/env python3
"""Patch-only ensemble over raw patch, soft anomaly, and token dynamics signals."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


KEY_COLUMNS = ["subset", "source_model", "filename"]


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in KEY_COLUMNS + columns if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    out = df[KEY_COLUMNS + columns].copy()
    for col in KEY_COLUMNS:
        out[col] = out[col].astype(str)
    return out


def _rank_by_real(df: pd.DataFrame, col: str) -> np.ndarray:
    real_mask = df["subset"].str.lower().eq("real").to_numpy()
    real = np.sort(df.loc[real_mask, col].to_numpy(dtype=np.float64))
    if len(real) == 0:
        raise ValueError(f"No real rows for calibration: {col}")
    values = df[col].to_numpy(dtype=np.float64)
    return np.searchsorted(real, values, side="right") / float(len(real))


def _metrics(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    real = df[df["subset"].str.lower() == "real"]
    fake = df[df["subset"].str.lower() != "real"]
    rows = []
    for source, group in fake.groupby("source_model", sort=True):
        y = np.concatenate([np.ones(len(real)), np.zeros(len(group))])
        s = np.concatenate([real[score_col].to_numpy(float), group[score_col].to_numpy(float)])
        rows.append(
            {
                "source_model": source,
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


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = _read(args.raw_patch_csv, [args.raw_patch_col]).rename(columns={args.raw_patch_col: "raw_patch"})
    soft = _read(args.soft_csv, [args.soft_col]).rename(columns={args.soft_col: "soft_signal"})
    token = _read(args.token_csv, [args.token_col]).rename(columns={args.token_col: "token_signal"})
    df = raw.merge(soft, on=KEY_COLUMNS, how="inner", validate="one_to_one").merge(
        token,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    df["raw_rank"] = _rank_by_real(df, "raw_patch").astype(np.float32)
    df["soft_rank"] = _rank_by_real(df, "soft_signal").astype(np.float32)
    df["token_rank"] = _rank_by_real(df, "token_signal").astype(np.float32)
    denom = max(args.raw_weight + args.soft_weight + args.token_weight, 1e-8)
    df["final_score"] = (
        args.raw_weight * df["raw_rank"]
        + args.soft_weight * df["soft_rank"]
        + args.token_weight * df["token_rank"]
    ) / denom
    df["patch_innovation_v1"] = df["final_score"]

    metric_frames = []
    summary_rows = []
    for score_col in ["raw_rank", "soft_rank", "token_rank", "final_score"]:
        metrics = _metrics(df, score_col)
        metric_frames.append(metrics.assign(score_col=score_col))
        avg = metrics[metrics["source_model"] == "Average"].iloc[0]
        summary_rows.append(
            {
                "score_col": score_col,
                "avg_auc": float(avg["auc"]),
                "avg_ap": float(avg["ap"]),
                "min_auc": float(metrics[metrics["source_model"] != "Average"]["auc"].min()),
                "min_ap": float(metrics[metrics["source_model"] != "Average"]["ap"].min()),
                "n_sources": int(len(metrics) - 1),
                "n_rows": int(len(df)),
                "raw_weight": args.raw_weight,
                "soft_weight": args.soft_weight,
                "token_weight": args.token_weight,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["avg_auc", "avg_ap"], ascending=False)
    per_source = pd.concat(metric_frames, ignore_index=True)
    return df, summary, per_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-patch-csv", type=Path, required=True)
    parser.add_argument("--soft-csv", type=Path, required=True)
    parser.add_argument("--token-csv", type=Path, required=True)
    parser.add_argument("--raw-patch-col", default="final_score")
    parser.add_argument("--soft-col", default="soft_lite_patch_top1_tau0p4_real_pct_real")
    parser.add_argument("--token-col", default="patch_delta_norm_std")
    parser.add_argument("--raw-weight", type=float, default=0.50)
    parser.add_argument("--soft-weight", type=float, default=0.25)
    parser.add_argument("--token-weight", type=float, default=0.25)
    parser.add_argument("--output-scores-csv", type=Path, required=True)
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    parser.add_argument("--output-per-source-csv", type=Path, required=True)
    args = parser.parse_args()

    scores, summary, per_source = run(args)
    args.output_scores_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_per_source_csv.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_scores_csv, index=False)
    summary.to_csv(args.output_summary_csv, index=False)
    per_source.to_csv(args.output_per_source_csv, index=False)
    print(summary.to_string(index=False))
    print(f"Saved scores -> {args.output_scores_csv}")
    print(f"Saved summary -> {args.output_summary_csv}")


if __name__ == "__main__":
    main()
