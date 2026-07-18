#!/usr/bin/env python3
"""Small additive patch-frequency correction on top of raw patch score."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


KEY_COLUMNS = ["subset", "source_model", "filename"]


def _read(path: Path, cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in KEY_COLUMNS + cols if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    out = df[KEY_COLUMNS + cols].copy()
    for col in KEY_COLUMNS:
        out[col] = out[col].astype(str)
    return out


def _real_rank(df: pd.DataFrame, col: str) -> np.ndarray:
    real = np.sort(df.loc[df["subset"].str.lower().eq("real"), col].to_numpy(float))
    if len(real) == 0:
        raise ValueError(f"No real rows for rank calibration: {col}")
    return np.searchsorted(real, df[col].to_numpy(float), side="right") / float(len(real))


def _metrics(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    real = df[df["subset"].str.lower().eq("real")]
    rows = []
    for source, fake in df[~df["subset"].str.lower().eq("real")].groupby("source_model", sort=True):
        y = np.concatenate([np.ones(len(real)), np.zeros(len(fake))])
        s = np.concatenate([real[score_col].to_numpy(float), fake[score_col].to_numpy(float)])
        rows.append(
            {
                "source_model": source,
                "n_real": len(real),
                "n_fake": len(fake),
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
    freq = _read(args.frequency_csv, [args.frequency_col]).rename(columns={args.frequency_col: "frequency_signal"})
    df = raw.merge(freq, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    df["raw_rank"] = _real_rank(df, "raw_patch").astype(np.float32)
    df["frequency_rank"] = _real_rank(df, "frequency_signal").astype(np.float32)
    freq_component = 1.0 - df["frequency_rank"] if args.invert_frequency else df["frequency_rank"]
    df["final_score"] = (1.0 - args.frequency_weight) * df["raw_rank"] + args.frequency_weight * freq_component
    df["patch_frequency_additive_v1"] = df["final_score"]

    per_source_frames = []
    summary_rows = []
    for score_col in ["raw_rank", "frequency_rank", "final_score"]:
        metrics = _metrics(df, score_col)
        per_source_frames.append(metrics.assign(score_col=score_col))
        avg = metrics[metrics["source_model"] == "Average"].iloc[0]
        source_metrics = metrics[metrics["source_model"] != "Average"]
        summary_rows.append(
            {
                "score_col": score_col,
                "avg_auc": float(avg["auc"]),
                "avg_ap": float(avg["ap"]),
                "min_auc": float(source_metrics["auc"].min()),
                "min_ap": float(source_metrics["ap"].min()),
                "n_sources": int(len(source_metrics)),
                "n_rows": int(len(df)),
                "frequency_weight": args.frequency_weight,
                "frequency_col": args.frequency_col,
                "invert_frequency": bool(args.invert_frequency),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["avg_auc", "avg_ap"], ascending=False)
    per_source = pd.concat(per_source_frames, ignore_index=True)
    return df, summary, per_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-patch-csv", type=Path, required=True)
    parser.add_argument("--frequency-csv", type=Path, required=True)
    parser.add_argument("--raw-patch-col", default="final_score")
    parser.add_argument("--frequency-col", default="score_anti_jitter_tail")
    parser.add_argument("--frequency-weight", type=float, default=0.05)
    parser.add_argument("--invert-frequency", action="store_true")
    parser.add_argument("--output-scores-csv", type=Path, required=True)
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    parser.add_argument("--output-per-source-csv", type=Path, required=True)
    args = parser.parse_args()

    scores, summary, per_source = run(args)
    args.output_scores_csv.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_scores_csv, index=False)
    summary.to_csv(args.output_summary_csv, index=False)
    per_source.to_csv(args.output_per_source_csv, index=False)
    print(summary.to_string(index=False))
    print(f"Saved scores -> {args.output_scores_csv}")
    print(f"Saved summary -> {args.output_summary_csv}")


if __name__ == "__main__":
    main()
