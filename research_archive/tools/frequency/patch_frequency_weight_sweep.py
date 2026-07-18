#!/usr/bin/env python3
"""Sweep additive patch-frequency correction weights against raw patch rank."""

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


def _per_source_metrics(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
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
    return pd.DataFrame(rows)


def _parse_weights(value: str) -> list[float]:
    out = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = _read(args.raw_patch_csv, [args.raw_patch_col]).rename(columns={args.raw_patch_col: "raw_patch"})
    freq = _read(args.frequency_csv, [args.frequency_col]).rename(columns={args.frequency_col: "frequency_signal"})
    df = raw.merge(freq, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    df["raw_rank"] = _real_rank(df, "raw_patch").astype(np.float32)
    df["frequency_rank"] = _real_rank(df, "frequency_signal").astype(np.float32)
    df["frequency_component"] = 1.0 - df["frequency_rank"] if args.invert_frequency else df["frequency_rank"]

    raw_metrics = _per_source_metrics(df, "raw_rank").rename(columns={"auc": "raw_auc", "ap": "raw_ap"})
    freq_metrics = _per_source_metrics(df, "frequency_rank").rename(
        columns={"auc": "frequency_auc", "ap": "frequency_ap"}
    )
    summary_rows = []
    per_source_rows = []
    score_cols = ["raw_rank", "frequency_rank"]

    for weight in args.weights:
        score_col = f"frequency_add_w{str(weight).replace('.', 'p')}"
        score_cols.append(score_col)
        df[score_col] = (1.0 - weight) * df["raw_rank"] + weight * df["frequency_component"]
        metrics = _per_source_metrics(df, score_col).rename(columns={"auc": "auc", "ap": "ap"})
        merged = raw_metrics.merge(metrics, on=["source_model", "n_real", "n_fake"], how="inner")
        merged["weight"] = weight
        merged["frequency_col"] = args.frequency_col
        merged["invert_frequency"] = bool(args.invert_frequency)
        merged["delta_auc"] = merged["auc"] - merged["raw_auc"]
        merged["delta_ap"] = merged["ap"] - merged["raw_ap"]
        per_source_rows.append(merged)
        summary_rows.append(
            {
                "weight": weight,
                "frequency_col": args.frequency_col,
                "invert_frequency": bool(args.invert_frequency),
                "avg_auc": float(merged["auc"].mean()),
                "avg_ap": float(merged["ap"].mean()),
                "min_auc": float(merged["auc"].min()),
                "min_ap": float(merged["ap"].min()),
                "avg_delta_auc": float(merged["delta_auc"].mean()),
                "avg_delta_ap": float(merged["delta_ap"].mean()),
                "min_delta_auc": float(merged["delta_auc"].min()),
                "min_delta_ap": float(merged["delta_ap"].min()),
                "improved_auc_sources": int((merged["delta_auc"] > 0).sum()),
                "improved_ap_sources": int((merged["delta_ap"] > 0).sum()),
                "n_sources": int(len(merged)),
                "n_rows": int(len(df)),
            }
        )

    raw_avg = raw_metrics[["raw_auc", "raw_ap"]].mean()
    freq_avg = freq_metrics[["frequency_auc", "frequency_ap"]].mean()
    summary_rows.append(
        {
            "weight": 0.0,
            "frequency_col": "raw_rank_baseline",
            "invert_frequency": False,
            "avg_auc": float(raw_avg["raw_auc"]),
            "avg_ap": float(raw_avg["raw_ap"]),
            "min_auc": float(raw_metrics["raw_auc"].min()),
            "min_ap": float(raw_metrics["raw_ap"].min()),
            "avg_delta_auc": 0.0,
            "avg_delta_ap": 0.0,
            "min_delta_auc": 0.0,
            "min_delta_ap": 0.0,
            "improved_auc_sources": 0,
            "improved_ap_sources": 0,
            "n_sources": int(len(raw_metrics)),
            "n_rows": int(len(df)),
        }
    )
    summary_rows.append(
        {
            "weight": 1.0,
            "frequency_col": "frequency_rank_only",
            "invert_frequency": bool(args.invert_frequency),
            "avg_auc": float(freq_avg["frequency_auc"]),
            "avg_ap": float(freq_avg["frequency_ap"]),
            "min_auc": float(freq_metrics["frequency_auc"].min()),
            "min_ap": float(freq_metrics["frequency_ap"].min()),
            "avg_delta_auc": float((freq_metrics["frequency_auc"].to_numpy() - raw_metrics["raw_auc"].to_numpy()).mean()),
            "avg_delta_ap": float((freq_metrics["frequency_ap"].to_numpy() - raw_metrics["raw_ap"].to_numpy()).mean()),
            "min_delta_auc": float((freq_metrics["frequency_auc"].to_numpy() - raw_metrics["raw_auc"].to_numpy()).min()),
            "min_delta_ap": float((freq_metrics["frequency_ap"].to_numpy() - raw_metrics["raw_ap"].to_numpy()).min()),
            "improved_auc_sources": int((freq_metrics["frequency_auc"].to_numpy() > raw_metrics["raw_auc"].to_numpy()).sum()),
            "improved_ap_sources": int((freq_metrics["frequency_ap"].to_numpy() > raw_metrics["raw_ap"].to_numpy()).sum()),
            "n_sources": int(len(freq_metrics)),
            "n_rows": int(len(df)),
        }
    )

    summary = pd.DataFrame(summary_rows).sort_values(["avg_delta_auc", "avg_delta_ap"], ascending=False)
    per_source = pd.concat(per_source_rows, ignore_index=True) if per_source_rows else pd.DataFrame()
    scores = df[KEY_COLUMNS + ["raw_patch", "frequency_signal", "raw_rank", "frequency_rank"] + score_cols[2:]].copy()
    return scores, summary, per_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-patch-csv", type=Path, required=True)
    parser.add_argument("--frequency-csv", type=Path, required=True)
    parser.add_argument("--raw-patch-col", default="final_score")
    parser.add_argument("--frequency-col", default="score_anti_jitter_tail")
    parser.add_argument("--weights", type=_parse_weights, default=_parse_weights("0.03,0.05,0.07,0.10"))
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
