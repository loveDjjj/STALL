#!/usr/bin/env python3
"""Slice analysis for sample-level persistence gates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


KEY_COLUMNS = ["subset", "source_model", "filename"]


def _read(path: Path, score_col: str, out_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in KEY_COLUMNS + [score_col] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    out = df[KEY_COLUMNS + [score_col]].copy()
    for col in KEY_COLUMNS:
        out[col] = out[col].astype(str)
    return out.rename(columns={score_col: out_col})


def _rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(1.0, float(len(values) - 1))


def _score_metrics(real_scores: np.ndarray, fake_scores: np.ndarray) -> tuple[float, float]:
    y = np.concatenate([np.ones(len(real_scores)), np.zeros(len(fake_scores))])
    s = np.concatenate([real_scores, fake_scores])
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def _metrics_by_model(df: pd.DataFrame, real_df: pd.DataFrame, fake_df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows = []
    real_scores = real_df[score_col].to_numpy(float)
    for model, group in fake_df.groupby("source_model", sort=True):
        auc, ap = _score_metrics(real_scores, group[score_col].to_numpy(float))
        rows.append({"source_model": model, "auc": auc, "ap": ap, "n_real": len(real_df), "n_fake": len(group)})
    return pd.DataFrame(rows)


def _make_scores(args: argparse.Namespace) -> pd.DataFrame:
    df = _read(args.global_csv, args.global_score_col, "global_score").merge(
        _read(args.raw_patch_csv, args.raw_patch_score_col, "raw_patch_score"),
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    ).merge(
        _read(args.persistence_csv, args.persistence_score_col, "persistence_score"),
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    df["global_rank"] = _rank01(df["global_score"].to_numpy(float))
    df["raw_rank"] = _rank01(df["raw_patch_score"].to_numpy(float))
    df["persistence_rank"] = _rank01(df["persistence_score"].to_numpy(float))
    df["base_score"] = args.base_alpha * df["global_rank"] + (1.0 - args.base_alpha) * df["raw_rank"]
    df["disagreement"] = np.abs(df["global_rank"] - df["raw_rank"])
    df["persistence_conf"] = np.abs(df["persistence_rank"] - 0.5) * 2.0
    if args.gate_mode == "both":
        gate = (df["disagreement"].to_numpy(float) >= args.disagreement_threshold) & (
            df["persistence_conf"].to_numpy(float) >= args.confidence_threshold
        )
        df["gate"] = gate.astype(float)
    elif args.gate_mode == "disagreement":
        df["gate"] = (df["disagreement"].to_numpy(float) >= args.disagreement_threshold).astype(float)
    elif args.gate_mode == "confidence":
        df["gate"] = (df["persistence_conf"].to_numpy(float) >= args.confidence_threshold).astype(float)
    else:
        raise ValueError(f"Unsupported gate mode for slice analysis: {args.gate_mode}")
    df["gated_score"] = (1.0 - args.intervention_weight * df["gate"]) * df["base_score"] + (
        args.intervention_weight * df["gate"]
    ) * df["persistence_rank"]
    return df


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = _make_scores(args)
    real = df[df["subset"].str.lower() == "real"].copy()
    fake = df[df["subset"].str.lower() != "real"].copy()
    base_by_model = _metrics_by_model(df, real, fake, "base_score").rename(columns={"auc": "base_auc", "ap": "base_ap"})
    gated_by_model = _metrics_by_model(df, real, fake, "gated_score").rename(
        columns={"auc": "gated_auc", "ap": "gated_ap"}
    )
    per_model = base_by_model.merge(gated_by_model, on=["source_model", "n_real", "n_fake"], how="inner")
    per_model["delta_auc"] = per_model["gated_auc"] - per_model["base_auc"]
    per_model["delta_ap"] = per_model["gated_ap"] - per_model["base_ap"]
    fake_gate = fake.groupby("source_model")["gate"].mean().rename("fake_gate_mean")
    fake_dis = fake.groupby("source_model")["disagreement"].mean().rename("fake_disagreement_mean")
    fake_conf = fake.groupby("source_model")["persistence_conf"].mean().rename("fake_conf_mean")
    per_model = per_model.merge(fake_gate, on="source_model").merge(fake_dis, on="source_model").merge(fake_conf, on="source_model")
    per_model.insert(0, "dataset", args.dataset)

    # Hard-case buckets are defined by fake disagreement quantiles, then compared
    # against all real videos. This answers which fake difficulty band changes.
    quantiles = [float(x) for x in args.fake_disagreement_quantiles.split(",") if x.strip()]
    cuts = np.unique(np.quantile(fake["disagreement"].to_numpy(float), quantiles))
    if len(cuts) < 2:
        cuts = np.array([fake["disagreement"].min(), fake["disagreement"].max()])
    fake = fake.copy()
    fake["disagreement_bin"] = pd.cut(fake["disagreement"], bins=cuts, include_lowest=True, duplicates="drop")
    bucket_rows = []
    real_base = real["base_score"].to_numpy(float)
    real_gated = real["gated_score"].to_numpy(float)
    for bucket, group in fake.groupby("disagreement_bin", observed=True):
        base_auc, base_ap = _score_metrics(real_base, group["base_score"].to_numpy(float))
        gated_auc, gated_ap = _score_metrics(real_gated, group["gated_score"].to_numpy(float))
        bucket_rows.append(
            {
                "dataset": args.dataset,
                "bucket": str(bucket),
                "n_fake": len(group),
                "fake_gate_mean": float(group["gate"].mean()),
                "fake_disagreement_mean": float(group["disagreement"].mean()),
                "fake_conf_mean": float(group["persistence_conf"].mean()),
                "base_auc": base_auc,
                "base_ap": base_ap,
                "gated_auc": gated_auc,
                "gated_ap": gated_ap,
                "delta_auc": gated_auc - base_auc,
                "delta_ap": gated_ap - base_ap,
            }
        )
    buckets = pd.DataFrame(bucket_rows)

    overview = pd.DataFrame(
        [
            {
                "dataset": args.dataset,
                "n_rows": len(df),
                "n_real": len(real),
                "n_fake": len(fake),
                "gate_mean_all": float(df["gate"].mean()),
                "gate_mean_real": float(real["gate"].mean()),
                "gate_mean_fake": float(fake["gate"].mean()),
                "base_auc_mean": float(per_model["base_auc"].mean()),
                "base_ap_mean": float(per_model["base_ap"].mean()),
                "gated_auc_mean": float(per_model["gated_auc"].mean()),
                "gated_ap_mean": float(per_model["gated_ap"].mean()),
                "delta_auc_mean": float(per_model["delta_auc"].mean()),
                "delta_ap_mean": float(per_model["delta_ap"].mean()),
            }
        ]
    )
    return overview, per_model.sort_values(["delta_auc", "delta_ap"], ascending=False), buckets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--global-csv", type=Path, required=True)
    parser.add_argument("--raw-patch-csv", type=Path, required=True)
    parser.add_argument("--persistence-csv", type=Path, required=True)
    parser.add_argument("--persistence-score-col", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--global-score-col", default="final_score")
    parser.add_argument("--raw-patch-score-col", default="final_score")
    parser.add_argument("--base-alpha", type=float, default=0.60)
    parser.add_argument("--intervention-weight", type=float, default=0.15)
    parser.add_argument("--gate-mode", default="both")
    parser.add_argument("--disagreement-threshold", type=float, default=0.20)
    parser.add_argument("--confidence-threshold", type=float, default=0.30)
    parser.add_argument("--fake-disagreement-quantiles", default="0,0.25,0.50,0.75,0.90,1.0")
    args = parser.parse_args()

    overview, per_model, buckets = run(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.dataset
    overview.to_csv(args.output_dir / f"{prefix}_sample_gate_overview.csv", index=False)
    per_model.to_csv(args.output_dir / f"{prefix}_sample_gate_per_model.csv", index=False)
    buckets.to_csv(args.output_dir / f"{prefix}_sample_gate_disagreement_buckets.csv", index=False)
    print("OVERVIEW")
    print(overview.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nTOP PER MODEL")
    print(per_model.head(20).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nBUCKETS")
    print(buckets.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
