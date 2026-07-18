#!/usr/bin/env python3
"""Audit a fallback score file against the raw alpha default by source."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


KEY_COLUMNS = ["subset", "source_model", "filename"]


def _read_score(path: Path, score_col: str, out_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in [*KEY_COLUMNS, score_col] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    out = df[[*KEY_COLUMNS, score_col]].copy()
    for col in KEY_COLUMNS:
        out[col] = out[col].astype(str)
    return out.rename(columns={score_col: out_col})


def _auc_ap(real_scores: pd.Series, fake_scores: pd.Series) -> tuple[float, float]:
    y = [1] * len(real_scores) + [0] * len(fake_scores)
    s = [*real_scores.astype(float), *fake_scores.astype(float)]
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def _source_metrics(df: pd.DataFrame, score_col: str, prefix: str) -> pd.DataFrame:
    real = df[df["subset"].str.lower() == "real"]
    fake = df[df["subset"].str.lower() != "real"]
    rows = []
    real_scores = real[score_col]
    for source, group in fake.groupby("source_model", sort=True):
        auc, ap = _auc_ap(real_scores, group[score_col])
        rows.append(
            {
                "source_model": source,
                f"{prefix}_auc": auc,
                f"{prefix}_ap": ap,
                "n_fake": len(group),
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    default = _read_score(args.global_csv, args.global_score_col, "global_score").merge(
        _read_score(args.raw_patch_csv, args.raw_patch_score_col, "raw_patch_score"),
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    default["default_score"] = (
        args.alpha * default["global_score"].astype(float)
        + (1.0 - args.alpha) * default["raw_patch_score"].astype(float)
    )
    fallback = _read_score(args.fallback_csv, args.fallback_score_col, "fallback_score")
    df = default[[*KEY_COLUMNS, "default_score"]].merge(
        fallback,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    if len(df) != len(default):
        raise ValueError(f"Fallback overlap {len(df)} does not match default rows {len(default)}")

    default_metrics = _source_metrics(df, "default_score", "default")
    fallback_metrics = _source_metrics(df, "fallback_score", "fallback")
    audit = default_metrics.merge(fallback_metrics, on=["source_model", "n_fake"], validate="one_to_one")
    audit["delta_auc"] = audit["fallback_auc"] - audit["default_auc"]
    audit["delta_ap"] = audit["fallback_ap"] - audit["default_ap"]
    audit.insert(0, "dataset", args.dataset)
    audit = audit.sort_values(["delta_auc", "delta_ap", "source_model"], ascending=[True, True, True])

    summary = pd.DataFrame(
        [
            {
                "dataset": args.dataset,
                "n_sources": int(len(audit)),
                "n_negative_auc": int((audit["delta_auc"] < 0).sum()),
                "n_negative_ap": int((audit["delta_ap"] < 0).sum()),
                "min_delta_auc": float(audit["delta_auc"].min()),
                "min_delta_ap": float(audit["delta_ap"].min()),
                "mean_delta_auc": float(audit["delta_auc"].mean()),
                "mean_delta_ap": float(audit["delta_ap"].mean()),
                "weighted_mean_delta_auc": float((audit["delta_auc"] * audit["n_fake"]).sum() / audit["n_fake"].sum()),
                "weighted_mean_delta_ap": float((audit["delta_ap"] * audit["n_fake"]).sum() / audit["n_fake"].sum()),
                "worst_auc_source": str(audit.iloc[0]["source_model"]),
                "worst_ap_source": str(audit.sort_values(["delta_ap", "delta_auc"]).iloc[0]["source_model"]),
            }
        ]
    )
    return audit, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--global-csv", type=Path, required=True)
    parser.add_argument("--raw-patch-csv", type=Path, required=True)
    parser.add_argument("--fallback-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--global-score-col", default="final_score")
    parser.add_argument("--raw-patch-score-col", default="final_score")
    parser.add_argument("--fallback-score-col", default="final_score")
    parser.add_argument("--alpha", type=float, default=0.60)
    args = parser.parse_args()

    audit, summary = run(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved audit -> {args.output_csv}")
    print(f"Saved summary -> {args.summary_csv}")


if __name__ == "__main__":
    main()
