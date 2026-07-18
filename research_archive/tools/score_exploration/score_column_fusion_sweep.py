#!/usr/bin/env python3
"""Sweep rank fusion between a global score and arbitrary score columns."""

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


def _metrics(df: pd.DataFrame, score_col: str) -> tuple[float, float]:
    real = df[df["subset"].str.lower() == "real"]
    fake = df[df["subset"].str.lower() != "real"]
    aucs = []
    aps = []
    for _, group in fake.groupby("source_model", sort=True):
        y = np.concatenate([np.ones(len(real)), np.zeros(len(group))])
        s = np.concatenate([real[score_col].to_numpy(float), group[score_col].to_numpy(float)])
        aucs.append(roc_auc_score(y, s))
        aps.append(average_precision_score(y, s))
    return float(np.mean(aucs)), float(np.mean(aps))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--global-csv", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--global-score-col", default="final_score")
    parser.add_argument("--candidate-cols", default=None)
    parser.add_argument("--alphas", default="0.50,0.55,0.60,0.65,0.70")
    args = parser.parse_args()

    cand = pd.read_csv(args.candidate_csv)
    if args.candidate_cols is None:
        candidate_cols = [c for c in cand.columns if c not in KEY_COLUMNS]
    else:
        candidate_cols = [c.strip() for c in args.candidate_cols.split(",") if c.strip()]
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    global_df = _read(args.global_csv, args.global_score_col, "global_score")
    rows = []
    for col in candidate_cols:
        candidate_df = _read(args.candidate_csv, col, "candidate_score")
        df = global_df.merge(candidate_df, on=KEY_COLUMNS, how="inner", validate="one_to_one")
        df["global_rank"] = _rank01(df["global_score"].to_numpy(float))
        df["candidate_rank"] = _rank01(df["candidate_score"].to_numpy(float))
        auc, ap = _metrics(df.assign(final_score=df["candidate_rank"]), "final_score")
        rows.append(
            {
                "dataset": args.dataset,
                "candidate_col": col,
                "alpha": np.nan,
                "avg_auc": auc,
                "avg_ap": ap,
                "kind": "candidate_only_rank",
                "n_rows": len(df),
            }
        )
        for alpha in alphas:
            final = alpha * df["global_rank"].to_numpy(float) + (1.0 - alpha) * df["candidate_rank"].to_numpy(float)
            auc, ap = _metrics(df.assign(final_score=final), "final_score")
            rows.append(
                {
                    "dataset": args.dataset,
                    "candidate_col": col,
                    "alpha": alpha,
                    "avg_auc": auc,
                    "avg_ap": ap,
                    "kind": "rank_fusion",
                    "n_rows": len(df),
                }
            )
    out = pd.DataFrame(rows).sort_values(["avg_auc", "avg_ap"], ascending=False)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(out.head(30).to_string(index=False, formatters={"avg_auc": lambda x: f"{x:.4f}", "avg_ap": lambda x: f"{x:.4f}"}))
    print(f"Saved fusion sweep -> {args.output_csv}")


if __name__ == "__main__":
    main()
