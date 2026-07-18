#!/usr/bin/env python3
"""Grid-search three-way score fusion from existing CSV files."""

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


def _metrics(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    real = df[df["subset"].str.lower() == "real"]
    fake = df[df["subset"].str.lower() != "real"]
    rows = []
    for model, fake_group in fake.groupby("source_model", sort=True):
        y_true = np.concatenate([np.ones(len(real)), np.zeros(len(fake_group))])
        y_score = np.concatenate(
            [
                real[score_col].to_numpy(dtype=np.float64),
                fake_group[score_col].to_numpy(dtype=np.float64),
            ]
        )
        rows.append(
            {
                "source_model": model,
                "n_real": len(real),
                "n_fake": len(fake_group),
                "auc": roc_auc_score(y_true, y_score),
                "ap": average_precision_score(y_true, y_score),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No fake rows found")
    out.loc[len(out)] = {
        "source_model": "Average",
        "n_real": int(round(out["n_real"].mean())),
        "n_fake": int(round(out["n_fake"].mean())),
        "auc": out["auc"].mean(),
        "ap": out["ap"].mean(),
    }
    return out


def _rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(1.0, float(len(values) - 1))


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = _read(args.score_a_csv, args.score_a_col, "score_a")
    b = _read(args.score_b_csv, args.score_b_col, "score_b")
    c = _read(args.score_c_csv, args.score_c_col, "score_c")
    df = a.merge(b, on=KEY_COLUMNS, how="inner", validate="one_to_one").merge(
        c, on=KEY_COLUMNS, how="inner", validate="one_to_one"
    )
    if args.rank_normalize:
        for col in ["score_a", "score_b", "score_c"]:
            df[col] = _rank01(df[col].to_numpy(dtype=np.float64))

    weights = []
    step = args.weight_step
    n = int(round(1.0 / step))
    for i in range(n + 1):
        wa = i * step
        for j in range(n + 1 - i):
            wb = j * step
            wc = 1.0 - wa - wb
            if wc < -1e-8:
                continue
            weights.append((round(wa, 10), round(wb, 10), round(wc, 10)))

    summary_rows = []
    per_model_frames = []
    for wa, wb, wc in weights:
        scored = df.copy()
        scored["final_score"] = wa * scored["score_a"] + wb * scored["score_b"] + wc * scored["score_c"]
        metrics = _metrics(scored, "final_score")
        avg = metrics[metrics["source_model"] == "Average"].iloc[0]
        summary_rows.append(
            {
                "dataset": args.dataset,
                "tag": args.tag,
                "score_a": args.score_a_name,
                "score_b": args.score_b_name,
                "score_c": args.score_c_name,
                "wa": wa,
                "wb": wb,
                "wc": wc,
                "avg_auc": float(avg["auc"]),
                "avg_ap": float(avg["ap"]),
                "n_rows": len(scored),
                "rank_normalize": args.rank_normalize,
            }
        )
        metrics = metrics.copy()
        metrics.insert(0, "wc", wc)
        metrics.insert(0, "wb", wb)
        metrics.insert(0, "wa", wa)
        metrics.insert(0, "tag", args.tag)
        metrics.insert(0, "dataset", args.dataset)
        per_model_frames.append(metrics)
    return (
        pd.DataFrame(summary_rows).sort_values(["avg_auc", "avg_ap"], ascending=False),
        pd.concat(per_model_frames, ignore_index=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--score-a-csv", type=Path, required=True)
    parser.add_argument("--score-b-csv", type=Path, required=True)
    parser.add_argument("--score-c-csv", type=Path, required=True)
    parser.add_argument("--score-a-col", default="final_score")
    parser.add_argument("--score-b-col", default="final_score")
    parser.add_argument("--score-c-col", required=True)
    parser.add_argument("--score-a-name", default="global")
    parser.add_argument("--score-b-name", default="best_patch")
    parser.add_argument("--score-c-name", default="tail_proxy")
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--rank-normalize", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, per_model = run(args)
    prefix = f"{args.dataset}_{args.tag}"
    summary.to_csv(args.output_dir / f"{prefix}_three_score_summary.csv", index=False)
    per_model.to_csv(args.output_dir / f"{prefix}_three_score_per_model.csv", index=False)
    best = summary.iloc[0]
    print(
        f"{args.dataset}/{args.tag}: best wa={best.wa:.2f} wb={best.wb:.2f} wc={best.wc:.2f} "
        f"AUC={best.avg_auc:.4f} AP={best.avg_ap:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
