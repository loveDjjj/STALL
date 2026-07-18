#!/usr/bin/env python3
"""Dataset-level reliability diagnostics for persistence candidates.

This tool intentionally separates unsupervised diagnostics from outcome
metrics. The diagnostics use only score distributions and real rows; the
outcome columns are appended to judge whether a gate could be plausible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


KEY_COLUMNS = ["subset", "source_model", "filename"]


def _rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(1.0, float(len(values) - 1))


def _read_score(path: Path, score_col: str, out_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in KEY_COLUMNS + [score_col] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    out = df[KEY_COLUMNS + [score_col]].copy()
    for col in KEY_COLUMNS:
        out[col] = out[col].astype(str)
    return out.rename(columns={score_col: out_col})


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


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _bootstrap_rank_stability(values: np.ndarray, seed: int, rounds: int = 100) -> float:
    if len(values) < 4:
        return 0.0
    rng = np.random.RandomState(seed)
    qs = []
    n = len(values)
    for _ in range(rounds):
        sample = values[rng.randint(0, n, size=n)]
        qs.append(np.quantile(sample, [0.05, 0.50, 0.95]))
    qs = np.asarray(qs)
    # Lower variability means more stable. Convert to a bounded reliability proxy.
    variability = float(np.mean(np.std(qs, axis=0)))
    return float(1.0 / (1.0 + variability))


def run(args: argparse.Namespace) -> pd.DataFrame:
    global_df = _read_score(args.global_csv, args.global_score_col, "global_score")
    raw_df = _read_score(args.raw_patch_csv, args.raw_patch_score_col, "raw_patch_score")
    cand_raw = pd.read_csv(args.persistence_csv)
    candidate_cols = [c.strip() for c in args.candidate_cols.split(",") if c.strip()]
    rows = []
    for candidate_col in candidate_cols:
        cand_df = cand_raw[KEY_COLUMNS + [candidate_col]].copy()
        for col in KEY_COLUMNS:
            cand_df[col] = cand_df[col].astype(str)
        cand_df = cand_df.rename(columns={candidate_col: "candidate_score"})
        df = global_df.merge(raw_df, on=KEY_COLUMNS, how="inner", validate="one_to_one").merge(
            cand_df, on=KEY_COLUMNS, how="inner", validate="one_to_one"
        )
        df["global_rank"] = _rank01(df["global_score"].to_numpy(float))
        df["raw_rank"] = _rank01(df["raw_patch_score"].to_numpy(float))
        df["candidate_rank"] = _rank01(df["candidate_score"].to_numpy(float))
        real = df[df["subset"].str.lower() == "real"]

        candidate = df["candidate_rank"].to_numpy(float)
        raw = df["raw_rank"].to_numpy(float)
        global_rank = df["global_rank"].to_numpy(float)
        real_candidate = real["candidate_rank"].to_numpy(float)
        real_raw = real["raw_rank"].to_numpy(float)
        real_global = real["global_rank"].to_numpy(float)

        raw_default = args.raw_default_alpha * global_rank + (1.0 - args.raw_default_alpha) * raw
        cand_default = args.candidate_alpha * global_rank + (1.0 - args.candidate_alpha) * candidate
        raw_auc, raw_ap = _metrics(df.assign(final_score=raw_default), "final_score")
        cand_auc, cand_ap = _metrics(df.assign(final_score=cand_default), "final_score")
        cand_only_auc, cand_only_ap = _metrics(df.assign(final_score=candidate), "final_score")

        real_q05, real_q50, real_q95 = np.quantile(real_candidate, [0.05, 0.50, 0.95])
        raw_real_q05, raw_real_q50, raw_real_q95 = np.quantile(real_raw, [0.05, 0.50, 0.95])
        rows.append(
            {
                "dataset": args.dataset,
                "candidate_col": candidate_col,
                "n_rows": len(df),
                "n_real": len(real),
                "candidate_alpha": args.candidate_alpha,
                "raw_default_alpha": args.raw_default_alpha,
                # Unsupervised / real-only diagnostics.
                "real_candidate_iqr": float(np.quantile(real_candidate, 0.75) - np.quantile(real_candidate, 0.25)),
                "real_candidate_q05": float(real_q05),
                "real_candidate_q50": float(real_q50),
                "real_candidate_q95": float(real_q95),
                "real_candidate_tail_low_frac": float(np.mean(real_candidate < 0.05)),
                "real_candidate_tail_high_frac": float(np.mean(real_candidate > 0.95)),
                "real_candidate_bootstrap_stability": _bootstrap_rank_stability(real_candidate, args.seed),
                "real_raw_iqr": float(np.quantile(real_raw, 0.75) - np.quantile(real_raw, 0.25)),
                "real_raw_q05": float(raw_real_q05),
                "real_raw_q50": float(raw_real_q50),
                "real_raw_q95": float(raw_real_q95),
                "real_candidate_raw_corr": _safe_corr(real_candidate, real_raw),
                "all_candidate_raw_corr": _safe_corr(candidate, raw),
                "real_candidate_global_corr": _safe_corr(real_candidate, real_global),
                "all_candidate_global_corr": _safe_corr(candidate, global_rank),
                "real_candidate_raw_disagreement_mean": float(np.mean(np.abs(real_candidate - real_raw))),
                "all_candidate_raw_disagreement_mean": float(np.mean(np.abs(candidate - raw))),
                "real_global_raw_disagreement_mean": float(np.mean(np.abs(real_global - real_raw))),
                # Outcome columns, not available to a real unsupervised gate.
                "candidate_only_auc": cand_only_auc,
                "candidate_only_ap": cand_only_ap,
                "candidate_fusion_auc": cand_auc,
                "candidate_fusion_ap": cand_ap,
                "raw_default_auc": raw_auc,
                "raw_default_ap": raw_ap,
                "delta_auc": cand_auc - raw_auc,
                "delta_ap": cand_ap - raw_ap,
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "delta_auc", "delta_ap"], ascending=[True, False, False])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--global-csv", type=Path, required=True)
    parser.add_argument("--raw-patch-csv", type=Path, required=True)
    parser.add_argument("--persistence-csv", type=Path, required=True)
    parser.add_argument("--candidate-cols", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--global-score-col", default="final_score")
    parser.add_argument("--raw-patch-score-col", default="final_score")
    parser.add_argument("--candidate-alpha", type=float, default=0.60)
    parser.add_argument("--raw-default-alpha", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = run(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(
        out[
            [
                "dataset",
                "candidate_col",
                "candidate_fusion_auc",
                "candidate_fusion_ap",
                "raw_default_auc",
                "raw_default_ap",
                "delta_auc",
                "delta_ap",
                "real_candidate_raw_corr",
                "real_candidate_raw_disagreement_mean",
                "real_candidate_bootstrap_stability",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    print(f"Saved reliability diagnostics -> {args.output_csv}")


if __name__ == "__main__":
    main()
