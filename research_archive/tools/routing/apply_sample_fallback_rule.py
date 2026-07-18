#!/usr/bin/env python3
"""Apply the fixed sample-level fallback rule and write final scores.

Validated rule:
    use split score when persistence_rank - base_score >= 0,
    otherwise use universal score.

The output score file keeps the project-standard key columns plus final_score,
with optional debug columns for reproducibility.
"""

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


def _metrics_by_source(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    real = df[df["subset"].str.lower() == "real"]
    fake = df[df["subset"].str.lower() != "real"]
    real_scores = real[score_col].to_numpy(float)
    rows = []
    for source, group in fake.groupby("source_model", sort=True):
        auc, ap = _score_metrics(real_scores, group[score_col].to_numpy(float))
        rows.append({"source_model": source, "auc": auc, "ap": ap, "n_fake": len(group)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out.loc[len(out)] = {
            "source_model": "Average",
            "auc": float(out["auc"].mean()),
            "ap": float(out["ap"].mean()),
            "n_fake": int(out["n_fake"].sum()),
        }
    return out


def make_scores(args: argparse.Namespace) -> pd.DataFrame:
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
    df["persistence_rank_raw"] = _rank01(df["persistence_score"].to_numpy(float))
    df["persistence_rank"] = np.clip(
        0.5 + args.persistence_scale * (df["persistence_rank_raw"] - 0.5),
        0.0,
        1.0,
    )
    if args.base_space == "rank":
        df["base_score"] = args.base_alpha * df["global_rank"] + (1.0 - args.base_alpha) * df["raw_rank"]
    elif args.base_space == "raw":
        df["base_score"] = args.base_alpha * df["global_score"] + (1.0 - args.base_alpha) * df["raw_patch_score"]
    else:
        raise ValueError(f"Unsupported base space: {args.base_space}")
    df["disagreement"] = np.abs(df["global_rank"] - df["raw_rank"])
    df["persistence_conf"] = np.abs(df["persistence_rank"] - 0.5) * 2.0
    df["sample_gate"] = (
        (df["disagreement"].to_numpy(float) >= args.disagreement_threshold)
        & (df["persistence_conf"].to_numpy(float) >= args.confidence_threshold)
    ).astype(float)

    source_gate_mean = (
        df[df["subset"].str.lower() != "real"].groupby("source_model")["sample_gate"].mean().rename("source_gate_mean")
    )
    df = df.merge(source_gate_mean, on="source_model", how="left")
    df["source_gate_mean"] = df["source_gate_mean"].fillna(0.0)

    universal_weight = args.universal_weight * df["sample_gate"]
    df["universal_score"] = (1.0 - universal_weight) * df["base_score"] + universal_weight * df["persistence_rank"]

    is_real = df["subset"].str.lower() == "real"
    fake_source_enabled = (df["source_gate_mean"] >= args.source_gate_threshold) & (~is_real)
    split_weight = df["sample_gate"] * (
        args.real_weight * is_real.to_numpy(float) + args.fake_weight * fake_source_enabled.to_numpy(float)
    )
    df["split_score"] = (1.0 - split_weight) * df["base_score"] + split_weight * df["persistence_rank"]
    df["persistence_minus_base"] = df["persistence_rank"] - df["base_score"]
    df["split_minus_universal"] = df["split_score"] - df["universal_score"]
    if args.selector_feature == "always":
        use_split = np.ones(len(df), dtype=bool)
    elif args.selector_feature == "never":
        use_split = np.zeros(len(df), dtype=bool)
    else:
        values = df[args.selector_feature].to_numpy(float)
        if args.selector_direction == "ge":
            use_split = values >= args.selector_threshold
        elif args.selector_direction == "lt":
            use_split = values < args.selector_threshold
        else:
            raise ValueError(f"Unsupported selector direction: {args.selector_direction}")
    if args.real_policy == "split":
        use_split = np.where(is_real, True, use_split)
    elif args.real_policy == "universal":
        use_split = np.where(is_real, False, use_split)
    elif args.real_policy == "rule":
        pass
    else:
        raise ValueError(f"Unsupported real policy: {args.real_policy}")
    df["use_split"] = use_split.astype(bool)
    df["final_score"] = np.where(df["use_split"], df["split_score"], df["universal_score"])
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--global-csv", type=Path, required=True)
    parser.add_argument("--raw-patch-csv", type=Path, required=True)
    parser.add_argument("--persistence-csv", type=Path, required=True)
    parser.add_argument("--persistence-score-col", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--global-score-col", default="final_score")
    parser.add_argument("--raw-patch-score-col", default="final_score")
    parser.add_argument("--base-alpha", type=float, default=0.60)
    parser.add_argument("--base-space", choices=["rank", "raw"], default="rank")
    parser.add_argument("--universal-weight", type=float, default=0.15)
    parser.add_argument("--real-weight", type=float, default=0.25)
    parser.add_argument("--fake-weight", type=float, default=0.25)
    parser.add_argument("--source-gate-threshold", type=float, default=0.45)
    parser.add_argument("--disagreement-threshold", type=float, default=0.20)
    parser.add_argument("--confidence-threshold", type=float, default=0.30)
    parser.add_argument("--persistence-scale", type=float, default=1.0)
    parser.add_argument("--fallback-threshold", type=float, default=0.0, help="Deprecated alias for default selector threshold")
    parser.add_argument("--selector-feature", default="persistence_minus_base")
    parser.add_argument("--selector-threshold", type=float, default=None)
    parser.add_argument("--selector-direction", choices=["ge", "lt"], default="ge")
    parser.add_argument("--real-policy", choices=["split", "universal", "rule"], default="rule")
    parser.add_argument("--minimal-output", action="store_true")
    args = parser.parse_args()
    if args.selector_threshold is None:
        args.selector_threshold = args.fallback_threshold

    df = make_scores(args)
    output_cols = KEY_COLUMNS + ["final_score"]
    if not args.minimal_output:
        output_cols += [
            "base_score",
            "universal_score",
            "split_score",
            "global_rank",
            "raw_rank",
            "persistence_rank_raw",
            "persistence_rank",
            "persistence_minus_base",
            "split_minus_universal",
            "sample_gate",
            "source_gate_mean",
            "use_split",
        ]
    metrics = _metrics_by_source(df, "final_score")
    metrics.insert(0, "dataset", args.dataset)
    metrics.insert(1, "score_name", "sample_fallback_persistence_minus_base_ge0")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    df[output_cols].to_csv(args.output_csv, index=False)
    metrics.to_csv(args.metrics_csv, index=False)

    avg = metrics[metrics["source_model"] == "Average"].iloc[0]
    print(f"{args.dataset}: Average AUC={avg.auc:.4f} AP={avg.ap:.4f} rows={len(df)}")
    print(f"Saved final scores -> {args.output_csv}")
    print(f"Saved metrics -> {args.metrics_csv}")


if __name__ == "__main__":
    main()
