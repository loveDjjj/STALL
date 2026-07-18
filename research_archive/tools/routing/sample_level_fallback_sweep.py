#!/usr/bin/env python3
"""Sample-level fallback between universal and split persistence scores.

The source-level predictor was too weak. This tool tries a simpler question:
can per-sample confidence/disagreement signals decide when to use the stronger
split score instead of the universal sample gate?
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


def _parse_floats(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


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
    if args.base_space == "rank":
        df["base_score"] = args.base_alpha * df["global_rank"] + (1.0 - args.base_alpha) * df["raw_rank"]
    elif args.base_space == "raw":
        df["base_score"] = args.base_alpha * df["global_score"] + (1.0 - args.base_alpha) * df["raw_patch_score"]
    else:
        raise ValueError(f"Unsupported base space: {args.base_space}")
    df["disagreement"] = np.abs(df["global_rank"] - df["raw_rank"])
    df["persistence_conf"] = np.abs(df["persistence_rank"] - 0.5) * 2.0
    df["persistence_margin"] = np.abs(df["persistence_rank"] - df["base_score"])
    df["persistence_minus_base"] = df["persistence_rank"] - df["base_score"]
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
    df["split_minus_universal"] = df["split_score"] - df["universal_score"]
    return df


def _selector(df: pd.DataFrame, feature: str, threshold: float, direction: str, real_policy: str) -> np.ndarray:
    if feature == "always":
        use_split = np.ones(len(df), dtype=bool)
    elif feature == "never":
        use_split = np.zeros(len(df), dtype=bool)
    else:
        values = df[feature].to_numpy(float)
        if direction == "ge":
            use_split = values >= threshold
        elif direction == "lt":
            use_split = values < threshold
        else:
            raise ValueError(f"Unknown direction: {direction}")
    is_real = df["subset"].str.lower() == "real"
    if real_policy == "split":
        use_split = np.where(is_real, True, use_split)
    elif real_policy == "universal":
        use_split = np.where(is_real, False, use_split)
    elif real_policy == "rule":
        pass
    else:
        raise ValueError(f"Unknown real policy: {real_policy}")
    return use_split.astype(bool)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _make_scores(args)
    base = _metrics_by_source(df.assign(final_score=df["base_score"]), "final_score").rename(
        columns={"auc": "base_auc", "ap": "base_ap"}
    )
    universal = _metrics_by_source(df.assign(final_score=df["universal_score"]), "final_score").rename(
        columns={"auc": "universal_auc", "ap": "universal_ap"}
    )
    split = _metrics_by_source(df.assign(final_score=df["split_score"]), "final_score").rename(
        columns={"auc": "split_auc", "ap": "split_ap"}
    )
    reference = base.merge(universal, on=["source_model", "n_fake"]).merge(split, on=["source_model", "n_fake"])

    rows = []
    per_source_rows = []
    features = [f.strip() for f in args.features.split(",") if f.strip()]
    for feature in features:
        thresholds = [np.nan] if feature in {"always", "never"} else _parse_floats(args.thresholds)
        directions = ["ge"] if feature in {"always", "never"} else ["ge", "lt"]
        for threshold in thresholds:
            for direction in directions:
                for real_policy in [x.strip() for x in args.real_policies.split(",") if x.strip()]:
                    use_split = _selector(df, feature, threshold, direction, real_policy)
                    final = np.where(use_split, df["split_score"], df["universal_score"])
                    metrics = _metrics_by_source(df.assign(final_score=final), "final_score").rename(
                        columns={"auc": "fallback_auc", "ap": "fallback_ap"}
                    )
                    merged = reference.merge(metrics, on=["source_model", "n_fake"])
                    merged["delta_vs_base_auc"] = merged["fallback_auc"] - merged["base_auc"]
                    merged["delta_vs_base_ap"] = merged["fallback_ap"] - merged["base_ap"]
                    merged["delta_vs_universal_auc"] = merged["fallback_auc"] - merged["universal_auc"]
                    merged["delta_vs_universal_ap"] = merged["fallback_ap"] - merged["universal_ap"]
                    merged["delta_vs_split_auc"] = merged["fallback_auc"] - merged["split_auc"]
                    merged["delta_vs_split_ap"] = merged["fallback_ap"] - merged["split_ap"]
                    rows.append(
                        {
                            "dataset": args.dataset,
                            "feature": feature,
                            "threshold": threshold,
                            "direction": direction,
                            "real_policy": real_policy,
                            "split_rate_all": float(use_split.mean()),
                            "split_rate_fake": float(use_split[df["subset"].str.lower() != "real"].mean()),
                            "mean_delta_vs_base_auc": float(merged["delta_vs_base_auc"].mean()),
                            "min_delta_vs_base_auc": float(merged["delta_vs_base_auc"].min()),
                            "mean_delta_vs_base_ap": float(merged["delta_vs_base_ap"].mean()),
                            "min_delta_vs_base_ap": float(merged["delta_vs_base_ap"].min()),
                            "mean_delta_vs_universal_auc": float(merged["delta_vs_universal_auc"].mean()),
                            "min_delta_vs_universal_auc": float(merged["delta_vs_universal_auc"].min()),
                            "mean_delta_vs_universal_ap": float(merged["delta_vs_universal_ap"].mean()),
                            "min_delta_vs_universal_ap": float(merged["delta_vs_universal_ap"].min()),
                            "mean_delta_vs_split_auc": float(merged["delta_vs_split_auc"].mean()),
                            "min_delta_vs_split_auc": float(merged["delta_vs_split_auc"].min()),
                            "mean_delta_vs_split_ap": float(merged["delta_vs_split_ap"].mean()),
                            "min_delta_vs_split_ap": float(merged["delta_vs_split_ap"].min()),
                            "n_sources": len(merged),
                        }
                    )
                    if args.write_per_source:
                        detail = merged.copy()
                        detail.insert(0, "dataset", args.dataset)
                        detail.insert(1, "feature", feature)
                        detail.insert(2, "threshold", threshold)
                        detail.insert(3, "direction", direction)
                        detail.insert(4, "real_policy", real_policy)
                        per_source_rows.append(detail)
    summary = pd.DataFrame(rows).sort_values(
        ["min_delta_vs_universal_auc", "mean_delta_vs_universal_auc", "min_delta_vs_base_auc"],
        ascending=False,
    )
    per_source = pd.concat(per_source_rows, ignore_index=True) if per_source_rows else pd.DataFrame()
    return summary, per_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--global-csv", type=Path, required=True)
    parser.add_argument("--raw-patch-csv", type=Path, required=True)
    parser.add_argument("--persistence-csv", type=Path, required=True)
    parser.add_argument("--persistence-score-col", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--per-source-csv", type=Path, default=None)
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
    parser.add_argument(
        "--features",
        default=(
            "always,never,sample_gate,disagreement,persistence_conf,persistence_margin,"
            "persistence_minus_base,split_minus_universal"
        ),
    )
    parser.add_argument("--thresholds", default="-0.20,-0.10,-0.05,0.00,0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70")
    parser.add_argument("--real-policies", default="split,universal,rule")
    parser.add_argument("--write-per-source", action="store_true")
    args = parser.parse_args()

    summary, per_source = run(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    if args.per_source_csv is not None:
        per_source.to_csv(args.per_source_csv, index=False)
    print(summary.head(30).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved sample-level fallback sweep -> {args.output_csv}")
    if args.per_source_csv is not None:
        print(f"Saved per-source details -> {args.per_source_csv}")


if __name__ == "__main__":
    main()
