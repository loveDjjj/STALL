#!/usr/bin/env python3
"""Held-out source-family validation for split persistence gates.

For each fake source_model, evaluate with that source held out from threshold
selection. The fixed-threshold row tests direct transfer of a chosen source
stat threshold. The leave-one-source row chooses the threshold on all other fake
sources, then evaluates only on the held-out fake source.
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
    elif args.gate_mode == "disagreement":
        gate = df["disagreement"].to_numpy(float) >= args.disagreement_threshold
    elif args.gate_mode == "confidence":
        gate = df["persistence_conf"].to_numpy(float) >= args.confidence_threshold
    else:
        raise ValueError(f"Unsupported gate mode: {args.gate_mode}")
    df["sample_gate"] = gate.astype(float)
    return df


def _source_stats(df: pd.DataFrame) -> pd.DataFrame:
    fake = df[df["subset"].str.lower() != "real"]
    rows = []
    for source, group in fake.groupby("source_model", sort=True):
        rows.append(
            {
                "source_model": source,
                "n_fake": len(group),
                "fake_gate_mean": float(group["sample_gate"].mean()),
                "fake_disagreement_mean": float(group["disagreement"].mean()),
                "fake_conf_mean": float(group["persistence_conf"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _split_score(
    df: pd.DataFrame,
    enabled_sources: set[str],
    real_weight: float,
    fake_weight: float,
) -> np.ndarray:
    is_real = df["subset"].str.lower() == "real"
    fake_enabled = df["source_model"].isin(enabled_sources) & (~is_real)
    weight = df["sample_gate"].to_numpy(float) * (
        real_weight * is_real.to_numpy(float) + fake_weight * fake_enabled.to_numpy(float)
    )
    return (1.0 - weight) * df["base_score"].to_numpy(float) + weight * df["persistence_rank"].to_numpy(float)


def _universal_score(df: pd.DataFrame, weight: float) -> np.ndarray:
    gate = df["sample_gate"].to_numpy(float)
    return (1.0 - weight * gate) * df["base_score"].to_numpy(float) + (weight * gate) * df[
        "persistence_rank"
    ].to_numpy(float)


def _per_source_metrics(df: pd.DataFrame, score: np.ndarray) -> dict[str, tuple[float, float]]:
    temp = df.assign(_score=score)
    real = temp[temp["subset"].str.lower() == "real"]
    fake = temp[temp["subset"].str.lower() != "real"]
    real_scores = real["_score"].to_numpy(float)
    out = {}
    for source, group in fake.groupby("source_model", sort=True):
        out[source] = _score_metrics(real_scores, group["_score"].to_numpy(float))
    return out


def _training_mean_delta(
    df: pd.DataFrame,
    heldout_source: str,
    enabled_sources: set[str],
    real_weight: float,
    fake_weight: float,
) -> tuple[float, float]:
    base_metrics = _per_source_metrics(df, df["base_score"].to_numpy(float))
    split_metrics = _per_source_metrics(df, _split_score(df, enabled_sources, real_weight, fake_weight))
    auc_deltas = []
    ap_deltas = []
    for source in base_metrics:
        if source == heldout_source:
            continue
        auc_deltas.append(split_metrics[source][0] - base_metrics[source][0])
        ap_deltas.append(split_metrics[source][1] - base_metrics[source][1])
    if not auc_deltas:
        return np.nan, np.nan
    return float(np.mean(auc_deltas)), float(np.mean(ap_deltas))


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _make_scores(args)
    stats = _source_stats(df)
    source_stat = args.source_stat_name
    if source_stat not in stats.columns:
        raise ValueError(f"Unknown source stat: {source_stat}")

    base_metrics = _per_source_metrics(df, df["base_score"].to_numpy(float))
    universal_metrics = _per_source_metrics(df, _universal_score(df, args.reference_weight))
    thresholds = _parse_floats(args.thresholds)
    rows = []
    for heldout_source in sorted(base_metrics):
        fixed_enabled = set(stats.loc[stats[source_stat] >= args.fixed_threshold, "source_model"].astype(str))
        fixed_score = _split_score(df, fixed_enabled, args.real_weight, args.fake_weight)
        fixed_metrics = _per_source_metrics(df, fixed_score)

        rows.append(
            {
                "dataset": args.dataset,
                "validation_mode": "fixed_threshold",
                "heldout_source": heldout_source,
                "selected_threshold": args.fixed_threshold,
                "enabled_sources": ";".join(sorted(fixed_enabled)),
                "heldout_enabled": heldout_source in fixed_enabled,
                "train_delta_auc": np.nan,
                "train_delta_ap": np.nan,
                "base_auc": base_metrics[heldout_source][0],
                "base_ap": base_metrics[heldout_source][1],
                "universal_auc": universal_metrics[heldout_source][0],
                "universal_ap": universal_metrics[heldout_source][1],
                "split_auc": fixed_metrics[heldout_source][0],
                "split_ap": fixed_metrics[heldout_source][1],
                "delta_auc": fixed_metrics[heldout_source][0] - base_metrics[heldout_source][0],
                "delta_ap": fixed_metrics[heldout_source][1] - base_metrics[heldout_source][1],
                "delta_vs_universal_auc": fixed_metrics[heldout_source][0] - universal_metrics[heldout_source][0],
                "delta_vs_universal_ap": fixed_metrics[heldout_source][1] - universal_metrics[heldout_source][1],
                "heldout_source_stat": float(
                    stats.loc[stats["source_model"] == heldout_source, source_stat].iloc[0]
                ),
                "n_fake": int(stats.loc[stats["source_model"] == heldout_source, "n_fake"].iloc[0]),
            }
        )

        best = None
        for threshold in thresholds:
            enabled = set(stats.loc[stats[source_stat] >= threshold, "source_model"].astype(str))
            train_auc_delta, train_ap_delta = _training_mean_delta(
                df,
                heldout_source=heldout_source,
                enabled_sources=enabled,
                real_weight=args.real_weight,
                fake_weight=args.fake_weight,
            )
            score = _split_score(df, enabled, args.real_weight, args.fake_weight)
            metrics = _per_source_metrics(df, score)
            candidate = {
                "threshold": threshold,
                "enabled": enabled,
                "train_auc_delta": train_auc_delta,
                "train_ap_delta": train_ap_delta,
                "split_auc": metrics[heldout_source][0],
                "split_ap": metrics[heldout_source][1],
            }
            if best is None or (
                candidate["train_auc_delta"],
                candidate["train_ap_delta"],
                -abs(candidate["threshold"] - args.fixed_threshold),
            ) > (
                best["train_auc_delta"],
                best["train_ap_delta"],
                -abs(best["threshold"] - args.fixed_threshold),
            ):
                best = candidate
        if best is not None:
            rows.append(
                {
                    "dataset": args.dataset,
                    "validation_mode": "leave_one_source_select_threshold",
                    "heldout_source": heldout_source,
                    "selected_threshold": best["threshold"],
                    "enabled_sources": ";".join(sorted(best["enabled"])),
                    "heldout_enabled": heldout_source in best["enabled"],
                    "train_delta_auc": best["train_auc_delta"],
                    "train_delta_ap": best["train_ap_delta"],
                    "base_auc": base_metrics[heldout_source][0],
                    "base_ap": base_metrics[heldout_source][1],
                    "universal_auc": universal_metrics[heldout_source][0],
                    "universal_ap": universal_metrics[heldout_source][1],
                    "split_auc": best["split_auc"],
                    "split_ap": best["split_ap"],
                    "delta_auc": best["split_auc"] - base_metrics[heldout_source][0],
                    "delta_ap": best["split_ap"] - base_metrics[heldout_source][1],
                    "delta_vs_universal_auc": best["split_auc"] - universal_metrics[heldout_source][0],
                    "delta_vs_universal_ap": best["split_ap"] - universal_metrics[heldout_source][1],
                    "heldout_source_stat": float(
                        stats.loc[stats["source_model"] == heldout_source, source_stat].iloc[0]
                    ),
                    "n_fake": int(stats.loc[stats["source_model"] == heldout_source, "n_fake"].iloc[0]),
                }
            )
    return pd.DataFrame(rows), stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--global-csv", type=Path, required=True)
    parser.add_argument("--raw-patch-csv", type=Path, required=True)
    parser.add_argument("--persistence-csv", type=Path, required=True)
    parser.add_argument("--persistence-score-col", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--source-stats-csv", type=Path, default=None)
    parser.add_argument("--global-score-col", default="final_score")
    parser.add_argument("--raw-patch-score-col", default="final_score")
    parser.add_argument("--base-alpha", type=float, default=0.60)
    parser.add_argument("--reference-weight", type=float, default=0.15)
    parser.add_argument("--real-weight", type=float, default=0.25)
    parser.add_argument("--fake-weight", type=float, default=0.25)
    parser.add_argument("--gate-mode", default="both")
    parser.add_argument("--disagreement-threshold", type=float, default=0.20)
    parser.add_argument("--confidence-threshold", type=float, default=0.30)
    parser.add_argument("--source-stat-name", default="fake_gate_mean")
    parser.add_argument("--fixed-threshold", type=float, default=0.45)
    parser.add_argument("--thresholds", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50")
    args = parser.parse_args()

    out, stats = run(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    if args.source_stats_csv is not None:
        args.source_stats_csv.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(args.source_stats_csv, index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved held-out source validation -> {args.output_csv}")
    if args.source_stats_csv is not None:
        print(f"Saved source stats -> {args.source_stats_csv}")


if __name__ == "__main__":
    main()
