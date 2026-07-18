#!/usr/bin/env python3
"""Split real-side correction from fake-side source routing.

The previous source-aware diagnostic mixed two effects:
- real videos could receive persistence correction;
- fake videos could receive source-family-gated persistence intervention.

This tool sweeps separate real_weight and fake_weight values to test whether
the two effects can be tuned independently under the same metric protocol.
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


def _metrics(df: pd.DataFrame, score_col: str) -> tuple[float, float]:
    real = df[df["subset"].str.lower() == "real"]
    fake = df[df["subset"].str.lower() != "real"]
    aucs = []
    aps = []
    real_scores = real[score_col].to_numpy(float)
    for _, group in fake.groupby("source_model", sort=True):
        auc, ap = _score_metrics(real_scores, group[score_col].to_numpy(float))
        aucs.append(auc)
        aps.append(ap)
    return float(np.mean(aucs)), float(np.mean(aps))


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


def _source_stats(df: pd.DataFrame, reference_weight: float) -> pd.DataFrame:
    reference = (1.0 - reference_weight * df["sample_gate"]) * df["base_score"] + (
        reference_weight * df["sample_gate"]
    ) * df["persistence_rank"]
    df = df.assign(reference_score=reference)
    real = df[df["subset"].str.lower() == "real"]
    fake = df[df["subset"].str.lower() != "real"]
    real_base = real["base_score"].to_numpy(float)
    real_ref = real["reference_score"].to_numpy(float)
    rows = []
    for source, group in fake.groupby("source_model", sort=True):
        base_auc, base_ap = _score_metrics(real_base, group["base_score"].to_numpy(float))
        ref_auc, ref_ap = _score_metrics(real_ref, group["reference_score"].to_numpy(float))
        rows.append(
            {
                "source_model": source,
                "n_fake": len(group),
                "source_delta_auc": ref_auc - base_auc,
                "source_delta_ap": ref_ap - base_ap,
                "fake_gate_mean": float(group["sample_gate"].mean()),
                "fake_disagreement_mean": float(group["disagreement"].mean()),
                "fake_conf_mean": float(group["persistence_conf"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _enabled_sources(args: argparse.Namespace, stats: pd.DataFrame, stat_name: str, threshold: float) -> set[str]:
    if args.source_mode == "oracle_positive":
        mask = (stats["source_delta_auc"] > args.oracle_min_delta_auc) & (
            stats["source_delta_ap"] > args.oracle_min_delta_ap
        )
    elif args.source_mode == "stat_threshold":
        if stat_name not in stats.columns:
            raise ValueError(f"Unknown source stat: {stat_name}")
        mask = stats[stat_name] >= threshold
    else:
        raise ValueError(f"Unknown source mode: {args.source_mode}")
    return set(stats.loc[mask, "source_model"].astype(str))


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _make_scores(args)
    stats = _source_stats(df, args.reference_weight)
    base_auc, base_ap = _metrics(df.assign(final_score=df["base_score"]), "final_score")
    universal = (1.0 - args.reference_weight * df["sample_gate"]) * df["base_score"] + (
        args.reference_weight * df["sample_gate"]
    ) * df["persistence_rank"]
    universal_auc, universal_ap = _metrics(df.assign(final_score=universal), "final_score")

    is_real = df["subset"].str.lower() == "real"
    rows = []
    stat_names = ["oracle"] if args.source_mode == "oracle_positive" else [
        s.strip() for s in args.source_stat_names.split(",") if s.strip()
    ]
    stat_thresholds = [np.nan] if args.source_mode == "oracle_positive" else _parse_floats(args.stat_thresholds)
    for stat_name in stat_names:
        for stat_threshold in stat_thresholds:
            enabled = _enabled_sources(args, stats, stat_name, stat_threshold)
            fake_source_prior = df["source_model"].isin(enabled).astype(float)
            fake_source_prior = fake_source_prior.mask(is_real, 0.0)
            real_prior = is_real.astype(float)
            fake_prior = (~is_real).astype(float) * fake_source_prior
            for real_weight in _parse_floats(args.real_weights):
                for fake_weight in _parse_floats(args.fake_weights):
                    effective_weight = df["sample_gate"] * (real_weight * real_prior + fake_weight * fake_prior)
                    final = (1.0 - effective_weight) * df["base_score"] + effective_weight * df["persistence_rank"]
                    auc, ap = _metrics(df.assign(final_score=final), "final_score")
                    rows.append(
                        {
                            "dataset": args.dataset,
                            "persistence_col": args.persistence_score_col,
                            "base_alpha": args.base_alpha,
                            "gate_mode": args.gate_mode,
                            "disagreement_threshold": args.disagreement_threshold,
                            "confidence_threshold": args.confidence_threshold,
                            "source_mode": args.source_mode,
                            "source_stat_name": stat_name,
                            "source_stat_threshold": stat_threshold,
                            "enabled_sources": ";".join(sorted(enabled)),
                            "n_enabled_sources": len(enabled),
                            "real_weight": real_weight,
                            "fake_weight": fake_weight,
                            "avg_auc": auc,
                            "avg_ap": ap,
                            "base_auc": base_auc,
                            "base_ap": base_ap,
                            "universal_auc": universal_auc,
                            "universal_ap": universal_ap,
                            "delta_auc": auc - base_auc,
                            "delta_ap": ap - base_ap,
                            "delta_vs_universal_auc": auc - universal_auc,
                            "delta_vs_universal_ap": ap - universal_ap,
                            "real_gate_mean": float((df["sample_gate"] * real_prior).mean()),
                            "fake_gate_mean": float((df["sample_gate"] * fake_prior).mean()),
                            "effective_weight_mean": float(effective_weight.mean()),
                            "n_rows": len(df),
                        }
                    )
    out = pd.DataFrame(rows).sort_values(["avg_auc", "avg_ap"], ascending=False)
    return out, stats


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
    parser.add_argument("--real-weights", default="0.00,0.05,0.10,0.15,0.20,0.25")
    parser.add_argument("--fake-weights", default="0.00,0.05,0.10,0.15,0.20,0.25")
    parser.add_argument("--gate-mode", default="both")
    parser.add_argument("--disagreement-threshold", type=float, default=0.20)
    parser.add_argument("--confidence-threshold", type=float, default=0.30)
    parser.add_argument("--source-mode", choices=["oracle_positive", "stat_threshold"], default="stat_threshold")
    parser.add_argument("--source-stat-names", default="fake_gate_mean")
    parser.add_argument("--stat-thresholds", default="0.45")
    parser.add_argument("--oracle-min-delta-auc", type=float, default=0.0)
    parser.add_argument("--oracle-min-delta-ap", type=float, default=0.0)
    args = parser.parse_args()

    out, stats = run(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    if args.source_stats_csv is not None:
        args.source_stats_csv.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(args.source_stats_csv, index=False)
    print(out.head(30).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved split real/fake gate sweep -> {args.output_csv}")
    if args.source_stats_csv is not None:
        print(f"Saved source stats -> {args.source_stats_csv}")


if __name__ == "__main__":
    main()
