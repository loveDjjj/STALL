#!/usr/bin/env python3
"""Source-aware persistence gate diagnostics.

This tool tests whether sample-level persistence intervention is mainly useful
for specific fake source families. It supports:

- oracle_positive: enable source gate only for fake source_model groups whose
  measured per-source delta is positive under the same sample gate.
- stat_threshold: enable source gate from label-free source statistics computed
  from fake rows, such as fake gate mean, disagreement, or confidence.

The score calculation intentionally matches sample_level_persistence_gate.py.
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
    df["universal_gated_score"] = (1.0 - args.intervention_weight * df["sample_gate"]) * df["base_score"] + (
        args.intervention_weight * df["sample_gate"]
    ) * df["persistence_rank"]
    return df


def _per_source_delta(df: pd.DataFrame) -> pd.DataFrame:
    real = df[df["subset"].str.lower() == "real"]
    fake = df[df["subset"].str.lower() != "real"]
    real_base = real["base_score"].to_numpy(float)
    real_gated = real["universal_gated_score"].to_numpy(float)
    rows = []
    for source, group in fake.groupby("source_model", sort=True):
        base_auc, base_ap = _score_metrics(real_base, group["base_score"].to_numpy(float))
        gated_auc, gated_ap = _score_metrics(real_gated, group["universal_gated_score"].to_numpy(float))
        rows.append(
            {
                "source_model": source,
                "n_fake": len(group),
                "base_auc": base_auc,
                "base_ap": base_ap,
                "universal_gated_auc": gated_auc,
                "universal_gated_ap": gated_ap,
                "source_delta_auc": gated_auc - base_auc,
                "source_delta_ap": gated_ap - base_ap,
                "fake_gate_mean": float(group["sample_gate"].mean()),
                "fake_disagreement_mean": float(group["disagreement"].mean()),
                "fake_conf_mean": float(group["persistence_conf"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _enabled_sources(
    mode: str,
    source_stats: pd.DataFrame,
    stat_name: str,
    stat_threshold: float,
    min_delta_auc: float,
    min_delta_ap: float,
) -> set[str]:
    if mode == "oracle_positive":
        mask = (source_stats["source_delta_auc"] > min_delta_auc) & (source_stats["source_delta_ap"] > min_delta_ap)
    elif mode == "stat_threshold":
        if stat_name not in source_stats.columns:
            raise ValueError(f"Unknown source stat: {stat_name}")
        mask = source_stats[stat_name] >= stat_threshold
    else:
        raise ValueError(f"Unknown source mode: {mode}")
    return set(source_stats.loc[mask, "source_model"].astype(str))


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _make_scores(args)
    base_auc, base_ap = _metrics(df.assign(final_score=df["base_score"]), "final_score")
    universal_auc, universal_ap = _metrics(
        df.assign(final_score=df["universal_gated_score"]),
        "final_score",
    )
    source_stats = _per_source_delta(df)

    rows = []
    for mode in [m.strip() for m in args.source_modes.split(",") if m.strip()]:
        if mode == "oracle_positive":
            thresholds = [(np.nan, args.oracle_min_delta_auc, args.oracle_min_delta_ap)]
            stat_names = ["oracle"]
        else:
            thresholds = [(t, np.nan, np.nan) for t in _parse_floats(args.stat_thresholds)]
            stat_names = [s.strip() for s in args.source_stat_names.split(",") if s.strip()]

        for stat_name in stat_names:
            for stat_t, min_auc, min_ap in thresholds:
                enabled = _enabled_sources(
                    mode=mode,
                    source_stats=source_stats,
                    stat_name=stat_name,
                    stat_threshold=float(stat_t) if not np.isnan(stat_t) else np.nan,
                    min_delta_auc=float(min_auc) if not np.isnan(min_auc) else 0.0,
                    min_delta_ap=float(min_ap) if not np.isnan(min_ap) else 0.0,
                )
                source_prior = df["source_model"].isin(enabled).astype(float)
                is_real = df["subset"].str.lower() == "real"
                if args.real_policy == "gate":
                    source_prior = source_prior.mask(is_real, 1.0)
                elif args.real_policy == "base":
                    source_prior = source_prior.mask(is_real, 0.0)
                else:
                    raise ValueError(f"Unknown real policy: {args.real_policy}")
                effective_gate = df["sample_gate"] * source_prior
                final_score = (1.0 - args.intervention_weight * effective_gate) * df["base_score"] + (
                    args.intervention_weight * effective_gate
                ) * df["persistence_rank"]
                auc, ap = _metrics(df.assign(final_score=final_score), "final_score")
                rows.append(
                    {
                        "dataset": args.dataset,
                        "persistence_col": args.persistence_score_col,
                        "base_alpha": args.base_alpha,
                        "intervention_weight": args.intervention_weight,
                        "gate_mode": args.gate_mode,
                        "disagreement_threshold": args.disagreement_threshold,
                        "confidence_threshold": args.confidence_threshold,
                        "source_mode": mode,
                        "source_stat_name": stat_name,
                        "source_stat_threshold": stat_t,
                        "real_policy": args.real_policy,
                        "enabled_sources": ";".join(sorted(enabled)),
                        "n_enabled_sources": len(enabled),
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
                        "effective_gate_mean": float(effective_gate.mean()),
                        "n_rows": len(df),
                    }
                )
    return pd.DataFrame(rows).sort_values(["avg_auc", "avg_ap"], ascending=False), source_stats


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
    parser.add_argument("--intervention-weight", type=float, default=0.15)
    parser.add_argument("--gate-mode", default="both")
    parser.add_argument("--disagreement-threshold", type=float, default=0.20)
    parser.add_argument("--confidence-threshold", type=float, default=0.30)
    parser.add_argument("--source-modes", default="oracle_positive,stat_threshold")
    parser.add_argument("--source-stat-names", default="fake_gate_mean,fake_disagreement_mean,fake_conf_mean")
    parser.add_argument("--stat-thresholds", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50")
    parser.add_argument("--oracle-min-delta-auc", type=float, default=0.0)
    parser.add_argument("--oracle-min-delta-ap", type=float, default=0.0)
    parser.add_argument("--real-policy", choices=["gate", "base"], default="gate")
    args = parser.parse_args()

    out, source_stats = run(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    if args.source_stats_csv is not None:
        args.source_stats_csv.parent.mkdir(parents=True, exist_ok=True)
        source_stats.to_csv(args.source_stats_csv, index=False)
    print(out.head(30).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved source-aware gate diagnostic -> {args.output_csv}")
    if args.source_stats_csv is not None:
        print(f"Saved source stats -> {args.source_stats_csv}")


if __name__ == "__main__":
    main()
