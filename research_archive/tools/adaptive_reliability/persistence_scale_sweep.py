#!/usr/bin/env python3
"""Sweep persistence-rank scaling for the raw-base split-minus fallback."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


KEY_COLUMNS = ["subset", "source_model", "filename"]


def _read(path: Path, score_col: str, out_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in [*KEY_COLUMNS, score_col] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    out = df[[*KEY_COLUMNS, score_col]].copy()
    for col in KEY_COLUMNS:
        out[col] = out[col].astype(str)
    return out.rename(columns={score_col: out_col})


def _rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(1.0, float(len(values) - 1))


def _auc_ap(real_scores: pd.Series, fake_scores: pd.Series) -> tuple[float, float]:
    y = [1] * len(real_scores) + [0] * len(fake_scores)
    s = [*real_scores.astype(float), *fake_scores.astype(float)]
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def _source_metrics(df: pd.DataFrame, score_col: str, prefix: str) -> pd.DataFrame:
    real = df[df["subset"].str.lower() == "real"]
    fake = df[df["subset"].str.lower() != "real"]
    real_scores = real[score_col]
    rows = []
    for source, group in fake.groupby("source_model", sort=True):
        auc, ap = _auc_ap(real_scores, group[score_col])
        rows.append({"source_model": source, f"{prefix}_auc": auc, f"{prefix}_ap": ap, "n_fake": len(group)})
    return pd.DataFrame(rows)


def _parse_floats(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def _make_base(args: argparse.Namespace) -> pd.DataFrame:
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
    df["base_score"] = args.base_alpha * df["global_score"] + (1.0 - args.base_alpha) * df["raw_patch_score"]
    df["default_score"] = df["base_score"]
    df["disagreement"] = np.abs(df["global_rank"] - df["raw_rank"])
    return df


def _apply_scaled_fallback(df: pd.DataFrame, args: argparse.Namespace, scale: float) -> pd.DataFrame:
    out = df.copy()
    out["persistence_rank"] = np.clip(0.5 + scale * (out["persistence_rank_raw"] - 0.5), 0.0, 1.0)
    out["persistence_conf"] = np.abs(out["persistence_rank"] - 0.5) * 2.0
    out["sample_gate"] = (
        (out["disagreement"].to_numpy(float) >= args.disagreement_threshold)
        & (out["persistence_conf"].to_numpy(float) >= args.confidence_threshold)
    ).astype(float)
    source_gate_mean = (
        out[out["subset"].str.lower() != "real"].groupby("source_model")["sample_gate"].mean().rename("source_gate_mean")
    )
    out = out.merge(source_gate_mean, on="source_model", how="left")
    out["source_gate_mean"] = out["source_gate_mean"].fillna(0.0)

    universal_weight = args.universal_weight * out["sample_gate"]
    out["universal_score"] = (1.0 - universal_weight) * out["base_score"] + universal_weight * out["persistence_rank"]

    is_real = out["subset"].str.lower() == "real"
    fake_source_enabled = (out["source_gate_mean"] >= args.source_gate_threshold) & (~is_real)
    split_weight = out["sample_gate"] * (
        args.real_weight * is_real.to_numpy(float) + args.fake_weight * fake_source_enabled.to_numpy(float)
    )
    out["split_score"] = (1.0 - split_weight) * out["base_score"] + split_weight * out["persistence_rank"]
    out["split_minus_universal"] = out["split_score"] - out["universal_score"]
    use_split = out["split_minus_universal"].to_numpy(float) < args.selector_threshold
    if args.real_policy == "split":
        use_split = np.where(is_real, True, use_split)
    elif args.real_policy == "universal":
        use_split = np.where(is_real, False, use_split)
    elif args.real_policy != "rule":
        raise ValueError(f"Unsupported real policy: {args.real_policy}")
    out["final_score"] = np.where(use_split, out["split_score"], out["universal_score"])
    return out


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = _make_base(args)
    default_metrics = _source_metrics(base, "default_score", "default")
    summaries = []
    per_source_rows = []
    for scale in _parse_floats(args.scales):
        scored = _apply_scaled_fallback(base, args, scale)
        metrics = _source_metrics(scored, "final_score", "fallback")
        audit = default_metrics.merge(metrics, on=["source_model", "n_fake"], validate="one_to_one")
        audit["delta_auc"] = audit["fallback_auc"] - audit["default_auc"]
        audit["delta_ap"] = audit["fallback_ap"] - audit["default_ap"]
        audit.insert(0, "scale", scale)
        audit.insert(0, "dataset", args.dataset)
        per_source_rows.append(audit)
        summaries.append(
            {
                "dataset": args.dataset,
                "scale": scale,
                "n_sources": int(len(audit)),
                "n_negative_auc": int((audit["delta_auc"] < 0).sum()),
                "n_negative_ap": int((audit["delta_ap"] < 0).sum()),
                "mean_delta_auc": float(audit["delta_auc"].mean()),
                "min_delta_auc": float(audit["delta_auc"].min()),
                "mean_delta_ap": float(audit["delta_ap"].mean()),
                "min_delta_ap": float(audit["delta_ap"].min()),
                "weighted_mean_delta_auc": float((audit["delta_auc"] * audit["n_fake"]).sum() / audit["n_fake"].sum()),
                "weighted_mean_delta_ap": float((audit["delta_ap"] * audit["n_fake"]).sum() / audit["n_fake"].sum()),
                "worst_auc_source": str(audit.sort_values(["delta_auc", "delta_ap"]).iloc[0]["source_model"]),
                "worst_ap_source": str(audit.sort_values(["delta_ap", "delta_auc"]).iloc[0]["source_model"]),
            }
        )
    summary = pd.DataFrame(summaries).sort_values(
        ["n_negative_auc", "n_negative_ap", "min_delta_ap", "min_delta_auc"],
        ascending=[True, True, False, False],
    )
    per_source = pd.concat(per_source_rows, ignore_index=True)
    return summary, per_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--global-csv", type=Path, required=True)
    parser.add_argument("--raw-patch-csv", type=Path, required=True)
    parser.add_argument("--persistence-csv", type=Path, required=True)
    parser.add_argument("--persistence-score-col", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--per-source-csv", type=Path, required=True)
    parser.add_argument("--global-score-col", default="final_score")
    parser.add_argument("--raw-patch-score-col", default="final_score")
    parser.add_argument("--base-alpha", type=float, default=0.60)
    parser.add_argument("--universal-weight", type=float, default=0.15)
    parser.add_argument("--real-weight", type=float, default=0.25)
    parser.add_argument("--fake-weight", type=float, default=0.25)
    parser.add_argument("--source-gate-threshold", type=float, default=0.45)
    parser.add_argument("--disagreement-threshold", type=float, default=0.20)
    parser.add_argument("--confidence-threshold", type=float, default=0.30)
    parser.add_argument("--selector-threshold", type=float, default=0.0)
    parser.add_argument("--real-policy", choices=["split", "universal", "rule"], default="split")
    parser.add_argument("--scales", default="0.50,0.75,0.90,1.00,1.10,1.25,1.50,2.00")
    args = parser.parse_args()

    summary, per_source = run(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.per_source_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    per_source.to_csv(args.per_source_csv, index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved scale sweep -> {args.output_csv}")
    print(f"Saved per-source sweep -> {args.per_source_csv}")


if __name__ == "__main__":
    main()
