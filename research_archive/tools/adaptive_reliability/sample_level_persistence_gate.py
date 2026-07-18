#!/usr/bin/env python3
"""Sample-level gate for persistence intervention.

Default score is global+raw_patch. Persistence can intervene only when:
- global and raw patch disagree enough; and
- persistence has enough confidence relative to its real-calibrated score.

The gate is per sample, not per dataset.
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


def run(args: argparse.Namespace) -> pd.DataFrame:
    global_df = _read(args.global_csv, args.global_score_col, "global_score")
    raw_df = _read(args.raw_patch_csv, args.raw_patch_score_col, "raw_patch_score")
    persistence_df = _read(args.persistence_csv, args.persistence_score_col, "persistence_score")
    df = global_df.merge(raw_df, on=KEY_COLUMNS, how="inner", validate="one_to_one").merge(
        persistence_df, on=KEY_COLUMNS, how="inner", validate="one_to_one"
    )
    df["global_rank"] = _rank01(df["global_score"].to_numpy(float))
    df["raw_rank"] = _rank01(df["raw_patch_score"].to_numpy(float))
    df["persistence_rank"] = _rank01(df["persistence_score"].to_numpy(float))

    base_alpha = args.base_alpha
    base = base_alpha * df["global_rank"].to_numpy(float) + (1.0 - base_alpha) * df["raw_rank"].to_numpy(float)
    persistence = df["persistence_rank"].to_numpy(float)
    global_rank = df["global_rank"].to_numpy(float)
    raw_rank = df["raw_rank"].to_numpy(float)
    disagreement = np.abs(global_rank - raw_rank)
    persistence_conf = np.abs(persistence - 0.5) * 2.0

    base_auc, base_ap = _metrics(df.assign(final_score=base), "final_score")
    rows = [
        {
            "dataset": args.dataset,
            "persistence_col": args.persistence_score_col,
            "base_alpha": base_alpha,
            "intervention_weight": 0.0,
            "gate_mode": "base",
            "disagreement_threshold": np.nan,
            "confidence_threshold": np.nan,
            "avg_auc": base_auc,
            "avg_ap": base_ap,
            "base_auc": base_auc,
            "base_ap": base_ap,
            "delta_auc": 0.0,
            "delta_ap": 0.0,
            "gate_mean": 0.0,
            "n_rows": len(df),
        }
    ]

    for weight in _parse_floats(args.intervention_weights):
        for mode in [m.strip() for m in args.gate_modes.split(",") if m.strip()]:
            for dis_t in _parse_floats(args.disagreement_thresholds):
                for conf_t in _parse_floats(args.confidence_thresholds):
                    if mode == "disagreement":
                        gate = (disagreement >= dis_t).astype(float)
                    elif mode == "confidence":
                        gate = (persistence_conf >= conf_t).astype(float)
                    elif mode == "both":
                        gate = ((disagreement >= dis_t) & (persistence_conf >= conf_t)).astype(float)
                    elif mode == "soft_product":
                        gate = np.clip(disagreement * persistence_conf, 0.0, 1.0)
                    elif mode == "soft_conf":
                        gate = np.clip(persistence_conf, 0.0, 1.0)
                    elif mode == "soft_disagreement":
                        gate = np.clip(disagreement, 0.0, 1.0)
                    else:
                        raise ValueError(f"Unknown gate mode: {mode}")
                    final = (1.0 - weight * gate) * base + (weight * gate) * persistence
                    auc, ap = _metrics(df.assign(final_score=final), "final_score")
                    rows.append(
                        {
                            "dataset": args.dataset,
                            "persistence_col": args.persistence_score_col,
                            "base_alpha": base_alpha,
                            "intervention_weight": weight,
                            "gate_mode": mode,
                            "disagreement_threshold": dis_t,
                            "confidence_threshold": conf_t,
                            "avg_auc": auc,
                            "avg_ap": ap,
                            "base_auc": base_auc,
                            "base_ap": base_ap,
                            "delta_auc": auc - base_auc,
                            "delta_ap": ap - base_ap,
                            "gate_mean": float(gate.mean()),
                            "n_rows": len(df),
                        }
                    )
    return pd.DataFrame(rows).sort_values(["avg_auc", "avg_ap"], ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--global-csv", type=Path, required=True)
    parser.add_argument("--raw-patch-csv", type=Path, required=True)
    parser.add_argument("--persistence-csv", type=Path, required=True)
    parser.add_argument("--persistence-score-col", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--global-score-col", default="final_score")
    parser.add_argument("--raw-patch-score-col", default="final_score")
    parser.add_argument("--base-alpha", type=float, default=0.60)
    parser.add_argument("--intervention-weights", default="0.05,0.10,0.15,0.20")
    parser.add_argument("--gate-modes", default="disagreement,confidence,both,soft_product,soft_conf,soft_disagreement")
    parser.add_argument("--disagreement-thresholds", default="0.10,0.20,0.30")
    parser.add_argument("--confidence-thresholds", default="0.30,0.50,0.70")
    args = parser.parse_args()

    out = run(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(
        out.head(25).to_string(
            index=False,
            formatters={
                "avg_auc": lambda x: f"{x:.4f}",
                "avg_ap": lambda x: f"{x:.4f}",
                "delta_auc": lambda x: f"{x:+.4f}",
                "delta_ap": lambda x: f"{x:+.4f}",
                "gate_mean": lambda x: f"{x:.4f}",
            },
        )
    )
    print(f"Saved sample-level gate sweep -> {args.output_csv}")


if __name__ == "__main__":
    main()
