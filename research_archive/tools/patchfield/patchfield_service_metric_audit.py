#!/usr/bin/env python3
"""Audit PatchField corrections with real-only fixed-FPR service metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from patchfield_reliability_correction_sweep import KEY_COLUMNS, _read, _real_rank
from patchfield_stability_gate_sweep import FROZEN_EXTRA_COLUMNS, _correct, _features, _gate


def _parse_fprs(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _fixed_fpr_metrics(df: pd.DataFrame, score_col: str, fprs: list[float]) -> dict[str, float]:
    real_scores = df.loc[df["subset"].str.lower() == "real", score_col].to_numpy(float)
    fake = df[df["subset"].str.lower() != "real"]
    if len(real_scores) == 0 or len(fake) == 0:
        raise ValueError("Need both real and fake rows")

    out: dict[str, float] = {}
    for fpr in fprs:
        threshold = float(np.quantile(real_scores, fpr, method="higher"))
        recalls = []
        for _source, group in fake.groupby("source_model", sort=True):
            fake_scores = group[score_col].to_numpy(float)
            recalls.append(float(np.mean(fake_scores <= threshold)))
        tag = str(fpr).replace(".", "p")
        out[f"threshold_fpr{tag}"] = threshold
        out[f"fake_recall_mean_fpr{tag}"] = float(np.mean(recalls))
        out[f"fake_recall_min_fpr{tag}"] = float(np.min(recalls))
    return out


def _load_scores(args: argparse.Namespace) -> pd.DataFrame:
    frozen = _read(args.frozen_final_scores_csv, FROZEN_EXTRA_COLUMNS).rename(
        columns={"final_score": "frozen_final_score"}
    )
    patch = _read(args.patchfield_scores_csv, [args.patchfield_score_col])
    df = frozen.merge(patch, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(df) == 0:
        raise ValueError("No overlap between frozen scores and PatchField scores")

    real_mask = df["subset"].str.lower() == "real"
    df["patchfield_rank"] = _real_rank(df[args.patchfield_score_col], real_mask)
    return df


def run(args: argparse.Namespace) -> pd.DataFrame:
    df = _load_scores(args)
    feat = _features(df)
    fprs = _parse_fprs(args.fprs)
    final = feat["final"]
    pf = feat["pf"]

    rows = []
    candidates = [
        ("frozen_alpha060", None, None, None, None),
        ("universal_pf_over_final_ge_t0p7_w0p015", "convex", "pf_over_final_ge", 0.70, 0.015),
    ]
    if args.best_mode and args.best_gate:
        candidates.append(
            (
                f"dataset_best_{args.best_gate}_t{args.best_threshold:g}_w{args.best_weight:g}",
                args.best_mode,
                args.best_gate,
                args.best_threshold,
                args.best_weight,
            )
        )

    scored = df[[*KEY_COLUMNS, "frozen_final_score", "patchfield_rank"]].copy()
    for name, mode, gate_name, threshold, weight in candidates:
        if mode is None:
            col = "frozen_final_score"
            gate_mean = 0.0
        else:
            gate = _gate(feat, str(gate_name), float(threshold))
            col = name
            scored[col] = _correct(final, pf, gate, float(weight), str(mode))
            gate_mean = float(gate.mean())
        row = {
            "dataset": args.dataset,
            "candidate": name,
            "mode": mode or "none",
            "gate": gate_name or "none",
            "threshold": threshold if threshold is not None else np.nan,
            "weight": weight if weight is not None else np.nan,
            "n_rows": len(df),
            "n_sources": int(df[df["subset"].str.lower() != "real"]["source_model"].nunique()),
            "gate_mean": gate_mean,
        }
        row.update(_fixed_fpr_metrics(scored, col, fprs))
        rows.append(row)

    out = pd.DataFrame(rows)
    for fpr in fprs:
        tag = str(fpr).replace(".", "p")
        base_mean = float(out.loc[out["candidate"] == "frozen_alpha060", f"fake_recall_mean_fpr{tag}"].iloc[0])
        base_min = float(out.loc[out["candidate"] == "frozen_alpha060", f"fake_recall_min_fpr{tag}"].iloc[0])
        out[f"delta_fake_recall_mean_fpr{tag}"] = out[f"fake_recall_mean_fpr{tag}"] - base_mean
        out[f"delta_fake_recall_min_fpr{tag}"] = out[f"fake_recall_min_fpr{tag}"] - base_min
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--frozen-final-scores-csv", type=Path, required=True)
    parser.add_argument("--patchfield-scores-csv", type=Path, required=True)
    parser.add_argument("--patchfield-score-col", required=True)
    parser.add_argument("--best-mode", default="")
    parser.add_argument("--best-gate", default="")
    parser.add_argument("--best-threshold", type=float, default=0.0)
    parser.add_argument("--best-weight", type=float, default=0.0)
    parser.add_argument("--fprs", default="0.001,0.005,0.01")
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    out = run(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved -> {args.output_csv}")


if __name__ == "__main__":
    main()
