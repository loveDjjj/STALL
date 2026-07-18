#!/usr/bin/env python3
"""Sweep simple monotone PatchField final-score soft calibrators."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from patchfield_reliability_correction_sweep import KEY_COLUMNS, _read, _real_rank, _source_metrics
from patchfield_service_gate_sweep import _fixed_fpr_per_source, _format_float, _parse_floats, _service_objective
from patchfield_stability_gate_sweep import FROZEN_EXTRA_COLUMNS, _features


def _soft_strength(pf_over_final: np.ndarray, low: float, high: float, low_weight: float, high_weight: float) -> np.ndarray:
    if high <= low:
        raise ValueError("high threshold must be greater than low threshold")
    ramp = np.clip((pf_over_final - low) / (high - low), 0.0, 1.0)
    active = (pf_over_final >= low).astype(float)
    return active * (low_weight + ramp * (high_weight - low_weight))


def _correct(final: np.ndarray, pf: np.ndarray, strength: np.ndarray) -> np.ndarray:
    return np.clip((1.0 - strength) * final + strength * pf, 0.0, 1.0)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    frozen = _read(args.frozen_final_scores_csv, FROZEN_EXTRA_COLUMNS).rename(
        columns={"final_score": "frozen_final_score"}
    )
    patch = _read(args.patchfield_scores_csv, [args.patchfield_score_col])
    df = frozen.merge(patch, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(df) == 0:
        raise ValueError("No overlap between frozen scores and PatchField scores")

    real_mask = df["subset"].str.lower() == "real"
    df["patchfield_rank"] = _real_rank(df[args.patchfield_score_col], real_mask)
    feat = _features(df)
    final = feat["final"]
    pf = feat["pf"]
    pf_over_final = feat["pf_over_final"]
    fprs = _parse_floats(args.fprs)

    frozen_auc_ap = _source_metrics(df, "frozen_final_score", "frozen")
    frozen_service = _fixed_fpr_per_source(df, "frozen_final_score", fprs, "frozen")
    frozen_metrics = frozen_auc_ap.merge(frozen_service, on=["source_model", "n_fake"], validate="one_to_one")

    rows = []
    per_source_rows = []
    for low in args.low_thresholds:
        for high in args.high_thresholds:
            if high <= low:
                continue
            for low_weight in args.low_weights:
                for high_weight in args.high_weights:
                    if high_weight < low_weight:
                        continue
                    strength = _soft_strength(pf_over_final, low, high, low_weight, high_weight)
                    if strength.mean() <= 0.0:
                        continue
                    col = (
                        f"pfsoft_low{_format_float(low)}_high{_format_float(high)}"
                        f"_wl{_format_float(low_weight)}_wh{_format_float(high_weight)}"
                    )
                    scored = df.copy()
                    scored[col] = _correct(final, pf, strength)
                    candidate_auc_ap = _source_metrics(scored, col, "candidate")
                    candidate_service = _fixed_fpr_per_source(scored, col, fprs, "candidate")
                    candidate_metrics = candidate_auc_ap.merge(
                        candidate_service, on=["source_model", "n_fake"], validate="one_to_one"
                    )
                    audit = frozen_metrics.merge(
                        candidate_metrics, on=["source_model", "n_fake"], validate="one_to_one"
                    )
                    audit["dataset"] = args.dataset
                    audit["candidate_col"] = col
                    audit["low_threshold"] = low
                    audit["high_threshold"] = high
                    audit["low_weight"] = low_weight
                    audit["high_weight"] = high_weight
                    audit["delta_auc"] = audit["candidate_auc"] - audit["frozen_auc"]
                    audit["delta_ap"] = audit["candidate_ap"] - audit["frozen_ap"]
                    for fpr in fprs:
                        tag = _format_float(fpr)
                        audit[f"delta_fake_recall_fpr{tag}"] = (
                            audit[f"candidate_fake_recall_fpr{tag}"] - audit[f"frozen_fake_recall_fpr{tag}"]
                        )
                    per_source_rows.append(audit)

                    auc_ap_regressions = int(((audit["delta_auc"] < 0.0) | (audit["delta_ap"] < 0.0)).sum())
                    row = {
                        "dataset": args.dataset,
                        "candidate_col": col,
                        "low_threshold": low,
                        "high_threshold": high,
                        "low_weight": low_weight,
                        "high_weight": high_weight,
                        "n_rows": len(df),
                        "n_sources": len(audit),
                        "active_mean": float((strength > 0.0).mean()),
                        "strength_mean": float(strength.mean()),
                        "avg_delta_auc": float(audit["delta_auc"].mean()),
                        "avg_delta_ap": float(audit["delta_ap"].mean()),
                        "min_delta_auc": float(audit["delta_auc"].min()),
                        "min_delta_ap": float(audit["delta_ap"].min()),
                        "auc_ap_regressed_sources": auc_ap_regressions,
                        "service_objective": _service_objective(audit, fprs),
                    }
                    for fpr in fprs:
                        tag = _format_float(fpr)
                        deltas = audit[f"delta_fake_recall_fpr{tag}"]
                        row[f"avg_delta_fake_recall_fpr{tag}"] = float(deltas.mean())
                        row[f"min_delta_fake_recall_fpr{tag}"] = float(deltas.min())
                        row[f"service_regressed_sources_fpr{tag}"] = int((deltas < 0.0).sum())
                    rows.append(row)

    summary = pd.DataFrame(rows).sort_values(
        ["auc_ap_regressed_sources", "service_objective", "avg_delta_auc", "avg_delta_ap"],
        ascending=[True, False, False, False],
    )
    per_source = pd.concat(per_source_rows, ignore_index=True) if per_source_rows else pd.DataFrame()
    return summary, per_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--frozen-final-scores-csv", type=Path, required=True)
    parser.add_argument("--patchfield-scores-csv", type=Path, required=True)
    parser.add_argument("--patchfield-score-col", required=True)
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    parser.add_argument("--output-per-source-csv", type=Path, required=True)
    parser.add_argument("--fprs", default="0.001,0.005,0.01")
    parser.add_argument("--low-thresholds", type=_parse_floats, default=_parse_floats("0.40,0.50,0.60"))
    parser.add_argument("--high-thresholds", type=_parse_floats, default=_parse_floats("0.65,0.70,0.75,0.80"))
    parser.add_argument("--low-weights", type=_parse_floats, default=_parse_floats("0.0025,0.005,0.0075,0.01"))
    parser.add_argument("--high-weights", type=_parse_floats, default=_parse_floats("0.01,0.015,0.02,0.03"))
    args = parser.parse_args()

    summary, per_source = run(args)
    args.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary_csv, index=False)
    per_source.to_csv(args.output_per_source_csv, index=False)
    print(summary.head(40).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved summary -> {args.output_summary_csv}")
    print(f"Saved per-source -> {args.output_per_source_csv}")


if __name__ == "__main__":
    main()
