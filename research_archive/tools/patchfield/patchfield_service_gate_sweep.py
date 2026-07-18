#!/usr/bin/env python3
"""Search PatchField final-score corrections with fixed-FPR service metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from patchfield_reliability_correction_sweep import KEY_COLUMNS, _read, _real_rank, _source_metrics
from patchfield_stability_gate_sweep import FROZEN_EXTRA_COLUMNS, _correct, _features, _gate


def _parse_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _format_float(value: float) -> str:
    return f"{value:.6g}".replace(".", "p").replace("-", "m")


def _fixed_fpr_per_source(df: pd.DataFrame, score_col: str, fprs: list[float], prefix: str) -> pd.DataFrame:
    real = df[df["subset"].str.lower() == "real"]
    fake = df[df["subset"].str.lower() != "real"]
    real_scores = real[score_col].to_numpy(float)
    rows = []
    for source, group in fake.groupby("source_model", sort=True):
        row = {"source_model": source, "n_fake": len(group)}
        fake_scores = group[score_col].to_numpy(float)
        for fpr in fprs:
            threshold = float(np.quantile(real_scores, fpr, method="higher"))
            tag = _format_float(fpr)
            row[f"{prefix}_fake_recall_fpr{tag}"] = float(np.mean(fake_scores <= threshold))
            row[f"{prefix}_threshold_fpr{tag}"] = threshold
        rows.append(row)
    return pd.DataFrame(rows)


def _service_objective(audit: pd.DataFrame, fprs: list[float]) -> float:
    value = 0.0
    for fpr in fprs:
        tag = _format_float(fpr)
        mean_delta = float(audit[f"delta_fake_recall_fpr{tag}"].mean())
        min_delta = float(audit[f"delta_fake_recall_fpr{tag}"].min())
        regressions = int((audit[f"delta_fake_recall_fpr{tag}"] < 0.0).sum())
        weight = 1.0 / max(fpr, 1e-12)
        value += weight * (mean_delta + 0.5 * min(0.0, min_delta) - 0.01 * regressions)
    return value / sum(1.0 / max(fpr, 1e-12) for fpr in fprs)


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
    fprs = _parse_floats(args.fprs)

    frozen_auc_ap = _source_metrics(df, "frozen_final_score", "frozen")
    frozen_service = _fixed_fpr_per_source(df, "frozen_final_score", fprs, "frozen")
    frozen_metrics = frozen_auc_ap.merge(frozen_service, on=["source_model", "n_fake"], validate="one_to_one")

    rows = []
    per_source_rows = []
    for mode in args.modes.split(","):
        for gate_name in args.gates.split(","):
            for threshold in args.thresholds:
                if gate_name in {"all", "sample_gate", "soft_sample_gate"} and threshold != args.thresholds[0]:
                    continue
                gate = _gate(feat, gate_name, threshold)
                if gate.mean() <= 0.0:
                    continue
                for weight in args.weights:
                    col = f"pfsvc_{mode}_{gate_name}_t{_format_float(threshold)}_w{_format_float(weight)}"
                    scored = df.copy()
                    scored[col] = _correct(final, pf, gate, weight, mode)
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
                    audit["mode"] = mode
                    audit["gate"] = gate_name
                    audit["threshold"] = threshold
                    audit["weight"] = weight
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
                        "mode": mode,
                        "gate": gate_name,
                        "threshold": threshold,
                        "weight": weight,
                        "n_rows": len(df),
                        "n_sources": len(audit),
                        "gate_mean": float(gate.mean()),
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

    summary = pd.DataFrame(rows)
    sort_cols = ["auc_ap_regressed_sources", "service_objective", "avg_delta_auc", "avg_delta_ap"]
    summary = summary.sort_values(sort_cols, ascending=[True, False, False, False])
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
    parser.add_argument("--weights", type=_parse_floats, default=_parse_floats("0.005,0.01,0.015,0.02,0.03,0.04,0.05,0.06,0.08,0.10"))
    parser.add_argument("--thresholds", type=_parse_floats, default=_parse_floats("-0.10,-0.05,0.0,0.03,0.05,0.08,0.10,0.12,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80,0.90"))
    parser.add_argument("--modes", default="convex,suppress_overfinal,toward_min,directional")
    parser.add_argument(
        "--gates",
        default=(
            "all,sample_gate,soft_sample_gate,"
            "model_spread_ge,final_patch_gap_ge,alpha_patch_gap_ge,final_over_pf_ge,"
            "pf_over_final_ge,persistence_le,global_le,"
            "split_minus_universal_le,persistence_minus_base_le,over_pf_and_spread_ge,"
            "gap_and_spread_ge,over_pf_and_persistence_le,over_pf_and_global_le"
        ),
    )
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
