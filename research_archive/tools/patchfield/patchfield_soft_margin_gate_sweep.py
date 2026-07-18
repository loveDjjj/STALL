#!/usr/bin/env python3
"""Soft margin PatchField correction on frozen alpha=0.60 final scores.

This is a continuous version of the clean PatchField gate.  Instead of adding
another hard branch, PatchField is used as a small reliability calibrator whose
strength grows with the margin between PatchField rank and the frozen final
score.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from patchfield_reliability_correction_sweep import (
    KEY_COLUMNS,
    _format_float,
    _parse_floats,
    _parse_strs,
    _read,
    _real_rank,
    _source_metrics,
)


FROZEN_EXTRA_COLUMNS = [
    "base_score",
    "universal_score",
    "split_score",
    "global_rank",
    "raw_rank",
    "persistence_rank",
    "persistence_minus_base",
    "split_minus_universal",
    "sample_gate",
    "final_score",
]


def _soft_gate(final: np.ndarray, pf: np.ndarray, shape: str, start: float, width: float) -> np.ndarray:
    pf_over_final = pf - final
    final_over_pf = final - pf
    gap = np.abs(pf - final)
    eps = 1e-12

    if shape == "pf_over_ramp":
        margin = pf_over_final
    elif shape == "final_over_ramp":
        margin = final_over_pf
    elif shape == "gap_ramp":
        margin = gap
    elif shape == "pf_over_squared":
        margin = pf_over_final
    elif shape == "gap_squared":
        margin = gap
    else:
        raise ValueError(f"Unknown soft gate shape: {shape}")

    gate = np.clip((margin - start) / max(width, eps), 0.0, 1.0)
    if shape.endswith("_squared"):
        gate = gate * gate
    return gate


def _target(final: np.ndarray, pf: np.ndarray, mode: str) -> np.ndarray:
    if mode == "convex":
        return pf
    if mode == "raise_only":
        return np.maximum(final, pf)
    if mode == "lower_only":
        return np.minimum(final, pf)
    if mode == "toward_mid":
        return 0.5 * final + 0.5 * pf
    raise ValueError(f"Unknown mode: {mode}")


def _correct(final: np.ndarray, pf: np.ndarray, gate: np.ndarray, weight: float, mode: str) -> np.ndarray:
    strength = np.clip(weight * gate, 0.0, 1.0)
    target = _target(final, pf, mode)
    out = (1.0 - strength) * final + strength * target
    return np.clip(out, 0.0, 1.0)


def _objective(audit: pd.DataFrame) -> float:
    avg_auc = float(audit["delta_auc"].mean())
    avg_ap = float(audit["delta_ap"].mean())
    min_auc = float(audit["delta_auc"].min())
    min_ap = float(audit["delta_ap"].min())
    regressions = int(((audit["delta_auc"] < 0.0) | (audit["delta_ap"] < 0.0)).sum())
    return avg_auc + 0.5 * avg_ap + min(0.0, min_auc) + 0.5 * min(0.0, min_ap) - 0.002 * regressions


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frozen = _read(args.frozen_final_scores_csv, FROZEN_EXTRA_COLUMNS).rename(
        columns={"final_score": "frozen_final_score"}
    )
    patch = _read(args.patchfield_scores_csv, [args.patchfield_score_col])
    df = frozen.merge(patch, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(df) == 0:
        raise ValueError("No overlap between frozen scores and PatchField scores")

    real_mask = df["subset"].str.lower() == "real"
    df["patchfield_rank"] = _real_rank(df[args.patchfield_score_col], real_mask)
    frozen_metrics = _source_metrics(df, "frozen_final_score", "frozen")

    final = df["frozen_final_score"].to_numpy(float)
    pf = df["patchfield_rank"].to_numpy(float)
    real_np = real_mask.to_numpy()

    rows = []
    per_source_rows = []
    scores = df[
        [
            *KEY_COLUMNS,
            "frozen_final_score",
            "patchfield_rank",
            "global_rank",
            "raw_rank",
            "persistence_rank",
            "sample_gate",
        ]
    ].copy()

    for mode in args.modes:
        for shape in args.shapes:
            for start in args.starts:
                for width in args.widths:
                    gate = _soft_gate(final, pf, shape, start, width)
                    if gate.mean() <= 0.0:
                        continue
                    for weight in args.weights:
                        col = (
                            f"pfsm_{mode}_{shape}_s{_format_float(start)}_"
                            f"wd{_format_float(width)}_w{_format_float(weight)}"
                        )
                        scored = df.copy()
                        scored[col] = _correct(final, pf, gate, weight, mode)
                        metrics = _source_metrics(scored, col, "candidate")
                        audit = frozen_metrics.merge(metrics, on=["source_model", "n_fake"], validate="one_to_one")
                        audit["dataset"] = args.dataset
                        audit["candidate_col"] = col
                        audit["mode"] = mode
                        audit["shape"] = shape
                        audit["start"] = start
                        audit["width"] = width
                        audit["weight"] = weight
                        audit["delta_auc"] = audit["candidate_auc"] - audit["frozen_auc"]
                        audit["delta_ap"] = audit["candidate_ap"] - audit["frozen_ap"]
                        per_source_rows.append(audit)

                        regressions = int(((audit["delta_auc"] < 0.0) | (audit["delta_ap"] < 0.0)).sum())
                        row = {
                            "dataset": args.dataset,
                            "candidate_col": col,
                            "mode": mode,
                            "shape": shape,
                            "start": start,
                            "width": width,
                            "weight": weight,
                            "n_rows": len(df),
                            "n_sources": len(audit),
                            "gate_mean": float(gate.mean()),
                            "gate_real_mean": float(gate[real_np].mean()),
                            "gate_fake_mean": float(gate[~real_np].mean()),
                            "gate_active_mean": float((gate > 0.0).mean()),
                            "gate_active_real_mean": float((gate[real_np] > 0.0).mean()),
                            "gate_active_fake_mean": float((gate[~real_np] > 0.0).mean()),
                            "frozen_avg_auc": float(frozen_metrics["frozen_auc"].mean()),
                            "frozen_avg_ap": float(frozen_metrics["frozen_ap"].mean()),
                            "candidate_avg_auc": float(audit["candidate_auc"].mean()),
                            "candidate_avg_ap": float(audit["candidate_ap"].mean()),
                            "avg_delta_auc": float(audit["delta_auc"].mean()),
                            "avg_delta_ap": float(audit["delta_ap"].mean()),
                            "min_delta_auc": float(audit["delta_auc"].min()),
                            "min_delta_ap": float(audit["delta_ap"].min()),
                            "regressed_sources": regressions,
                            "improved_auc_sources": int((audit["delta_auc"] > 0.0).sum()),
                            "improved_ap_sources": int((audit["delta_ap"] > 0.0).sum()),
                            "stability_objective": _objective(audit),
                        }
                        rows.append(row)
                        if len(scores.columns) < args.max_score_columns + len(KEY_COLUMNS):
                            scores[col] = scored[col]

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        ["regressed_sources", "stability_objective", "candidate_avg_auc", "candidate_avg_ap"],
        ascending=[True, False, False, False],
    )
    per_source = pd.concat(per_source_rows, ignore_index=True) if per_source_rows else pd.DataFrame()
    return summary, per_source, scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--frozen-final-scores-csv", type=Path, required=True)
    parser.add_argument("--patchfield-scores-csv", type=Path, required=True)
    parser.add_argument("--patchfield-score-col", required=True)
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    parser.add_argument("--output-per-source-csv", type=Path, required=True)
    parser.add_argument("--output-scores-csv", type=Path, required=True)
    parser.add_argument("--weights", type=_parse_floats, default=_parse_floats("0.005,0.01,0.015,0.02,0.03,0.04,0.05,0.06,0.08,0.10"))
    parser.add_argument("--starts", type=_parse_floats, default=_parse_floats("0.30,0.40,0.50,0.60,0.70,0.80"))
    parser.add_argument("--widths", type=_parse_floats, default=_parse_floats("0.05,0.10,0.15,0.20,0.30,0.40"))
    parser.add_argument("--modes", type=_parse_strs, default=_parse_strs("convex,raise_only,lower_only,toward_mid"))
    parser.add_argument("--shapes", type=_parse_strs, default=_parse_strs("pf_over_ramp,pf_over_squared,gap_ramp,gap_squared,final_over_ramp"))
    parser.add_argument("--max-score-columns", type=int, default=60)
    args = parser.parse_args()

    summary, per_source, scores = run(args)
    args.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary_csv, index=False)
    per_source.to_csv(args.output_per_source_csv, index=False)
    scores.to_csv(args.output_scores_csv, index=False)
    print(summary.head(50).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved summary -> {args.output_summary_csv}")
    print(f"Saved per-source -> {args.output_per_source_csv}")
    print(f"Saved scores -> {args.output_scores_csv}")


if __name__ == "__main__":
    main()
