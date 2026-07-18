#!/usr/bin/env python3
"""PatchField correction applied directly to the frozen alpha=0.60 final score.

This tests a simpler framework: keep the current detector intact, then use
PatchField only as a counterfactual reliability correction on its final score.
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


def _gate(df: pd.DataFrame, name: str, threshold: float, policy: str) -> np.ndarray:
    is_real = df["subset"].str.lower() == "real"
    final = df["frozen_final_score"].to_numpy(float)
    pf = df["patchfield_rank"].to_numpy(float)
    raw = df["raw_rank"].to_numpy(float)
    global_rank = df["global_rank"].to_numpy(float)
    persistence = df["persistence_rank"].to_numpy(float)
    sample_gate = df["sample_gate"].to_numpy(float)

    if name == "all":
        gate = np.ones(len(df), dtype=float)
    elif name == "sample_gate":
        gate = sample_gate
    elif name == "final_pf_gap_ge":
        gate = (np.abs(final - pf) >= threshold).astype(float)
    elif name == "final_over_pf_ge":
        gate = ((final - pf) >= threshold).astype(float)
    elif name == "raw_over_pf_ge":
        gate = ((raw - pf) >= threshold).astype(float)
    elif name == "global_le":
        gate = (global_rank <= threshold).astype(float)
    elif name == "persistence_le":
        gate = (persistence <= threshold).astype(float)
    elif name == "final_le":
        gate = (final <= threshold).astype(float)
    elif name == "final_over_pf_and_persistence_le":
        gate = (((final - pf) >= 0.0) & (persistence <= threshold)).astype(float)
    elif name == "raw_over_pf_and_persistence_le":
        gate = (((raw - pf) >= 0.0) & (persistence <= threshold)).astype(float)
    elif name == "final_over_pf_and_global_le":
        gate = (((final - pf) >= 0.0) & (global_rank <= threshold)).astype(float)
    elif name == "raw_over_pf_and_global_le":
        gate = (((raw - pf) >= 0.0) & (global_rank <= threshold)).astype(float)
    else:
        raise ValueError(f"Unknown gate: {name}")

    if policy == "same":
        return gate
    if policy == "real_off":
        return np.where(is_real, 0.0, gate)
    if policy == "real_sample":
        return np.where(is_real, sample_gate, gate)
    raise ValueError(f"Unknown policy: {policy}")


def _correct(final: np.ndarray, pf: np.ndarray, gate: np.ndarray, weight: float, mode: str) -> np.ndarray:
    strength = np.clip(weight * gate, 0.0, 1.0)
    if mode == "suppress_overfinal":
        out = final - strength * np.maximum(final - pf, 0.0)
    elif mode == "toward_min":
        out = (1.0 - strength) * final + strength * np.minimum(final, pf)
    elif mode == "convex":
        out = (1.0 - strength) * final + strength * pf
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return np.clip(out, 0.0, 1.0)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    frozen = _read(
        args.frozen_final_scores_csv,
        ["global_rank", "raw_rank", "persistence_rank", "final_score", "sample_gate"],
    ).rename(columns={"final_score": "frozen_final_score"})
    patch = _read(args.patchfield_scores_csv, [args.patchfield_score_col])
    df = frozen.merge(patch, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(df) == 0:
        raise ValueError("No overlap between frozen scores and PatchField scores")

    real_mask = df["subset"].str.lower() == "real"
    df["patchfield_rank"] = _real_rank(df[args.patchfield_score_col], real_mask)
    frozen_metrics = _source_metrics(df, "frozen_final_score", "frozen")

    rows = []
    per_source_rows = []
    scores = df[[*KEY_COLUMNS, "global_rank", "raw_rank", "persistence_rank", "patchfield_rank", "frozen_final_score"]].copy()
    final = df["frozen_final_score"].to_numpy(float)
    pf = df["patchfield_rank"].to_numpy(float)
    for mode in args.modes:
        for gate_name in args.gates:
            for threshold in args.thresholds:
                for policy in args.policies:
                    gate = _gate(df, gate_name, threshold, policy)
                    if gate_name == "all" and threshold != args.thresholds[0]:
                        continue
                    if gate.mean() == 0.0:
                        continue
                    for weight in args.weights:
                        col = (
                            f"pffs_{mode}_{gate_name}_t{_format_float(threshold)}_"
                            f"{policy}_w{_format_float(weight)}"
                        )
                        scored = df.copy()
                        scored[col] = _correct(final, pf, gate, weight, mode)
                        metrics = _source_metrics(scored, col, "candidate")
                        audit = frozen_metrics.merge(metrics, on=["source_model", "n_fake"], validate="one_to_one")
                        audit["dataset"] = args.dataset
                        audit["candidate_col"] = col
                        audit["mode"] = mode
                        audit["gate"] = gate_name
                        audit["threshold"] = threshold
                        audit["policy"] = policy
                        audit["weight"] = weight
                        audit["delta_auc"] = audit["candidate_auc"] - audit["frozen_auc"]
                        audit["delta_ap"] = audit["candidate_ap"] - audit["frozen_ap"]
                        per_source_rows.append(audit)
                        rows.append(
                            {
                                "dataset": args.dataset,
                                "candidate_col": col,
                                "mode": mode,
                                "gate": gate_name,
                                "threshold": threshold,
                                "policy": policy,
                                "weight": weight,
                                "n_rows": len(scored),
                                "n_sources": len(audit),
                                "gate_mean": float(gate.mean()),
                                "gate_real_mean": float(gate[real_mask.to_numpy()].mean()),
                                "gate_fake_mean": float(gate[~real_mask.to_numpy()].mean()),
                                "frozen_avg_auc": float(frozen_metrics["frozen_auc"].mean()),
                                "frozen_avg_ap": float(frozen_metrics["frozen_ap"].mean()),
                                "candidate_avg_auc": float(audit["candidate_auc"].mean()),
                                "candidate_avg_ap": float(audit["candidate_ap"].mean()),
                                "avg_delta_auc": float(audit["delta_auc"].mean()),
                                "avg_delta_ap": float(audit["delta_ap"].mean()),
                                "min_delta_auc": float(audit["delta_auc"].min()),
                                "min_delta_ap": float(audit["delta_ap"].min()),
                                "improved_auc_sources": int((audit["delta_auc"] > 0).sum()),
                                "improved_ap_sources": int((audit["delta_ap"] > 0).sum()),
                            }
                        )
                        if len(scores.columns) < args.max_score_columns + len(KEY_COLUMNS):
                            scores[col] = scored[col]

    summary = pd.DataFrame(rows).sort_values(["candidate_avg_auc", "candidate_avg_ap"], ascending=False)
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
    parser.add_argument("--weights", type=_parse_floats, default=_parse_floats("0.02,0.04,0.06,0.08,0.10,0.15,0.20,0.25,0.30"))
    parser.add_argument("--thresholds", type=_parse_floats, default=_parse_floats("0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80"))
    parser.add_argument("--modes", type=_parse_strs, default=_parse_strs("suppress_overfinal,toward_min,convex"))
    parser.add_argument("--gates", type=_parse_strs, default=_parse_strs("all,sample_gate,final_pf_gap_ge,final_over_pf_ge,raw_over_pf_ge,global_le,persistence_le,final_le,final_over_pf_and_persistence_le,raw_over_pf_and_persistence_le,final_over_pf_and_global_le,raw_over_pf_and_global_le"))
    parser.add_argument("--policies", type=_parse_strs, default=_parse_strs("same,real_sample,real_off"))
    parser.add_argument("--max-score-columns", type=int, default=40)
    args = parser.parse_args()

    summary, per_source, scores = run(args)
    args.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary_csv, index=False)
    per_source.to_csv(args.output_per_source_csv, index=False)
    scores.to_csv(args.output_scores_csv, index=False)
    print(summary.head(40).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved summary -> {args.output_summary_csv}")
    print(f"Saved per-source -> {args.output_per_source_csv}")
    print(f"Saved scores -> {args.output_scores_csv}")


if __name__ == "__main__":
    main()
