#!/usr/bin/env python3
"""Bidirectional PatchField correction on the frozen final score.

The previous final-score sweep used one gate and one convex weight.  This sweep
separates the two useful actions:

  boost    final score toward PatchField only when PatchField is more real-like
  suppress final score toward PatchField only when PatchField is less real-like

Both gates are deployable and label-free.  The intent is to keep the simple
post-hoc correction framework while reducing source regressions from applying a
single symmetric rule everywhere.
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


def _gate(df: pd.DataFrame, name: str, threshold: float) -> np.ndarray:
    final = df["frozen_final_score"].to_numpy(float)
    pf = df["patchfield_rank"].to_numpy(float)
    raw = df["raw_rank"].to_numpy(float)
    global_rank = df["global_rank"].to_numpy(float)
    persistence = df["persistence_rank"].to_numpy(float)
    sample_gate = df["sample_gate"].to_numpy(float)

    if name == "all":
        return np.ones(len(df), dtype=float)
    if name == "none":
        return np.zeros(len(df), dtype=float)
    if name == "sample_gate":
        return sample_gate
    if name == "final_ge":
        return (final >= threshold).astype(float)
    if name == "global_ge":
        return (global_rank >= threshold).astype(float)
    if name == "raw_ge":
        return (raw >= threshold).astype(float)
    if name == "persistence_ge":
        return (persistence >= threshold).astype(float)
    if name == "final_le":
        return (final <= threshold).astype(float)
    if name == "global_le":
        return (global_rank <= threshold).astype(float)
    if name == "raw_le":
        return (raw <= threshold).astype(float)
    if name == "persistence_le":
        return (persistence <= threshold).astype(float)
    if name == "final_pf_gap_ge":
        return (np.abs(final - pf) >= threshold).astype(float)
    if name == "final_over_pf_ge":
        return ((final - pf) >= threshold).astype(float)
    if name == "pf_over_final_ge":
        return ((pf - final) >= threshold).astype(float)
    if name == "sample_or_final_ge":
        return np.maximum(sample_gate, (final >= threshold).astype(float))
    if name == "sample_or_global_ge":
        return np.maximum(sample_gate, (global_rank >= threshold).astype(float))
    if name == "sample_or_persistence_ge":
        return np.maximum(sample_gate, (persistence >= threshold).astype(float))
    if name == "final_ge_and_pf_over_final":
        return ((final >= threshold) & (pf > final)).astype(float)
    if name == "global_ge_and_pf_over_final":
        return ((global_rank >= threshold) & (pf > final)).astype(float)
    if name == "persistence_ge_and_pf_over_final":
        return ((persistence >= threshold) & (pf > final)).astype(float)
    if name == "final_le_and_final_over_pf":
        return ((final <= threshold) & (final > pf)).astype(float)
    if name == "global_le_and_final_over_pf":
        return ((global_rank <= threshold) & (final > pf)).astype(float)
    if name == "persistence_le_and_final_over_pf":
        return ((persistence <= threshold) & (final > pf)).astype(float)
    raise ValueError(f"Unknown gate: {name}")


def _correct(
    final: np.ndarray,
    pf: np.ndarray,
    boost_gate: np.ndarray,
    suppress_gate: np.ndarray,
    boost_weight: float,
    suppress_weight: float,
) -> np.ndarray:
    boost = np.clip(boost_weight * boost_gate, 0.0, 1.0) * np.maximum(pf - final, 0.0)
    suppress = np.clip(suppress_weight * suppress_gate, 0.0, 1.0) * np.maximum(final - pf, 0.0)
    return np.clip(final + boost - suppress, 0.0, 1.0)


def _read_inputs(args: argparse.Namespace) -> pd.DataFrame:
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
    return df


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = _read_inputs(args)
    real_mask = df["subset"].str.lower() == "real"
    frozen_metrics = _source_metrics(df, "frozen_final_score", "frozen")

    rows: list[dict[str, object]] = []
    per_source_rows: list[pd.DataFrame] = []
    scores = df[[*KEY_COLUMNS, "global_rank", "raw_rank", "persistence_rank", "patchfield_rank", "frozen_final_score"]].copy()

    final = df["frozen_final_score"].to_numpy(float)
    pf = df["patchfield_rank"].to_numpy(float)
    for boost_gate_name in args.boost_gates:
        for boost_threshold in args.boost_thresholds:
            boost_gate = _gate(df, boost_gate_name, boost_threshold)
            if boost_gate_name in {"all", "none", "sample_gate"} and boost_threshold != args.boost_thresholds[0]:
                continue
            for suppress_gate_name in args.suppress_gates:
                for suppress_threshold in args.suppress_thresholds:
                    suppress_gate = _gate(df, suppress_gate_name, suppress_threshold)
                    if suppress_gate_name in {"all", "none", "sample_gate"} and suppress_threshold != args.suppress_thresholds[0]:
                        continue
                    if boost_gate.mean() == 0.0 and suppress_gate.mean() == 0.0:
                        continue
                    for boost_weight in args.boost_weights:
                        for suppress_weight in args.suppress_weights:
                            if boost_weight == 0.0 and suppress_weight == 0.0:
                                continue
                            col = (
                                f"pfbd_b{boost_gate_name}_t{_format_float(boost_threshold)}_w{_format_float(boost_weight)}"
                                f"_s{suppress_gate_name}_t{_format_float(suppress_threshold)}_w{_format_float(suppress_weight)}"
                            )
                            scored = df.copy()
                            scored[col] = _correct(final, pf, boost_gate, suppress_gate, boost_weight, suppress_weight)
                            metrics = _source_metrics(scored, col, "candidate")
                            audit = frozen_metrics.merge(metrics, on=["source_model", "n_fake"], validate="one_to_one")
                            audit["dataset"] = args.dataset
                            audit["candidate_col"] = col
                            audit["boost_gate"] = boost_gate_name
                            audit["boost_threshold"] = boost_threshold
                            audit["boost_weight"] = boost_weight
                            audit["suppress_gate"] = suppress_gate_name
                            audit["suppress_threshold"] = suppress_threshold
                            audit["suppress_weight"] = suppress_weight
                            audit["delta_auc"] = audit["candidate_auc"] - audit["frozen_auc"]
                            audit["delta_ap"] = audit["candidate_ap"] - audit["frozen_ap"]
                            per_source_rows.append(audit)
                            rows.append(
                                {
                                    "dataset": args.dataset,
                                    "candidate_col": col,
                                    "boost_gate": boost_gate_name,
                                    "boost_threshold": boost_threshold,
                                    "boost_weight": boost_weight,
                                    "suppress_gate": suppress_gate_name,
                                    "suppress_threshold": suppress_threshold,
                                    "suppress_weight": suppress_weight,
                                    "n_rows": len(scored),
                                    "n_sources": len(audit),
                                    "boost_gate_mean": float(boost_gate.mean()),
                                    "boost_gate_real_mean": float(boost_gate[real_mask.to_numpy()].mean()),
                                    "boost_gate_fake_mean": float(boost_gate[~real_mask.to_numpy()].mean()),
                                    "suppress_gate_mean": float(suppress_gate.mean()),
                                    "suppress_gate_real_mean": float(suppress_gate[real_mask.to_numpy()].mean()),
                                    "suppress_gate_fake_mean": float(suppress_gate[~real_mask.to_numpy()].mean()),
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
    parser.add_argument("--boost-gates", type=_parse_strs, default=_parse_strs("all,final_ge,global_ge,persistence_ge,sample_or_final_ge,sample_or_global_ge,final_ge_and_pf_over_final,global_ge_and_pf_over_final"))
    parser.add_argument("--suppress-gates", type=_parse_strs, default=_parse_strs("none,all,final_over_pf_ge,final_pf_gap_ge,final_le,global_le,persistence_le,final_le_and_final_over_pf,global_le_and_final_over_pf"))
    parser.add_argument("--boost-thresholds", type=_parse_floats, default=_parse_floats("0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60"))
    parser.add_argument("--suppress-thresholds", type=_parse_floats, default=_parse_floats("0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60"))
    parser.add_argument("--boost-weights", type=_parse_floats, default=_parse_floats("0.02,0.04,0.06,0.08,0.10,0.15,0.20"))
    parser.add_argument("--suppress-weights", type=_parse_floats, default=_parse_floats("0.0,0.04,0.08,0.12,0.16,0.20,0.30"))
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
