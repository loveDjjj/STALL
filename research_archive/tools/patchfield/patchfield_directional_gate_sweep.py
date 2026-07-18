#!/usr/bin/env python3
"""Sweep deployable directional gates for PatchField reliability correction.

This focuses on the non-label gate: when should LSMI be allowed to suppress
raw patch overconfidence inside the alpha=0.60 fallback?
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from patchfield_reliability_correction_sweep import (
    KEY_COLUMNS,
    _corrected_rank,
    _format_float,
    _parse_floats,
    _parse_strs,
    _read,
    _real_rank,
    _score_fallback,
    _source_metrics,
)


def _gate(df: pd.DataFrame, name: str, raw_pf_min: float, aux_threshold: float) -> np.ndarray:
    raw = df["raw_rank"].to_numpy(float)
    pf = df["patchfield_rank"].to_numpy(float)
    global_rank = df["global_rank"].to_numpy(float)
    persistence = df["persistence_rank"].to_numpy(float)
    frozen = df["frozen_final_score"].to_numpy(float)
    overraw = raw - pf
    base = overraw >= raw_pf_min

    if name == "overraw":
        return base.astype(float)
    if name == "overraw_global_le":
        return (base & (global_rank <= aux_threshold)).astype(float)
    if name == "overraw_persistence_le":
        return (base & (persistence <= aux_threshold)).astype(float)
    if name == "overraw_frozen_le":
        return (base & (frozen <= aux_threshold)).astype(float)
    if name == "overraw_min_gp_le":
        return (base & (np.minimum(global_rank, persistence) <= aux_threshold)).astype(float)
    if name == "overraw_global_raw_gap_ge":
        return (base & ((raw - global_rank) >= aux_threshold)).astype(float)
    if name == "overraw_global_pf_gap_ge":
        return (base & ((global_rank - pf) >= aux_threshold)).astype(float)
    if name == "overraw_consensus_fake":
        return (base & (global_rank <= aux_threshold) & (persistence <= aux_threshold)).astype(float)
    raise ValueError(f"Unknown gate: {name}")


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    frozen = _read(args.frozen_final_scores_csv, ["global_rank", "raw_rank", "persistence_rank", "final_score"]).rename(
        columns={"final_score": "frozen_final_score"}
    )
    patch = _read(args.patchfield_scores_csv, [args.patchfield_score_col])
    df = frozen.merge(patch, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(df) == 0:
        raise ValueError("No overlap between frozen scores and PatchField scores")

    real_mask = df["subset"].str.lower() == "real"
    df["patchfield_rank"] = _real_rank(df[args.patchfield_score_col], real_mask)
    raw_scored = _score_fallback(df, "raw_rank", args, "raw")
    df = raw_scored
    raw_metrics = _source_metrics(df, "raw_final_score", "raw_system")
    frozen_metrics = _source_metrics(df, "frozen_final_score", "frozen_file")

    rows = []
    per_source_rows = []
    raw = df["raw_rank"].to_numpy(float)
    pf = df["patchfield_rank"].to_numpy(float)
    for gate_name in args.gates:
        for raw_pf_min in args.raw_pf_mins:
            for aux_threshold in args.aux_thresholds:
                gate = _gate(df, gate_name, raw_pf_min, aux_threshold)
                if gate.mean() == 0.0:
                    continue
                for mode in args.modes:
                    for weight in args.weights:
                        col = (
                            f"pfdg_{mode}_{gate_name}_rpf{_format_float(raw_pf_min)}_"
                            f"aux{_format_float(aux_threshold)}_w{_format_float(weight)}"
                        )
                        cand_input = df.copy()
                        cand_input[col] = _corrected_rank(raw, pf, gate, weight, mode)
                        scored = _score_fallback(cand_input, col, args, col)
                        metrics = _source_metrics(scored, f"{col}_final_score", "candidate")
                        audit = raw_metrics.merge(metrics, on=["source_model", "n_fake"], validate="one_to_one")
                        audit = audit.merge(frozen_metrics, on=["source_model", "n_fake"], validate="one_to_one")
                        audit["dataset"] = args.dataset
                        audit["candidate_col"] = col
                        audit["gate"] = gate_name
                        audit["raw_pf_min"] = raw_pf_min
                        audit["aux_threshold"] = aux_threshold
                        audit["mode"] = mode
                        audit["weight"] = weight
                        audit["delta_auc"] = audit["candidate_auc"] - audit["raw_system_auc"]
                        audit["delta_ap"] = audit["candidate_ap"] - audit["raw_system_ap"]
                        audit["delta_frozen_auc"] = audit["candidate_auc"] - audit["frozen_file_auc"]
                        audit["delta_frozen_ap"] = audit["candidate_ap"] - audit["frozen_file_ap"]
                        per_source_rows.append(audit)
                        rows.append(
                            {
                                "dataset": args.dataset,
                                "candidate_col": col,
                                "gate": gate_name,
                                "raw_pf_min": raw_pf_min,
                                "aux_threshold": aux_threshold,
                                "mode": mode,
                                "weight": weight,
                                "n_rows": len(scored),
                                "n_sources": len(audit),
                                "gate_mean": float(gate.mean()),
                                "gate_real_mean": float(gate[real_mask.to_numpy()].mean()),
                                "gate_fake_mean": float(gate[~real_mask.to_numpy()].mean()),
                                "raw_avg_auc": float(raw_metrics["raw_system_auc"].mean()),
                                "raw_avg_ap": float(raw_metrics["raw_system_ap"].mean()),
                                "frozen_avg_auc": float(frozen_metrics["frozen_file_auc"].mean()),
                                "frozen_avg_ap": float(frozen_metrics["frozen_file_ap"].mean()),
                                "candidate_avg_auc": float(audit["candidate_auc"].mean()),
                                "candidate_avg_ap": float(audit["candidate_ap"].mean()),
                                "avg_delta_auc": float(audit["delta_auc"].mean()),
                                "avg_delta_ap": float(audit["delta_ap"].mean()),
                                "avg_delta_frozen_auc": float(audit["delta_frozen_auc"].mean()),
                                "avg_delta_frozen_ap": float(audit["delta_frozen_ap"].mean()),
                                "min_delta_auc": float(audit["delta_auc"].min()),
                                "min_delta_ap": float(audit["delta_ap"].min()),
                                "min_delta_frozen_auc": float(audit["delta_frozen_auc"].min()),
                                "min_delta_frozen_ap": float(audit["delta_frozen_ap"].min()),
                                "improved_auc_sources": int((audit["delta_auc"] > 0).sum()),
                                "improved_ap_sources": int((audit["delta_ap"] > 0).sum()),
                                "improved_frozen_auc_sources": int((audit["delta_frozen_auc"] > 0).sum()),
                                "improved_frozen_ap_sources": int((audit["delta_frozen_ap"] > 0).sum()),
                            }
                        )

    summary = pd.DataFrame(rows).sort_values(["candidate_avg_auc", "candidate_avg_ap"], ascending=False)
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
    parser.add_argument("--gates", type=_parse_strs, default=_parse_strs("overraw,overraw_global_le,overraw_persistence_le,overraw_frozen_le,overraw_min_gp_le,overraw_global_raw_gap_ge,overraw_global_pf_gap_ge,overraw_consensus_fake"))
    parser.add_argument("--raw-pf-mins", type=_parse_floats, default=_parse_floats("0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.40"))
    parser.add_argument("--aux-thresholds", type=_parse_floats, default=_parse_floats("0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80"))
    parser.add_argument("--weights", type=_parse_floats, default=_parse_floats("0.02,0.04,0.06,0.08,0.10,0.15,0.20,0.25,0.30"))
    parser.add_argument("--modes", type=_parse_strs, default=_parse_strs("suppress_overraw,toward_min"))
    parser.add_argument("--alpha", type=float, default=0.60)
    parser.add_argument("--universal-weight", type=float, default=0.15)
    parser.add_argument("--real-weight", type=float, default=0.25)
    parser.add_argument("--fake-weight", type=float, default=0.25)
    parser.add_argument("--source-gate-threshold", type=float, default=0.45)
    parser.add_argument("--disagreement-threshold", type=float, default=0.20)
    parser.add_argument("--confidence-threshold", type=float, default=0.30)
    parser.add_argument("--selector-threshold", type=float, default=0.0)
    parser.add_argument("--real-policy", choices=["split", "universal", "rule"], default="split")
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
