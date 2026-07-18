#!/usr/bin/env python3
"""Stability-gated PatchField correction on frozen alpha=0.60 final scores.

This explores a cleaner deployable framework: the current detector remains the
main score, and PatchField only corrects samples where the detector's own
subscores disagree or show low stability.
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
    "source_gate_mean",
    "use_split",
    "final_score",
]


def _rank01(values: pd.Series, real_mask: pd.Series) -> np.ndarray:
    """Real-calibrated rank where larger means more fake-like."""
    return np.asarray(_real_rank(values, real_mask), dtype=float)


def _features(df: pd.DataFrame) -> dict[str, np.ndarray]:
    real_mask = df["subset"].str.lower() == "real"
    final = df["frozen_final_score"].to_numpy(float)
    pf = df["patchfield_rank"].to_numpy(float)
    global_rank = df["global_rank"].to_numpy(float)
    raw = df["raw_rank"].to_numpy(float)
    persistence = df["persistence_rank"].to_numpy(float)
    sample_gate = df["sample_gate"].fillna(0.0).to_numpy(float)
    source_gate_mean = df["source_gate_mean"].fillna(0.0).to_numpy(float)
    use_split = df["use_split"].fillna(0.0).to_numpy(float)

    base_rank = _rank01(df["base_score"], real_mask)
    universal_rank = _rank01(df["universal_score"], real_mask)
    split_rank = _rank01(df["split_score"], real_mask)

    model_spread = np.max(
        np.vstack([global_rank, raw, persistence, base_rank, universal_rank, split_rank]),
        axis=0,
    ) - np.min(
        np.vstack([global_rank, raw, persistence, base_rank, universal_rank, split_rank]),
        axis=0,
    )
    alpha_core_mean = (global_rank + raw + persistence) / 3.0
    alpha_patch_gap = np.abs(alpha_core_mean - pf)
    final_patch_gap = np.abs(final - pf)
    final_over_pf = final - pf
    pf_over_final = pf - final

    return {
        "final": final,
        "pf": pf,
        "global": global_rank,
        "raw": raw,
        "persistence": persistence,
        "base_rank": base_rank,
        "universal_rank": universal_rank,
        "split_rank": split_rank,
        "sample_gate": sample_gate,
        "source_gate_mean": source_gate_mean,
        "use_split": use_split,
        "persistence_minus_base": df["persistence_minus_base"].fillna(0.0).to_numpy(float),
        "split_minus_universal": df["split_minus_universal"].fillna(0.0).to_numpy(float),
        "model_spread": model_spread,
        "alpha_core_mean": alpha_core_mean,
        "alpha_patch_gap": alpha_patch_gap,
        "final_patch_gap": final_patch_gap,
        "final_over_pf": final_over_pf,
        "pf_over_final": pf_over_final,
        "global_raw_gap": np.abs(global_rank - raw),
        "global_persistence_gap": np.abs(global_rank - persistence),
        "raw_persistence_gap": np.abs(raw - persistence),
    }


def _gate(feat: dict[str, np.ndarray], name: str, threshold: float) -> np.ndarray:
    final_over_pf = feat["final_over_pf"]
    pf_over_final = feat["pf_over_final"]
    final_patch_gap = feat["final_patch_gap"]
    model_spread = feat["model_spread"]
    source_gate_mean = feat["source_gate_mean"]
    sample_gate = feat["sample_gate"]
    persistence = feat["persistence"]
    global_rank = feat["global"]
    use_split = feat["use_split"]
    split_minus_universal = feat["split_minus_universal"]
    persistence_minus_base = feat["persistence_minus_base"]
    alpha_patch_gap = feat["alpha_patch_gap"]

    if name == "all":
        return np.ones_like(final_over_pf)
    if name == "sample_gate":
        return sample_gate
    if name == "soft_sample_gate":
        return 0.25 + 0.75 * sample_gate
    # The source/use_split gates below are kept for diagnostics only. In the
    # frozen alpha=0.60 files, source_gate_mean is computed from fake sources
    # and filled with zero for real sources, while use_split may be forced by
    # real_policy. They must not be used as deployable default gates.
    if name == "source_gate_low":
        return (source_gate_mean <= threshold).astype(float)
    if name == "source_gate_high":
        return (source_gate_mean >= threshold).astype(float)
    if name == "model_spread_ge":
        return (model_spread >= threshold).astype(float)
    if name == "final_patch_gap_ge":
        return (final_patch_gap >= threshold).astype(float)
    if name == "alpha_patch_gap_ge":
        return (alpha_patch_gap >= threshold).astype(float)
    if name == "final_over_pf_ge":
        return (final_over_pf >= threshold).astype(float)
    if name == "pf_over_final_ge":
        return (pf_over_final >= threshold).astype(float)
    if name == "persistence_le":
        return (persistence <= threshold).astype(float)
    if name == "global_le":
        return (global_rank <= threshold).astype(float)
    if name == "use_split":
        return use_split
    if name == "no_split":
        return 1.0 - use_split
    if name == "split_minus_universal_le":
        return (split_minus_universal <= threshold).astype(float)
    if name == "persistence_minus_base_le":
        return (persistence_minus_base <= threshold).astype(float)
    if name == "over_pf_and_spread_ge":
        return ((final_over_pf >= 0.0) & (model_spread >= threshold)).astype(float)
    if name == "over_pf_and_source_low":
        return ((final_over_pf >= 0.0) & (source_gate_mean <= threshold)).astype(float)
    if name == "gap_and_source_low":
        return ((final_patch_gap >= 0.05) & (source_gate_mean <= threshold)).astype(float)
    if name == "gap_and_spread_ge":
        return ((final_patch_gap >= 0.05) & (model_spread >= threshold)).astype(float)
    if name == "over_pf_and_persistence_le":
        return ((final_over_pf >= 0.0) & (persistence <= threshold)).astype(float)
    if name == "over_pf_and_global_le":
        return ((final_over_pf >= 0.0) & (global_rank <= threshold)).astype(float)
    raise ValueError(f"Unknown gate: {name}")


def _correct(final: np.ndarray, pf: np.ndarray, gate: np.ndarray, weight: float, mode: str) -> np.ndarray:
    strength = np.clip(weight * gate, 0.0, 1.0)
    if mode == "convex":
        out = (1.0 - strength) * final + strength * pf
    elif mode == "suppress_overfinal":
        out = final - strength * np.maximum(final - pf, 0.0)
    elif mode == "toward_min":
        out = (1.0 - strength) * final + strength * np.minimum(final, pf)
    elif mode == "directional":
        target = np.where(pf < final, pf, final)
        out = (1.0 - strength) * final + strength * target
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return np.clip(out, 0.0, 1.0)


def _stability_objective(audit: pd.DataFrame) -> float:
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
    feat = _features(df)
    frozen_metrics = _source_metrics(df, "frozen_final_score", "frozen")

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
            "source_gate_mean",
            "use_split",
        ]
    ].copy()
    final = feat["final"]
    pf = feat["pf"]
    for mode in args.modes.split(","):
        for gate_name in args.gates.split(","):
            for threshold in args.thresholds:
                if gate_name in {"all", "sample_gate", "soft_sample_gate", "use_split", "no_split"} and threshold != args.thresholds[0]:
                    continue
                gate = _gate(feat, gate_name, threshold)
                if gate.mean() <= 0.0:
                    continue
                for weight in args.weights:
                    col = f"pfsg_{mode}_{gate_name}_t{_format_float(threshold)}_w{_format_float(weight)}"
                    scored = df.copy()
                    scored[col] = _correct(final, pf, gate, weight, mode)
                    metrics = _source_metrics(scored, col, "candidate")
                    audit = frozen_metrics.merge(metrics, on=["source_model", "n_fake"], validate="one_to_one")
                    audit["dataset"] = args.dataset
                    audit["candidate_col"] = col
                    audit["mode"] = mode
                    audit["gate"] = gate_name
                    audit["threshold"] = threshold
                    audit["weight"] = weight
                    audit["delta_auc"] = audit["candidate_auc"] - audit["frozen_auc"]
                    audit["delta_ap"] = audit["candidate_ap"] - audit["frozen_ap"]
                    per_source_rows.append(audit)

                    regressions = int(((audit["delta_auc"] < 0.0) | (audit["delta_ap"] < 0.0)).sum())
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
                        "regressed_sources": regressions,
                        "improved_auc_sources": int((audit["delta_auc"] > 0.0).sum()),
                        "improved_ap_sources": int((audit["delta_ap"] > 0.0).sum()),
                        "stability_objective": _stability_objective(audit),
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
    parser.add_argument("--weights", type=_parse_floats, default=_parse_floats("0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.10,0.12,0.15,0.20"))
    parser.add_argument("--thresholds", type=_parse_floats, default=_parse_floats("-0.20,-0.10,-0.05,0.0,0.03,0.05,0.08,0.10,0.12,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80,0.90"))
    parser.add_argument("--modes", default="convex,suppress_overfinal,toward_min,directional")
    parser.add_argument(
        "--gates",
        default=(
            "all,sample_gate,soft_sample_gate,"
            "model_spread_ge,final_patch_gap_ge,alpha_patch_gap_ge,final_over_pf_ge,"
            "pf_over_final_ge,persistence_le,global_le,"
            "split_minus_universal_le,persistence_minus_base_le,over_pf_and_spread_ge,"
            "gap_and_spread_ge,"
            "over_pf_and_persistence_le,over_pf_and_global_le"
        ),
    )
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
