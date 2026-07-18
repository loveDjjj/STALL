#!/usr/bin/env python3
"""Sweep PatchField as a reliability correction inside the alpha=0.60 fallback.

The goal is not to add another branch. PatchField is treated as a calibrated
patch-rank correction, then the existing sample fallback rule is reused.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


KEY_COLUMNS = ["subset", "source_model", "filename"]


def _read(path: Path, cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in KEY_COLUMNS + cols if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    out = df[KEY_COLUMNS + cols].copy()
    for col in KEY_COLUMNS:
        out[col] = out[col].astype(str)
    return out


def _parse_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _parse_strs(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _format_float(value: float) -> str:
    return f"{value:.6g}".replace(".", "p").replace("-", "m")


def _real_rank(values: pd.Series, real_mask: pd.Series) -> np.ndarray:
    real = np.sort(values[real_mask].to_numpy(float))
    if len(real) == 0:
        raise ValueError("Need real rows for calibration")
    return (np.searchsorted(real, values.to_numpy(float), side="right") / float(len(real))).astype(float)


def _auc_ap(real_scores: pd.Series, fake_scores: pd.Series) -> tuple[float, float]:
    y = np.r_[np.ones(len(real_scores), dtype=int), np.zeros(len(fake_scores), dtype=int)]
    s = np.r_[real_scores.to_numpy(float), fake_scores.to_numpy(float)]
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def _source_metrics(df: pd.DataFrame, score_col: str, prefix: str) -> pd.DataFrame:
    real = df[df["subset"].str.lower() == "real"]
    fake = df[df["subset"].str.lower() != "real"]
    rows = []
    for source, group in fake.groupby("source_model", sort=True):
        auc, ap = _auc_ap(real[score_col], group[score_col])
        rows.append({"source_model": source, "n_fake": len(group), f"{prefix}_auc": auc, f"{prefix}_ap": ap})
    return pd.DataFrame(rows)


def _score_fallback(df: pd.DataFrame, patch_col: str, args: argparse.Namespace, prefix: str) -> pd.DataFrame:
    out = df.copy()
    out[f"{prefix}_base_score"] = args.alpha * out["global_rank"] + (1.0 - args.alpha) * out[patch_col]
    out[f"{prefix}_disagreement"] = np.abs(out["global_rank"].to_numpy(float) - out[patch_col].to_numpy(float))
    out[f"{prefix}_persistence_conf"] = np.abs(out["persistence_rank"].to_numpy(float) - 0.5) * 2.0
    out[f"{prefix}_sample_gate"] = (
        (out[f"{prefix}_disagreement"].to_numpy(float) >= args.disagreement_threshold)
        & (out[f"{prefix}_persistence_conf"].to_numpy(float) >= args.confidence_threshold)
    ).astype(float)
    source_gate = (
        out[out["subset"].str.lower() != "real"]
        .groupby("source_model")[f"{prefix}_sample_gate"]
        .mean()
        .rename(f"{prefix}_source_gate_mean")
    )
    out = out.merge(source_gate, on="source_model", how="left")
    out[f"{prefix}_source_gate_mean"] = out[f"{prefix}_source_gate_mean"].fillna(0.0)

    universal_weight = args.universal_weight * out[f"{prefix}_sample_gate"]
    out[f"{prefix}_universal_score"] = (
        (1.0 - universal_weight) * out[f"{prefix}_base_score"] + universal_weight * out["persistence_rank"]
    )

    is_real = out["subset"].str.lower() == "real"
    fake_source_enabled = (out[f"{prefix}_source_gate_mean"] >= args.source_gate_threshold) & (~is_real)
    split_weight = out[f"{prefix}_sample_gate"] * (
        args.real_weight * is_real.to_numpy(float) + args.fake_weight * fake_source_enabled.to_numpy(float)
    )
    out[f"{prefix}_split_score"] = (
        (1.0 - split_weight) * out[f"{prefix}_base_score"] + split_weight * out["persistence_rank"]
    )
    out[f"{prefix}_split_minus_universal"] = out[f"{prefix}_split_score"] - out[f"{prefix}_universal_score"]
    use_split = out[f"{prefix}_split_minus_universal"].to_numpy(float) < args.selector_threshold
    if args.real_policy == "split":
        use_split = np.where(is_real, True, use_split)
    elif args.real_policy == "universal":
        use_split = np.where(is_real, False, use_split)
    elif args.real_policy != "rule":
        raise ValueError(f"Unsupported real policy: {args.real_policy}")
    out[f"{prefix}_use_split"] = use_split
    out[f"{prefix}_final_score"] = np.where(use_split, out[f"{prefix}_split_score"], out[f"{prefix}_universal_score"])
    return out


def _base_gate(df: pd.DataFrame, gate_name: str, threshold: float, real_policy: str) -> np.ndarray:
    is_real = df["subset"].str.lower() == "real"
    if gate_name == "all":
        gate = np.ones(len(df), dtype=float)
    elif gate_name == "raw_sample_gate":
        gate = df["raw_sample_gate"].to_numpy(float)
    elif gate_name == "source_gate_ge":
        gate = (df["raw_source_gate_mean"].to_numpy(float) >= threshold).astype(float)
    elif gate_name == "disagreement_ge":
        gate = (df["raw_disagreement"].to_numpy(float) >= threshold).astype(float)
    elif gate_name == "persistence_conf_ge":
        gate = (df["raw_persistence_conf"].to_numpy(float) >= threshold).astype(float)
    elif gate_name == "pf_gap_ge":
        gate = (np.abs(df["patchfield_rank"].to_numpy(float) - df["raw_rank"].to_numpy(float)) >= threshold).astype(float)
    else:
        raise ValueError(f"Unknown gate: {gate_name}")

    if real_policy == "same":
        return gate
    if real_policy == "real_on":
        return np.where(is_real, 1.0, gate)
    if real_policy == "real_off":
        return np.where(is_real, 0.0, gate)
    if real_policy == "real_sample":
        return np.where(is_real, df["raw_sample_gate"].to_numpy(float), gate)
    raise ValueError(f"Unknown correction real policy: {real_policy}")


def _corrected_rank(raw: np.ndarray, pf: np.ndarray, gate: np.ndarray, weight: float, mode: str) -> np.ndarray:
    strength = np.clip(weight * gate, 0.0, 1.0)
    if mode == "convex":
        out = (1.0 - strength) * raw + strength * pf
    elif mode == "suppress_overraw":
        out = raw - strength * np.maximum(raw - pf, 0.0)
    elif mode == "lift_underraw":
        out = raw + strength * np.maximum(pf - raw, 0.0)
    elif mode == "toward_min":
        out = (1.0 - strength) * raw + strength * np.minimum(raw, pf)
    elif mode == "toward_max":
        out = (1.0 - strength) * raw + strength * np.maximum(raw, pf)
    else:
        raise ValueError(f"Unknown correction mode: {mode}")
    return np.clip(out, 0.0, 1.0)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    scores = df[[*KEY_COLUMNS, "global_rank", "raw_rank", "persistence_rank", "patchfield_rank", "frozen_final_score", "raw_final_score"]].copy()
    raw = df["raw_rank"].to_numpy(float)
    pf = df["patchfield_rank"].to_numpy(float)
    for mode in args.modes:
        for gate_name in args.gates:
            for threshold in args.gate_thresholds:
                for correction_real_policy in args.correction_real_policies:
                    gate = _base_gate(df, gate_name, threshold, correction_real_policy)
                    if gate_name == "all" and threshold != args.gate_thresholds[0]:
                        continue
                    for weight in args.weights:
                        col = (
                            f"pfrel_{mode}_{gate_name}_t{_format_float(threshold)}_"
                            f"{correction_real_policy}_w{_format_float(weight)}"
                        )
                        cand_input = df.copy()
                        cand_input[col] = _corrected_rank(raw, pf, gate, weight, mode)
                        scored = _score_fallback(cand_input, col, args, col)
                        metrics = _source_metrics(scored, f"{col}_final_score", "candidate")
                        audit = raw_metrics.merge(metrics, on=["source_model", "n_fake"], validate="one_to_one")
                        audit["dataset"] = args.dataset
                        audit["candidate_col"] = col
                        audit["mode"] = mode
                        audit["gate"] = gate_name
                        audit["gate_threshold"] = threshold
                        audit["correction_real_policy"] = correction_real_policy
                        audit["weight"] = weight
                        audit["delta_auc"] = audit["candidate_auc"] - audit["raw_system_auc"]
                        audit["delta_ap"] = audit["candidate_ap"] - audit["raw_system_ap"]
                        per_source_rows.append(audit)
                        rows.append(
                            {
                                "dataset": args.dataset,
                                "patchfield_score_col": args.patchfield_score_col,
                                "candidate_col": col,
                                "mode": mode,
                                "gate": gate_name,
                                "gate_threshold": threshold,
                                "correction_real_policy": correction_real_policy,
                                "weight": weight,
                                "n_rows": len(scored),
                                "n_sources": len(audit),
                                "gate_mean": float(gate.mean()),
                                "gate_fake_mean": float(gate[~real_mask.to_numpy()].mean()),
                                "raw_avg_auc": float(raw_metrics["raw_system_auc"].mean()),
                                "raw_avg_ap": float(raw_metrics["raw_system_ap"].mean()),
                                "candidate_avg_auc": float(audit["candidate_auc"].mean()),
                                "candidate_avg_ap": float(audit["candidate_ap"].mean()),
                                "avg_delta_auc": float(audit["delta_auc"].mean()),
                                "avg_delta_ap": float(audit["delta_ap"].mean()),
                                "min_delta_auc": float(audit["delta_auc"].min()),
                                "min_delta_ap": float(audit["delta_ap"].min()),
                                "improved_auc_sources": int((audit["delta_auc"] > 0).sum()),
                                "improved_ap_sources": int((audit["delta_ap"] > 0).sum()),
                                "frozen_file_avg_auc": float(frozen_metrics["frozen_file_auc"].mean()),
                                "frozen_file_avg_ap": float(frozen_metrics["frozen_file_ap"].mean()),
                            }
                        )
                        if len(scores.columns) < args.max_score_columns + len(KEY_COLUMNS):
                            scores = scores.merge(
                                scored[[*KEY_COLUMNS, f"{col}_final_score"]],
                                on=KEY_COLUMNS,
                                how="left",
                                validate="one_to_one",
                            )

    summary = pd.DataFrame(rows).sort_values(["candidate_avg_auc", "candidate_avg_ap"], ascending=False)
    per_source = pd.concat(per_source_rows, ignore_index=True) if per_source_rows else pd.DataFrame()
    return scores, summary, per_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--frozen-final-scores-csv", type=Path, required=True)
    parser.add_argument("--patchfield-scores-csv", type=Path, required=True)
    parser.add_argument("--patchfield-score-col", required=True)
    parser.add_argument("--output-scores-csv", type=Path, required=True)
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    parser.add_argument("--output-per-source-csv", type=Path, required=True)
    parser.add_argument("--weights", type=_parse_floats, default=_parse_floats("0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50"))
    parser.add_argument("--modes", type=_parse_strs, default=_parse_strs("convex,suppress_overraw,toward_min"))
    parser.add_argument("--gates", type=_parse_strs, default=_parse_strs("all,raw_sample_gate,source_gate_ge,disagreement_ge,pf_gap_ge"))
    parser.add_argument("--gate-thresholds", type=_parse_floats, default=_parse_floats("0.0,0.15,0.20,0.25,0.30,0.35,0.40"))
    parser.add_argument("--correction-real-policies", type=_parse_strs, default=_parse_strs("same,real_on,real_off,real_sample"))
    parser.add_argument("--max-score-columns", type=int, default=40)
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

    scores, summary, per_source = run(args)
    args.output_scores_csv.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_scores_csv, index=False)
    summary.to_csv(args.output_summary_csv, index=False)
    per_source.to_csv(args.output_per_source_csv, index=False)
    print(summary.head(40).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved scores -> {args.output_scores_csv}")
    print(f"Saved summary -> {args.output_summary_csv}")
    print(f"Saved per-source -> {args.output_per_source_csv}")


if __name__ == "__main__":
    main()
