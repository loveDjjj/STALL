#!/usr/bin/env python3
"""Evaluate patch-frequency additive candidates inside the frozen fallback system.

This keeps the global alpha and sample fallback rule fixed, and only replaces
the patch rank input with frequency-additive patch ranks.
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


def _auc_ap(real_scores: pd.Series, fake_scores: pd.Series) -> tuple[float, float]:
    y = [1] * len(real_scores) + [0] * len(fake_scores)
    s = [*real_scores.astype(float), *fake_scores.astype(float)]
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def _source_metrics(df: pd.DataFrame, score_col: str, prefix: str) -> pd.DataFrame:
    real = df[df["subset"].str.lower() == "real"]
    fake = df[df["subset"].str.lower() != "real"]
    rows = []
    real_scores = real[score_col]
    for source, group in fake.groupby("source_model", sort=True):
        auc, ap = _auc_ap(real_scores, group[score_col])
        rows.append(
            {
                "source_model": source,
                f"{prefix}_auc": auc,
                f"{prefix}_ap": ap,
                "n_fake": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _score_fallback(
    df: pd.DataFrame,
    patch_col: str,
    args: argparse.Namespace,
    prefix: str,
) -> pd.DataFrame:
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
    out[f"{prefix}_final_score"] = np.where(
        use_split,
        out[f"{prefix}_split_score"],
        out[f"{prefix}_universal_score"],
    )
    return out


def _parse_candidate_cols(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_weights(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _format_weight(weight: float) -> str:
    text = f"{weight:.6g}".replace(".", "p")
    return f"w{text}"


def _real_calibrated_rank(scores: pd.Series, real_mask: pd.Series) -> np.ndarray:
    real_values = np.sort(scores[real_mask].to_numpy(float))
    if len(real_values) == 0:
        raise ValueError("Need at least one real sample to calibrate frequency ranks")
    ranks = np.searchsorted(real_values, scores.to_numpy(float), side="right") / float(len(real_values))
    return ranks.astype(float)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frozen = _read(
        args.frozen_final_scores_csv,
        ["global_rank", "raw_rank", "persistence_rank", "final_score"],
    ).rename(columns={"final_score": "frozen_final_score"})
    if args.frequency_signal_col:
        frequency = _read(args.frequency_scores_csv, [args.frequency_signal_col])
        df = frozen.merge(frequency, on=KEY_COLUMNS, how="inner", validate="one_to_one")
        if len(df) == 0:
            raise ValueError("No overlap between frozen final scores and frequency signal")
        real_mask = df["subset"].str.lower() == "real"
        frequency_rank = _real_calibrated_rank(df[args.frequency_signal_col], real_mask)
        candidate_cols = []
        for weight in args.weights:
            candidate_col = f"frequency_add_{_format_weight(weight)}"
            df[candidate_col] = (1.0 - weight) * df["raw_rank"].to_numpy(float) + weight * frequency_rank
            candidate_cols.append(candidate_col)
    else:
        candidate_cols = args.candidate_cols
        candidates = _read(args.frequency_scores_csv, candidate_cols)
        df = frozen.merge(candidates, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(df) == 0:
        raise ValueError("No overlap between frozen final scores and frequency candidates")

    # Reconstruct the raw-patch fallback on the same overlap, so deltas are not
    # affected by using a debug subset instead of the full original CSV.
    raw_scored = _score_fallback(df, "raw_rank", args, "raw")
    raw_metrics = _source_metrics(raw_scored, "raw_final_score", "raw_system")
    frozen_metrics = _source_metrics(raw_scored, "frozen_final_score", "frozen_file")

    summary_rows = []
    per_source_rows = []
    score_frames = [raw_scored]
    for candidate_col in candidate_cols:
        scored = _score_fallback(df, candidate_col, args, candidate_col)
        score_frames.append(scored[[*KEY_COLUMNS, f"{candidate_col}_final_score"]])
        cand_metrics = _source_metrics(scored, f"{candidate_col}_final_score", "candidate")
        audit = raw_metrics.merge(cand_metrics, on=["source_model", "n_fake"], validate="one_to_one")
        audit["candidate_col"] = candidate_col
        audit["dataset"] = args.dataset
        audit["delta_auc"] = audit["candidate_auc"] - audit["raw_system_auc"]
        audit["delta_ap"] = audit["candidate_ap"] - audit["raw_system_ap"]
        per_source_rows.append(audit)
        summary_rows.append(
            {
                "dataset": args.dataset,
                "candidate_col": candidate_col,
                "n_rows": int(len(scored)),
                "n_sources": int(len(audit)),
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
                "raw_mean_gate": float(raw_scored["raw_sample_gate"].mean()),
                "candidate_mean_gate": float(scored[f"{candidate_col}_sample_gate"].mean()),
                "frozen_file_avg_auc": float(frozen_metrics["frozen_file_auc"].mean()),
                "frozen_file_avg_ap": float(frozen_metrics["frozen_file_ap"].mean()),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["avg_delta_auc", "avg_delta_ap"], ascending=False)
    per_source = pd.concat(per_source_rows, ignore_index=True) if per_source_rows else pd.DataFrame()
    scores = raw_scored[
        [
            *KEY_COLUMNS,
            "global_rank",
            "raw_rank",
            "persistence_rank",
            "frozen_final_score",
            "raw_final_score",
            "raw_sample_gate",
            "raw_use_split",
        ]
    ].copy()
    for frame in score_frames[1:]:
        scores = scores.merge(frame, on=KEY_COLUMNS, how="left", validate="one_to_one")
    return scores, summary, per_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--frozen-final-scores-csv", type=Path, required=True)
    parser.add_argument("--frequency-scores-csv", type=Path, required=True)
    parser.add_argument("--candidate-cols", type=_parse_candidate_cols, default=_parse_candidate_cols("frequency_add_w0p03,frequency_add_w0p05,frequency_add_w0p07,frequency_add_w0p1"))
    parser.add_argument("--frequency-signal-col")
    parser.add_argument("--weights", type=_parse_weights, default=_parse_weights("0.03,0.05,0.07,0.10"))
    parser.add_argument("--output-scores-csv", type=Path, required=True)
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    parser.add_argument("--output-per-source-csv", type=Path, required=True)
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
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved scores -> {args.output_scores_csv}")
    print(f"Saved summary -> {args.output_summary_csv}")


if __name__ == "__main__":
    main()
