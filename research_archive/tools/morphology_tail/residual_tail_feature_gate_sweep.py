#!/usr/bin/env python3
"""Sweep auxiliary-feature gates for residual hard-tail correction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


KEY_COLUMNS = ["subset", "source_model", "filename"]
SCORE_COLUMNS = ["final_score", "base_score", "split_score", "universal_score"]


def _parse_csv(value: str) -> list[str]:
    return [x for x in value.split(",") if x]


def _parse_floats(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x]


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in KEY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    for col in KEY_COLUMNS:
        df[col] = df[col].astype(str)
    return df


def _load(frozen_csv: Path, feature_csv: Path, requested_features: list[str]) -> pd.DataFrame:
    frozen = _read(frozen_csv)
    missing = [c for c in SCORE_COLUMNS if c not in frozen.columns]
    if missing:
        raise ValueError(f"{frozen_csv} missing columns: {missing}")
    features = _read(feature_csv)
    missing_features = [c for c in requested_features if c not in features.columns]
    if missing_features:
        raise ValueError(f"{feature_csv} missing requested features: {missing_features}")
    feature_cols = requested_features
    overlap = frozen[[*KEY_COLUMNS, *SCORE_COLUMNS]].merge(
        features[[*KEY_COLUMNS, *feature_cols]],
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    if overlap.empty:
        raise ValueError("No overlapping rows between frozen and feature CSV")
    return overlap


def _is_real(df: pd.DataFrame) -> pd.Series:
    return df["subset"].astype(str).str.lower() == "real"


def _auc_ap(real_scores: np.ndarray, fake_scores: np.ndarray) -> tuple[float, float]:
    y = np.concatenate([np.ones(len(real_scores)), np.zeros(len(fake_scores))])
    s = np.concatenate([real_scores, fake_scores])
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def _source_metrics(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    real = df[_is_real(df)]
    fake = df[~_is_real(df)]
    real_scores = real[score_col].to_numpy(float)
    rows = []
    for source, group in fake.groupby("source_model", sort=True):
        auc, ap = _auc_ap(real_scores, group[score_col].to_numpy(float))
        rows.append({"source_model": source, "auc": auc, "ap": ap, "n_fake": int(len(group))})
    return pd.DataFrame(rows)


def _target_values(df: pd.DataFrame, target: str, cap_value: float) -> np.ndarray:
    if target == "cap_at_real_q10":
        return np.full(len(df), cap_value, dtype=float)
    if target == "base":
        return df["base_score"].to_numpy(float)
    if target == "min_base_split":
        return np.minimum(df["base_score"].to_numpy(float), df["split_score"].to_numpy(float))
    if target == "min_base_universal":
        return np.minimum(df["base_score"].to_numpy(float), df["universal_score"].to_numpy(float))
    raise ValueError(f"Unsupported target: {target}")


def _feature_threshold(values: np.ndarray, mode: str, q: float, real_mask: np.ndarray, fake_mask: np.ndarray) -> float:
    finite = np.isfinite(values)
    if mode == "all":
        pool = finite
    elif mode == "real":
        pool = finite & real_mask
    elif mode == "fake":
        pool = finite & fake_mask
    else:
        raise ValueError(f"Unsupported threshold pool: {mode}")
    if pool.sum() == 0:
        raise ValueError(f"No finite values for threshold pool {mode}")
    return float(np.quantile(values[pool], q))


def _mask(
    df: pd.DataFrame,
    feature: str,
    direction: str,
    threshold: float,
    score_threshold: float,
    apply_mode: str,
) -> np.ndarray:
    values = df[feature].to_numpy(float)
    if direction == "high":
        mask = values >= threshold
    elif direction == "low":
        mask = values <= threshold
    else:
        raise ValueError(f"Unsupported direction: {direction}")
    mask &= df["final_score"].to_numpy(float) >= score_threshold
    if apply_mode == "all_samples":
        return mask
    if apply_mode == "fake_only_oracle":
        return mask & (~_is_real(df)).to_numpy()
    raise ValueError(f"Unsupported apply mode: {apply_mode}")


def _evaluate(dataset: str, df: pd.DataFrame, score: np.ndarray, mask: np.ndarray, meta: dict[str, object]) -> tuple[dict[str, object], pd.DataFrame]:
    work = df.copy()
    work["candidate_score"] = score
    base = _source_metrics(work, "final_score").rename(columns={"auc": "base_auc", "ap": "base_ap"})
    cand = _source_metrics(work, "candidate_score").rename(columns={"auc": "candidate_auc", "ap": "candidate_ap"})
    per_source = base.merge(cand, on=["source_model", "n_fake"], validate="one_to_one")
    per_source["delta_auc"] = per_source["candidate_auc"] - per_source["base_auc"]
    per_source["delta_ap"] = per_source["candidate_ap"] - per_source["base_ap"]
    per_source.insert(0, "dataset", dataset)
    for key, value in meta.items():
        per_source[key] = value
    real_mask = _is_real(work).to_numpy()
    fake_mask = ~real_mask
    summary = {
        "dataset": dataset,
        **meta,
        "avg_auc": float(per_source["candidate_auc"].mean()),
        "avg_ap": float(per_source["candidate_ap"].mean()),
        "delta_avg_auc": float(per_source["delta_auc"].mean()),
        "delta_avg_ap": float(per_source["delta_ap"].mean()),
        "min_delta_auc": float(per_source["delta_auc"].min()),
        "min_delta_ap": float(per_source["delta_ap"].min()),
        "n_negative_auc": int((per_source["delta_auc"] < 0).sum()),
        "n_negative_ap": int((per_source["delta_ap"] < 0).sum()),
        "n_triggered": int(mask.sum()),
        "n_triggered_real": int((mask & real_mask).sum()),
        "n_triggered_fake": int((mask & fake_mask).sum()),
        "triggered_real_rate": float((mask & real_mask).sum() / max(1, real_mask.sum())),
        "triggered_fake_rate": float((mask & fake_mask).sum() / max(1, fake_mask.sum())),
    }
    return summary, per_source


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    requested_features = _parse_csv(args.features)
    df = _load(args.frozen_csv, args.feature_csv, requested_features)
    real_mask = _is_real(df).to_numpy()
    fake_mask = ~real_mask
    cap_value = float(np.quantile(df.loc[real_mask, "final_score"].to_numpy(float), args.real_cap_q))
    score_threshold = float(np.quantile(df.loc[fake_mask, "final_score"].to_numpy(float), args.score_q))
    rows = []
    source_tables = []
    for apply_mode in _parse_csv(args.apply_modes):
        for feature in requested_features:
            values = df[feature].to_numpy(float)
            for direction in _parse_csv(args.directions):
                for threshold_pool in _parse_csv(args.threshold_pools):
                    for feature_q in _parse_floats(args.feature_quantiles):
                        q = feature_q if direction == "high" else 1.0 - feature_q
                        threshold = _feature_threshold(values, threshold_pool, q, real_mask, fake_mask)
                        mask = _mask(df, feature, direction, threshold, score_threshold, apply_mode)
                        for strength in _parse_floats(args.strengths):
                            for target in _parse_csv(args.targets):
                                base_score = df["final_score"].to_numpy(float)
                                target_values = _target_values(df, target, cap_value)
                                score = base_score.copy()
                                score[mask] = (1.0 - strength) * base_score[mask] + strength * target_values[mask]
                                meta = {
                                    "feature_family": args.feature_family,
                                    "apply_mode": apply_mode,
                                    "feature": feature,
                                    "direction": direction,
                                    "threshold_pool": threshold_pool,
                                    "feature_q": feature_q,
                                    "feature_threshold": threshold,
                                    "score_q": args.score_q,
                                    "score_threshold": score_threshold,
                                    "real_cap_q": args.real_cap_q,
                                    "real_cap_value": cap_value,
                                    "strength": strength,
                                    "target": target,
                                }
                                summary, per_source = _evaluate(args.dataset, df, score, mask, meta)
                                rows.append(summary)
                                source_tables.append(per_source)

    summary = pd.DataFrame(rows)
    per_source = pd.concat(source_tables, ignore_index=True)
    summary["passes_source_nonregression"] = (
        (summary["n_negative_auc"] == 0)
        & (summary["n_negative_ap"] == 0)
        & (summary["min_delta_auc"] >= 0)
        & (summary["min_delta_ap"] >= 0)
    )
    best = summary.sort_values(
        ["passes_source_nonregression", "delta_avg_auc", "delta_avg_ap", "min_delta_ap"],
        ascending=[False, False, False, False],
    )
    return summary, per_source, best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--feature-family", required=True)
    parser.add_argument("--frozen-csv", type=Path, required=True)
    parser.add_argument("--feature-csv", type=Path, required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--directions", default="high")
    parser.add_argument("--threshold-pools", default="all,real")
    parser.add_argument("--feature-quantiles", default="0.90,0.95,0.98")
    parser.add_argument("--apply-modes", default="all_samples,fake_only_oracle")
    parser.add_argument("--score-q", type=float, default=0.90)
    parser.add_argument("--real-cap-q", type=float, default=0.10)
    parser.add_argument("--strengths", default="0.10,0.25,0.50,1.00")
    parser.add_argument("--targets", default="cap_at_real_q10,base,min_base_split,min_base_universal")
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    parser.add_argument("--output-source-csv", type=Path, required=True)
    parser.add_argument("--output-best-csv", type=Path, required=True)
    args = parser.parse_args()

    summary, per_source, best = run(args)
    args.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_source_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_best_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary_csv, index=False)
    per_source.to_csv(args.output_source_csv, index=False)
    best.to_csv(args.output_best_csv, index=False)
    print(best.head(20).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved summary -> {args.output_summary_csv}")
    print(f"Saved per-source -> {args.output_source_csv}")
    print(f"Saved best -> {args.output_best_csv}")


if __name__ == "__main__":
    main()
