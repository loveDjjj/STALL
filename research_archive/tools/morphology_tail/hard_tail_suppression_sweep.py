#!/usr/bin/env python3
"""Sweep conservative sample-level suppression for hard fake tails.

This is a diagnostic tool. The deployable mode applies the same rule to all
rows. The fake_only_oracle mode estimates upside if suppression were perfectly
restricted to fake rows and must not be promoted as a default rule.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


KEY_COLUMNS = ["subset", "source_model", "filename"]
REQUIRED_SCORE_COLUMNS = ["final_score", "base_score", "universal_score", "split_score", "persistence_rank"]


def _parse_floats(value: str) -> list[float]:
    return [float(x) for x in value.split(",")]


def _parse_feature_thresholds(value: str) -> list[tuple[str, float, str]]:
    out = []
    for item in value.split(","):
        feature, threshold, direction = item.split(":")
        if direction not in {"ge", "lt"}:
            raise ValueError(f"Unsupported direction in {item}")
        out.append((feature, float(threshold), direction))
    return out


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in [*KEY_COLUMNS, *REQUIRED_SCORE_COLUMNS] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    for col in KEY_COLUMNS:
        df[col] = df[col].astype(str)
    for col in REQUIRED_SCORE_COLUMNS:
        df[col] = df[col].astype(float)
    return df


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


def _real_quantile_threshold(df: pd.DataFrame, q: float) -> float:
    return float(np.quantile(df.loc[_is_real(df), "final_score"].to_numpy(float), q))


def _target_score(df: pd.DataFrame, target: str, threshold: float) -> np.ndarray:
    if target == "base":
        return df["base_score"].to_numpy(float)
    if target == "universal":
        return df["universal_score"].to_numpy(float)
    if target == "split":
        return df["split_score"].to_numpy(float)
    if target == "persistence":
        return df["persistence_rank"].to_numpy(float)
    if target == "min_base_split":
        return np.minimum(df["base_score"].to_numpy(float), df["split_score"].to_numpy(float))
    if target == "min_base_universal":
        return np.minimum(df["base_score"].to_numpy(float), df["universal_score"].to_numpy(float))
    if target == "cap_at_threshold":
        return np.full(len(df), threshold, dtype=float)
    raise ValueError(f"Unsupported target: {target}")


def _feature_mask(df: pd.DataFrame, feature: str, threshold: float, direction: str) -> np.ndarray:
    if feature not in df.columns:
        raise ValueError(f"Missing feature: {feature}")
    values = df[feature].to_numpy(float)
    if direction == "ge":
        return values >= threshold
    if direction == "lt":
        return values < threshold
    raise ValueError(f"Unsupported direction: {direction}")


def _candidate_score(
    df: pd.DataFrame,
    apply_mode: str,
    score_q: float,
    feature: str,
    feature_threshold: float,
    feature_direction: str,
    strength: float,
    target: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    threshold = _real_quantile_threshold(df, score_q)
    mask = df["final_score"].to_numpy(float) >= threshold
    mask &= _feature_mask(df, feature, feature_threshold, feature_direction)
    if apply_mode == "all_samples":
        pass
    elif apply_mode == "fake_only_oracle":
        mask &= (~_is_real(df)).to_numpy()
    else:
        raise ValueError(f"Unsupported apply mode: {apply_mode}")
    target_values = _target_score(df, target, threshold)
    base = df["final_score"].to_numpy(float)
    suppressed = base.copy()
    suppressed[mask] = (1.0 - strength) * base[mask] + strength * target_values[mask]
    return suppressed, mask, threshold


def _evaluate_candidate(
    dataset: str,
    df: pd.DataFrame,
    candidate_score: np.ndarray,
    mask: np.ndarray,
    metadata: dict[str, object],
) -> tuple[dict[str, object], pd.DataFrame]:
    work = df.copy()
    work["candidate_score"] = candidate_score
    base_metrics = _source_metrics(work, "final_score").rename(columns={"auc": "base_auc", "ap": "base_ap"})
    cand_metrics = _source_metrics(work, "candidate_score").rename(columns={"auc": "candidate_auc", "ap": "candidate_ap"})
    per_source = base_metrics.merge(cand_metrics, on=["source_model", "n_fake"], validate="one_to_one")
    per_source["delta_auc"] = per_source["candidate_auc"] - per_source["base_auc"]
    per_source["delta_ap"] = per_source["candidate_ap"] - per_source["base_ap"]
    per_source.insert(0, "dataset", dataset)
    for key, value in metadata.items():
        per_source[key] = value

    fake_mask = (~_is_real(work)).to_numpy()
    real_mask = _is_real(work).to_numpy()
    summary = {
        "dataset": dataset,
        **metadata,
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


def run_dataset(dataset: str, path: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _load(path)
    rows = []
    source_tables = []
    for apply_mode in args.apply_modes.split(","):
        for score_q in _parse_floats(args.score_quantiles):
            for feature, feature_threshold, feature_direction in _parse_feature_thresholds(args.feature_thresholds):
                for strength in _parse_floats(args.strengths):
                    for target in args.targets.split(","):
                        score, mask, threshold = _candidate_score(
                            df=df,
                            apply_mode=apply_mode,
                            score_q=score_q,
                            feature=feature,
                            feature_threshold=feature_threshold,
                            feature_direction=feature_direction,
                            strength=strength,
                            target=target,
                        )
                        metadata = {
                            "apply_mode": apply_mode,
                            "score_q": score_q,
                            "score_threshold": threshold,
                            "feature": feature,
                            "feature_threshold": feature_threshold,
                            "feature_direction": feature_direction,
                            "strength": strength,
                            "target": target,
                        }
                        summary, per_source = _evaluate_candidate(dataset, df, score, mask, metadata)
                        rows.append(summary)
                        source_tables.append(per_source)
    return pd.DataFrame(rows), pd.concat(source_tables, ignore_index=True)


def _aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    key_cols = [
        "apply_mode",
        "score_q",
        "feature",
        "feature_threshold",
        "feature_direction",
        "strength",
        "target",
    ]
    rows = []
    for key, group in summary.groupby(key_cols, sort=True):
        row = dict(zip(key_cols, key))
        row.update(
            {
                "n_datasets": int(len(group)),
                "mean_delta_avg_auc": float(group["delta_avg_auc"].mean()),
                "mean_delta_avg_ap": float(group["delta_avg_ap"].mean()),
                "min_delta_avg_auc": float(group["delta_avg_auc"].min()),
                "min_delta_avg_ap": float(group["delta_avg_ap"].min()),
                "max_negative_auc": int(group["n_negative_auc"].max()),
                "max_negative_ap": int(group["n_negative_ap"].max()),
                "min_source_delta_auc": float(group["min_delta_auc"].min()),
                "min_source_delta_ap": float(group["min_delta_ap"].min()),
                "total_triggered_real": int(group["n_triggered_real"].sum()),
                "total_triggered_fake": int(group["n_triggered_fake"].sum()),
            }
        )
        row["passes_source_nonregression"] = (
            row["max_negative_auc"] == 0
            and row["max_negative_ap"] == 0
            and row["min_source_delta_auc"] >= 0
            and row["min_source_delta_ap"] >= 0
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["passes_source_nonregression", "mean_delta_avg_auc", "mean_delta_avg_ap", "min_source_delta_ap"],
        ascending=[False, False, False, False],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="Dataset/file pairs, e.g. genvideo=path.csv")
    parser.add_argument("--apply-modes", default="all_samples,fake_only_oracle")
    parser.add_argument("--score-quantiles", default="0.90,0.95")
    parser.add_argument(
        "--feature-thresholds",
        default=(
            "sample_gate:0.0:ge,sample_gate:0.5:ge,"
            "split_minus_universal:0.0:lt,persistence_minus_base:0.0:lt"
        ),
    )
    parser.add_argument("--strengths", default="0.10,0.25,0.50")
    parser.add_argument("--targets", default="base,min_base_split,min_base_universal,cap_at_threshold")
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    parser.add_argument("--output-source-csv", type=Path, required=True)
    parser.add_argument("--output-aggregate-csv", type=Path, required=True)
    args = parser.parse_args()

    summaries = []
    source_tables = []
    for item in args.input:
        dataset, path_str = item.split("=", 1)
        summary, per_source = run_dataset(dataset, Path(path_str), args)
        summaries.append(summary)
        source_tables.append(per_source)

    summary_df = pd.concat(summaries, ignore_index=True)
    source_df = pd.concat(source_tables, ignore_index=True)
    aggregate = _aggregate(summary_df)

    args.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_source_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_aggregate_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output_summary_csv, index=False)
    source_df.to_csv(args.output_source_csv, index=False)
    aggregate.to_csv(args.output_aggregate_csv, index=False)
    print(aggregate.head(20).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved summary -> {args.output_summary_csv}")
    print(f"Saved per-source -> {args.output_source_csv}")
    print(f"Saved aggregate -> {args.output_aggregate_csv}")


if __name__ == "__main__":
    main()
