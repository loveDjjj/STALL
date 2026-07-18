#!/usr/bin/env python3
"""Held-out source validation for residual-tail feature gates."""

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


def _load(frozen_csv: Path, feature_csv: Path, features: list[str]) -> pd.DataFrame:
    frozen = _read(frozen_csv)
    missing = [c for c in SCORE_COLUMNS if c not in frozen.columns]
    if missing:
        raise ValueError(f"{frozen_csv} missing columns: {missing}")
    aux = _read(feature_csv)
    missing_features = [c for c in features if c not in aux.columns]
    if missing_features:
        raise ValueError(f"{feature_csv} missing features: {missing_features}")
    df = frozen[[*KEY_COLUMNS, *SCORE_COLUMNS]].merge(
        aux[[*KEY_COLUMNS, *features]],
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    if df.empty:
        raise ValueError("No overlapping rows")
    return df


def _is_real(df: pd.DataFrame) -> pd.Series:
    return df["subset"].astype(str).str.lower() == "real"


def _auc_ap(real_scores: np.ndarray, fake_scores: np.ndarray) -> tuple[float, float]:
    y = np.concatenate([np.ones(len(real_scores)), np.zeros(len(fake_scores))])
    s = np.concatenate([real_scores, fake_scores])
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def _source_auc_ap(df: pd.DataFrame, score: np.ndarray, source: str) -> tuple[float, float]:
    work = df.copy()
    work["_score"] = score
    real_scores = work.loc[_is_real(work), "_score"].to_numpy(float)
    fake_scores = work.loc[(~_is_real(work)) & (work["source_model"] == source), "_score"].to_numpy(float)
    return _auc_ap(real_scores, fake_scores)


def _target(df: pd.DataFrame, name: str, cap_value: float) -> np.ndarray:
    if name == "cap_at_real_q10":
        return np.full(len(df), cap_value, dtype=float)
    if name == "base":
        return df["base_score"].to_numpy(float)
    if name == "min_base_split":
        return np.minimum(df["base_score"].to_numpy(float), df["split_score"].to_numpy(float))
    if name == "min_base_universal":
        return np.minimum(df["base_score"].to_numpy(float), df["universal_score"].to_numpy(float))
    raise ValueError(f"Unsupported target: {name}")


def _threshold(values: np.ndarray, pool_mask: np.ndarray, q: float) -> float:
    finite = np.isfinite(values) & pool_mask
    if finite.sum() == 0:
        raise ValueError("No finite values for threshold")
    return float(np.quantile(values[finite], q))


def _candidate_mask(
    df: pd.DataFrame,
    feature: str,
    direction: str,
    threshold: float,
    score_threshold: float,
) -> np.ndarray:
    values = df[feature].to_numpy(float)
    if direction == "high":
        mask = values >= threshold
    elif direction == "low":
        mask = values <= threshold
    else:
        raise ValueError(f"Unsupported direction: {direction}")
    mask &= df["final_score"].to_numpy(float) >= score_threshold
    return mask


def _apply_score(df: pd.DataFrame, mask: np.ndarray, strength: float, target_name: str, cap_value: float) -> np.ndarray:
    base = df["final_score"].to_numpy(float)
    target = _target(df, target_name, cap_value)
    out = base.copy()
    out[mask] = (1.0 - strength) * base[mask] + strength * target[mask]
    return out


def _score_training_sources(
    df: pd.DataFrame,
    score: np.ndarray,
    train_sources: list[str],
) -> tuple[float, float, float, float, int, int]:
    base_score = df["final_score"].to_numpy(float)
    deltas = []
    for source in train_sources:
        base_auc, base_ap = _source_auc_ap(df, base_score, source)
        cand_auc, cand_ap = _source_auc_ap(df, score, source)
        deltas.append((cand_auc - base_auc, cand_ap - base_ap))
    arr = np.asarray(deltas, dtype=float)
    return (
        float(arr[:, 0].mean()),
        float(arr[:, 1].mean()),
        float(arr[:, 0].min()),
        float(arr[:, 1].min()),
        int((arr[:, 0] < 0).sum()),
        int((arr[:, 1] < 0).sum()),
    )


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = _parse_csv(args.features)
    df = _load(args.frozen_csv, args.feature_csv, features)
    real_mask = _is_real(df).to_numpy()
    fake_mask = ~real_mask
    fake_sources = sorted(df.loc[fake_mask, "source_model"].unique())
    cap_value = float(np.quantile(df.loc[real_mask, "final_score"].to_numpy(float), args.real_cap_q))
    score_threshold = float(np.quantile(df.loc[fake_mask, "final_score"].to_numpy(float), args.score_q))
    base_score = df["final_score"].to_numpy(float)

    rows = []
    selected_rows = []
    for heldout in fake_sources:
        train_sources = [s for s in fake_sources if s != heldout]
        train_fake_mask = fake_mask & df["source_model"].isin(train_sources).to_numpy()
        heldout_mask = fake_mask & (df["source_model"] == heldout).to_numpy()
        candidate_rows = []
        for feature in features:
            values = df[feature].to_numpy(float)
            for direction in _parse_csv(args.directions):
                for pool in _parse_csv(args.threshold_pools):
                    if pool == "all_train":
                        pool_mask = real_mask | train_fake_mask
                    elif pool == "real":
                        pool_mask = real_mask
                    elif pool == "train_fake":
                        pool_mask = train_fake_mask
                    else:
                        raise ValueError(f"Unsupported threshold pool: {pool}")
                    for feature_q in _parse_floats(args.feature_quantiles):
                        q = feature_q if direction == "high" else 1.0 - feature_q
                        threshold = _threshold(values, pool_mask, q)
                        base_mask = _candidate_mask(df, feature, direction, threshold, score_threshold)
                        for mode in _parse_csv(args.apply_modes):
                            if mode == "all_samples":
                                mask = base_mask
                            elif mode == "fake_only_oracle":
                                mask = base_mask & fake_mask
                            else:
                                raise ValueError(f"Unsupported mode: {mode}")
                            for strength in _parse_floats(args.strengths):
                                for target_name in _parse_csv(args.targets):
                                    score = _apply_score(df, mask, strength, target_name, cap_value)
                                    (
                                        train_mean_auc,
                                        train_mean_ap,
                                        train_min_auc,
                                        train_min_ap,
                                        train_neg_auc,
                                        train_neg_ap,
                                    ) = _score_training_sources(df, score, train_sources)
                                    base_auc, base_ap = _source_auc_ap(df, base_score, heldout)
                                    cand_auc, cand_ap = _source_auc_ap(df, score, heldout)
                                    row = {
                                        "dataset": args.dataset,
                                        "feature_family": args.feature_family,
                                        "heldout_source": heldout,
                                        "feature": feature,
                                        "direction": direction,
                                        "threshold_pool": pool,
                                        "feature_q": feature_q,
                                        "feature_threshold": threshold,
                                        "apply_mode": mode,
                                        "strength": strength,
                                        "target": target_name,
                                        "score_q": args.score_q,
                                        "score_threshold": score_threshold,
                                        "real_cap_q": args.real_cap_q,
                                        "real_cap_value": cap_value,
                                        "train_mean_delta_auc": train_mean_auc,
                                        "train_mean_delta_ap": train_mean_ap,
                                        "train_min_delta_auc": train_min_auc,
                                        "train_min_delta_ap": train_min_ap,
                                        "train_negative_auc": train_neg_auc,
                                        "train_negative_ap": train_neg_ap,
                                        "heldout_base_auc": base_auc,
                                        "heldout_base_ap": base_ap,
                                        "heldout_candidate_auc": cand_auc,
                                        "heldout_candidate_ap": cand_ap,
                                        "heldout_delta_auc": cand_auc - base_auc,
                                        "heldout_delta_ap": cand_ap - base_ap,
                                        "n_triggered_real": int((mask & real_mask).sum()),
                                        "n_triggered_train_fake": int((mask & train_fake_mask).sum()),
                                        "n_triggered_heldout_fake": int((mask & heldout_mask).sum()),
                                    }
                                    row["train_passes_nonregression"] = (
                                        train_neg_auc == 0
                                        and train_neg_ap == 0
                                        and train_min_auc >= 0
                                        and train_min_ap >= 0
                                    )
                                    row["selection_score"] = (
                                        train_mean_auc
                                        + train_mean_ap
                                        + min(0.0, train_min_auc)
                                        + min(0.0, train_min_ap)
                                        - 0.01 * (train_neg_auc + train_neg_ap)
                                    )
                                    candidate_rows.append(row)
        candidates = pd.DataFrame(candidate_rows)
        rows.append(candidates)
        for mode, mode_candidates in candidates.groupby("apply_mode", sort=True):
            eligible = mode_candidates[mode_candidates["train_passes_nonregression"]].copy()
            if eligible.empty:
                selected = mode_candidates.sort_values(
                    ["selection_score", "train_mean_delta_auc", "train_mean_delta_ap"],
                    ascending=False,
                ).iloc[0]
                selected = selected.copy()
                selected["selected_from"] = "best_training_score_no_nonregression_pass"
            else:
                selected = eligible.sort_values(
                    ["train_mean_delta_auc", "train_mean_delta_ap", "train_min_delta_ap"],
                    ascending=False,
                ).iloc[0]
                selected = selected.copy()
                selected["selected_from"] = "best_training_nonregression_pass"
            selected_rows.append(pd.DataFrame([selected]))

    all_candidates = pd.concat(rows, ignore_index=True)
    selected = pd.concat(selected_rows, ignore_index=True)
    return all_candidates, selected


def _summary(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, feature_family, apply_mode), group in selected.groupby(["dataset", "feature_family", "apply_mode"], dropna=False):
        rows.append(
            {
                "dataset": dataset,
                "feature_family": feature_family,
                "apply_mode": apply_mode,
                "n_heldout_sources": int(len(group)),
                "mean_heldout_delta_auc": float(group["heldout_delta_auc"].mean()),
                "mean_heldout_delta_ap": float(group["heldout_delta_ap"].mean()),
                "min_heldout_delta_auc": float(group["heldout_delta_auc"].min()),
                "min_heldout_delta_ap": float(group["heldout_delta_ap"].min()),
                "n_negative_heldout_auc": int((group["heldout_delta_auc"] < 0).sum()),
                "n_negative_heldout_ap": int((group["heldout_delta_ap"] < 0).sum()),
                "mean_train_delta_auc": float(group["train_mean_delta_auc"].mean()),
                "mean_train_delta_ap": float(group["train_mean_delta_ap"].mean()),
                "total_triggered_real": int(group["n_triggered_real"].sum()),
                "total_triggered_train_fake": int(group["n_triggered_train_fake"].sum()),
                "total_triggered_heldout_fake": int(group["n_triggered_heldout_fake"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["apply_mode", "mean_heldout_delta_auc", "mean_heldout_delta_ap"], ascending=[True, False, False])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--feature-family", required=True)
    parser.add_argument("--frozen-csv", type=Path, required=True)
    parser.add_argument("--feature-csv", type=Path, required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--directions", default="high")
    parser.add_argument("--threshold-pools", default="all_train,real")
    parser.add_argument("--feature-quantiles", default="0.90,0.95")
    parser.add_argument("--apply-modes", default="all_samples,fake_only_oracle")
    parser.add_argument("--score-q", type=float, default=0.90)
    parser.add_argument("--real-cap-q", type=float, default=0.10)
    parser.add_argument("--strengths", default="0.25,0.50,1.00")
    parser.add_argument("--targets", default="cap_at_real_q10,base,min_base_split,min_base_universal")
    parser.add_argument("--output-candidates-csv", type=Path, required=True)
    parser.add_argument("--output-selected-csv", type=Path, required=True)
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    args = parser.parse_args()

    candidates, selected = run(args)
    summary = _summary(selected)
    args.output_candidates_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_selected_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.output_candidates_csv, index=False)
    selected.to_csv(args.output_selected_csv, index=False)
    summary.to_csv(args.output_summary_csv, index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved candidates -> {args.output_candidates_csv}")
    print(f"Saved selected -> {args.output_selected_csv}")
    print(f"Saved summary -> {args.output_summary_csv}")


if __name__ == "__main__":
    main()
