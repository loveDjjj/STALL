#!/usr/bin/env python3
"""Probe whether auxiliary features separate residual hard fake tails."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


KEY_COLUMNS = ["subset", "source_model", "filename"]


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in KEY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    for col in KEY_COLUMNS:
        df[col] = df[col].astype(str)
    return df


def _is_real(df: pd.DataFrame) -> pd.Series:
    return df["subset"].astype(str).str.lower() == "real"


def _numeric_feature_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    cols = []
    for col in df.columns:
        if col in exclude or col in KEY_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            vals = df[col].dropna()
            if len(vals) and vals.nunique() > 1:
                cols.append(col)
    return cols


def _safe_metric(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def _probe_subset(df: pd.DataFrame, feature_cols: list[str], label_col: str, subset_name: str) -> pd.DataFrame:
    rows = []
    sub = df[df[label_col].notna()].copy()
    y = sub[label_col].to_numpy(int)
    for col in feature_cols:
        values = sub[col].to_numpy(float)
        good = np.isfinite(values)
        if good.sum() < 10 or len(np.unique(y[good])) < 2:
            continue
        auc, ap = _safe_metric(y[good], values[good])
        auc_neg, ap_neg = _safe_metric(y[good], -values[good])
        if np.isnan(auc):
            continue
        if auc_neg > auc:
            best_auc = auc_neg
            best_ap = ap_neg
            direction = "low"
        else:
            best_auc = auc
            best_ap = ap
            direction = "high"
        rows.append(
            {
                "probe": subset_name,
                "feature": col,
                "direction_for_hard_fake": direction,
                "auc": best_auc,
                "ap": best_ap,
                "auc_high": auc,
                "auc_low": auc_neg,
                "n": int(good.sum()),
                "n_positive": int(y[good].sum()),
                "n_negative": int((1 - y[good]).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["auc", "ap", "feature"], ascending=[False, False, True])


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    frozen = _read(args.frozen_csv)
    if "final_score" not in frozen.columns:
        raise ValueError(f"{args.frozen_csv} missing final_score")
    frozen = frozen[[*KEY_COLUMNS, "final_score"]].rename(columns={"final_score": "frozen_final_score"})
    features = _read(args.feature_csv)
    df = frozen.merge(features, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if df.empty:
        raise ValueError("No overlapping rows between frozen scores and feature CSV")

    real_mask = _is_real(df)
    fake_mask = ~real_mask
    real_q = float(np.quantile(df.loc[real_mask, "frozen_final_score"], args.real_low_q))
    fake_q = float(np.quantile(df.loc[fake_mask, "frozen_final_score"], args.fake_high_q))

    df["tail_label_all"] = np.nan
    df.loc[real_mask & (df["frozen_final_score"] <= real_q), "tail_label_all"] = 0
    df.loc[fake_mask & (df["frozen_final_score"] >= fake_q), "tail_label_all"] = 1

    rows = []
    for source, group in df[fake_mask].groupby("source_model", sort=True):
        source_q = float(np.quantile(group["frozen_final_score"], args.fake_high_q))
        idx = fake_mask & (df["source_model"] == source) & (df["frozen_final_score"] >= source_q)
        rows.append((source, idx))
    df["tail_label_source_balanced"] = np.nan
    df.loc[real_mask & (df["frozen_final_score"] <= real_q), "tail_label_source_balanced"] = 0
    for _, idx in rows:
        df.loc[idx, "tail_label_source_balanced"] = 1

    exclude = {
        "frozen_final_score",
        "tail_label_all",
        "tail_label_source_balanced",
        "patch_temp_mode",
        "patch_region_size",
    }
    feature_cols = _numeric_feature_columns(df, exclude)
    probes = [
        _probe_subset(df, feature_cols, "tail_label_all", "global_fake_high_vs_real_low"),
        _probe_subset(df, feature_cols, "tail_label_source_balanced", "source_balanced_fake_high_vs_real_low"),
    ]
    result = pd.concat(probes, ignore_index=True)

    summary = pd.DataFrame(
        [
            {
                "dataset": args.dataset,
                "feature_family": args.feature_family,
                "n_frozen_rows": int(len(frozen)),
                "n_feature_rows": int(len(features)),
                "n_overlap_rows": int(len(df)),
                "n_overlap_real": int(real_mask.sum()),
                "n_overlap_fake": int(fake_mask.sum()),
                "real_low_q": args.real_low_q,
                "real_low_threshold": real_q,
                "fake_high_q": args.fake_high_q,
                "fake_high_threshold": fake_q,
                "n_global_tail_positive": int((df["tail_label_all"] == 1).sum()),
                "n_global_tail_negative": int((df["tail_label_all"] == 0).sum()),
                "n_source_balanced_tail_positive": int((df["tail_label_source_balanced"] == 1).sum()),
                "n_source_balanced_tail_negative": int((df["tail_label_source_balanced"] == 0).sum()),
                "best_global_feature": (
                    str(result[result["probe"] == "global_fake_high_vs_real_low"].iloc[0]["feature"])
                    if len(result[result["probe"] == "global_fake_high_vs_real_low"])
                    else ""
                ),
                "best_global_auc": (
                    float(result[result["probe"] == "global_fake_high_vs_real_low"].iloc[0]["auc"])
                    if len(result[result["probe"] == "global_fake_high_vs_real_low"])
                    else np.nan
                ),
                "best_source_balanced_feature": (
                    str(result[result["probe"] == "source_balanced_fake_high_vs_real_low"].iloc[0]["feature"])
                    if len(result[result["probe"] == "source_balanced_fake_high_vs_real_low"])
                    else ""
                ),
                "best_source_balanced_auc": (
                    float(result[result["probe"] == "source_balanced_fake_high_vs_real_low"].iloc[0]["auc"])
                    if len(result[result["probe"] == "source_balanced_fake_high_vs_real_low"])
                    else np.nan
                ),
            }
        ]
    )
    result.insert(0, "feature_family", args.feature_family)
    result.insert(0, "dataset", args.dataset)
    return result, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--feature-family", required=True)
    parser.add_argument("--frozen-csv", type=Path, required=True)
    parser.add_argument("--feature-csv", type=Path, required=True)
    parser.add_argument("--real-low-q", type=float, default=0.10)
    parser.add_argument("--fake-high-q", type=float, default=0.90)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    args = parser.parse_args()

    result, summary = run(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(result.head(20).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved probe -> {args.output_csv}")
    print(f"Saved summary -> {args.summary_csv}")


if __name__ == "__main__":
    main()
