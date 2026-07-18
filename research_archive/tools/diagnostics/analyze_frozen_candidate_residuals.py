#!/usr/bin/env python3
"""Residual and headroom diagnostics for the frozen sample fallback candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


KEY_COLUMNS = ["subset", "source_model", "filename"]
SCORE_COLUMNS = ["base_score", "universal_score", "split_score", "final_score"]


def _is_real(df: pd.DataFrame) -> pd.Series:
    return df["subset"].astype(str).str.lower() == "real"


def _auc_ap(real_scores: pd.Series, fake_scores: pd.Series) -> tuple[float, float]:
    y_true = [1] * len(real_scores) + [0] * len(fake_scores)
    scores = [*real_scores.astype(float), *fake_scores.astype(float)]
    return float(roc_auc_score(y_true, scores)), float(average_precision_score(y_true, scores))


def _tail_stats(fake_scores: pd.Series, real_scores: pd.Series) -> dict[str, float]:
    fake = fake_scores.astype(float).to_numpy()
    real = real_scores.astype(float).to_numpy()
    return {
        "fake_score_q90": float(np.quantile(fake, 0.90)),
        "fake_score_q95": float(np.quantile(fake, 0.95)),
        "fake_score_max": float(np.max(fake)),
        "real_score_q05": float(np.quantile(real, 0.05)),
        "real_score_q10": float(np.quantile(real, 0.10)),
        "hard_fake_rate_above_real_q10": float((fake >= np.quantile(real, 0.10)).mean()),
        "hard_fake_rate_above_real_q05": float((fake >= np.quantile(real, 0.05)).mean()),
    }


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in [*KEY_COLUMNS, *SCORE_COLUMNS] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    for col in KEY_COLUMNS:
        df[col] = df[col].astype(str)
    for col in SCORE_COLUMNS:
        df[col] = df[col].astype(float)
    return df


def _source_rows(dataset: str, df: pd.DataFrame) -> pd.DataFrame:
    real = df[_is_real(df)]
    fake = df[~_is_real(df)]
    rows: list[dict[str, object]] = []

    for source, group in fake.groupby("source_model", sort=True):
        metrics: dict[str, tuple[float, float]] = {}
        for score_col in SCORE_COLUMNS:
            auc, ap = _auc_ap(real[score_col], group[score_col])
            metrics[score_col] = (auc, ap)

        primitive_scores = {
            "base": metrics["base_score"],
            "universal": metrics["universal_score"],
            "split": metrics["split_score"],
        }
        best_primitive_action = max(
            primitive_scores,
            key=lambda k: (primitive_scores[k][0] + primitive_scores[k][1], primitive_scores[k][0]),
        )
        primitive_auc, primitive_ap = primitive_scores[best_primitive_action]
        final_auc, final_ap = metrics["final_score"]

        row: dict[str, object] = {
            "dataset": dataset,
            "source_model": source,
            "n_fake": int(len(group)),
            "base_auc": metrics["base_score"][0],
            "base_ap": metrics["base_score"][1],
            "universal_auc": metrics["universal_score"][0],
            "universal_ap": metrics["universal_score"][1],
            "split_auc": metrics["split_score"][0],
            "split_ap": metrics["split_score"][1],
            "final_auc": final_auc,
            "final_ap": final_ap,
            "delta_final_vs_base_auc": final_auc - metrics["base_score"][0],
            "delta_final_vs_base_ap": final_ap - metrics["base_score"][1],
            "delta_final_vs_universal_auc": final_auc - metrics["universal_score"][0],
            "delta_final_vs_universal_ap": final_ap - metrics["universal_score"][1],
            "best_primitive_action": best_primitive_action,
            "best_primitive_auc": primitive_auc,
            "best_primitive_ap": primitive_ap,
            "primitive_headroom_auc": primitive_auc - final_auc,
            "primitive_headroom_ap": primitive_ap - final_ap,
            "primitive_headroom_mean": ((primitive_auc - final_auc) + (primitive_ap - final_ap)) / 2.0,
            "final_auc_gap_to_perfect": 1.0 - final_auc,
            "final_ap_gap_to_perfect": 1.0 - final_ap,
            "use_split_rate": float(group["use_split"].astype(float).mean()) if "use_split" in group.columns else np.nan,
            "sample_gate_mean": float(group["sample_gate"].astype(float).mean()) if "sample_gate" in group.columns else np.nan,
            "split_minus_universal_mean": (
                float(group["split_minus_universal"].astype(float).mean())
                if "split_minus_universal" in group.columns
                else np.nan
            ),
        }
        row.update(_tail_stats(group["final_score"], real["final_score"]))
        rows.append(row)

    out = pd.DataFrame(rows)
    out["priority_score"] = (
        out["primitive_headroom_mean"].clip(lower=0)
        + 0.5 * out["final_auc_gap_to_perfect"]
        + 0.5 * out["final_ap_gap_to_perfect"]
        + 0.1 * out["hard_fake_rate_above_real_q10"]
    )
    return out.sort_values(["priority_score", "primitive_headroom_mean", "source_model"], ascending=[False, False, True])


def _summary(source_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in source_df.groupby("dataset", sort=True):
        rows.append(_summarize_group(dataset, group))
    rows.append(_summarize_group("ALL", source_df))
    return pd.DataFrame(rows)


def _summarize_group(dataset: str, group: pd.DataFrame) -> dict[str, object]:
    worst_auc = group.sort_values(["final_auc", "final_ap", "source_model"]).iloc[0]
    worst_ap = group.sort_values(["final_ap", "final_auc", "source_model"]).iloc[0]
    top_priority = group.iloc[0]
    return {
        "dataset": dataset,
        "n_sources": int(len(group)),
        "mean_final_auc": float(group["final_auc"].mean()),
        "mean_final_ap": float(group["final_ap"].mean()),
        "min_final_auc": float(group["final_auc"].min()),
        "min_final_ap": float(group["final_ap"].min()),
        "mean_primitive_headroom_auc": float(group["primitive_headroom_auc"].mean()),
        "mean_primitive_headroom_ap": float(group["primitive_headroom_ap"].mean()),
        "max_primitive_headroom_auc": float(group["primitive_headroom_auc"].max()),
        "max_primitive_headroom_ap": float(group["primitive_headroom_ap"].max()),
        "n_sources_with_primitive_headroom": int((group["primitive_headroom_mean"] > 1e-12).sum()),
        "worst_auc_source": str(worst_auc["source_model"]),
        "worst_ap_source": str(worst_ap["source_model"]),
        "top_priority_source": str(top_priority["source_model"]),
        "top_priority_score": float(top_priority["priority_score"]),
    }


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for item in args.input:
        dataset, path_str = item.split("=", 1)
        frames.append(_source_rows(dataset, _load(Path(path_str))))
    source_df = pd.concat(frames, ignore_index=True)
    source_df = source_df.sort_values(["priority_score", "primitive_headroom_mean", "dataset"], ascending=[False, False, True])
    return source_df, _summary(source_df)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Dataset/file pairs such as genvideo=results/...final_scores.csv",
    )
    parser.add_argument("--output-source-csv", type=Path, required=True)
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    args = parser.parse_args()

    source_df, summary = run(args)
    args.output_source_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    source_df.to_csv(args.output_source_csv, index=False)
    summary.to_csv(args.output_summary_csv, index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved source residuals -> {args.output_source_csv}")
    print(f"Saved residual summary -> {args.output_summary_csv}")


if __name__ == "__main__":
    main()
