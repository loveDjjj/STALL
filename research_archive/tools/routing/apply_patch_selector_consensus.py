#!/usr/bin/env python3
"""Apply multi-seed source consensus decisions to full-system score CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


KEY_COLUMNS = ["subset", "source_model", "filename"]


def _parse_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split("=", 1)
    return name.strip(), Path(path.strip())


def _read_candidate(name: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in KEY_COLUMNS + ["raw_final_score"] if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    candidate_cols = [col for col in df.columns if col.startswith("frequency_add_") and col.endswith("_final_score")]
    if not candidate_cols:
        raise ValueError(f"{path} has no frequency_add_*_final_score columns")
    out = df[KEY_COLUMNS + ["raw_final_score"] + candidate_cols].copy()
    for col in KEY_COLUMNS:
        out[col] = out[col].astype(str)
    return out.rename(columns={col: f"{name}::{col}" for col in candidate_cols})


def _merge_candidates(specs: list[tuple[str, Path]]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for name, path in specs:
        df = _read_candidate(name, path)
        if merged is None:
            merged = df
        else:
            df = df.drop(columns=["raw_final_score"])
            merged = merged.merge(df, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if merged is None:
        raise ValueError("No candidate score CSVs provided")
    return merged


def _auc_ap(real_scores: pd.Series, fake_scores: pd.Series) -> tuple[float, float]:
    y = np.concatenate([np.ones(len(real_scores)), np.zeros(len(fake_scores))])
    scores = np.concatenate([real_scores.to_numpy(float), fake_scores.to_numpy(float)])
    return float(roc_auc_score(y, scores)), float(average_precision_score(y, scores))


def _choose_consensus_columns(
    source_summary: pd.DataFrame,
    long_df: pd.DataFrame,
    group: str,
    allowed_statuses: set[str],
) -> dict[str, str]:
    group_summary = source_summary[source_summary["stability_group"].astype(str) == group].copy()
    group_long = long_df[long_df["stability_group"].astype(str) == group].copy()
    if group_summary.empty:
        raise ValueError(f"No rows for stability group {group!r} in source summary")
    if group_long.empty:
        raise ValueError(f"No rows for stability group {group!r} in long CSV")

    decisions: dict[str, str] = {}
    for row in group_summary.itertuples(index=False):
        source = str(row.source_model)
        status = str(row.status)
        if status not in allowed_statuses:
            decisions[source] = "raw_final_score"
            continue
        selected = group_long[
            (group_long["source_model"].astype(str) == source)
            & (group_long["selected_family"].astype(str) != "raw")
            & (group_long["negative_eval"].astype(bool) == False)
        ].copy()
        if selected.empty:
            decisions[source] = "raw_final_score"
            continue
        counts = (
            selected.groupby("selected_col", as_index=False)
            .agg(
                n=("seed", "nunique"),
                mean_eval_delta_auc=("eval_delta_auc", "mean"),
                mean_eval_delta_ap=("eval_delta_ap", "mean"),
                min_eval_delta_auc=("eval_delta_auc", "min"),
                min_eval_delta_ap=("eval_delta_ap", "min"),
            )
            .sort_values(
                ["n", "min_eval_delta_ap", "min_eval_delta_auc", "mean_eval_delta_ap", "mean_eval_delta_auc"],
                ascending=[False, False, False, False, False],
            )
        )
        decisions[source] = str(counts.iloc[0]["selected_col"])
    return decisions


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    allowed_statuses = {value.strip() for value in args.include_statuses.split(",") if value.strip()}
    source_summary = pd.read_csv(args.source_summary_csv)
    long_df = pd.read_csv(args.long_csv)
    decisions = _choose_consensus_columns(source_summary, long_df, args.group, allowed_statuses)
    scores = _merge_candidates([_parse_spec(value) for value in args.candidate])

    missing_decision_sources = sorted(
        set(scores[scores["subset"].str.lower() != "real"]["source_model"].astype(str).unique()) - set(decisions)
    )
    for source in missing_decision_sources:
        decisions[source] = "raw_final_score"

    scores["consensus_selected_col"] = "raw_final_score"
    scores["consensus_selected_score"] = scores["raw_final_score"].astype(float)
    fake_mask = scores["subset"].str.lower() != "real"
    for source, col in decisions.items():
        if col != "raw_final_score" and col not in scores.columns:
            raise ValueError(f"Consensus selected column {col!r} for {source} is not present in merged scores")
        source_fake_mask = fake_mask & (scores["source_model"].astype(str) == source)
        source_real_mask = scores["subset"].str.lower() == "real"
        apply_mask = source_fake_mask | source_real_mask
        # For each source-vs-real evaluation, real rows use the same selected
        # candidate as the fake source, matching patch_signal_selector_validation.
        if col != "raw_final_score":
            scores.loc[source_fake_mask, "consensus_selected_score"] = scores.loc[source_fake_mask, col].astype(float)
        scores.loc[source_fake_mask, "consensus_selected_col"] = col

    per_rows = []
    eval_score_frames = []
    real_all = scores[scores["subset"].str.lower() == "real"]
    for source in sorted(scores[fake_mask]["source_model"].astype(str).unique()):
        fake = scores[fake_mask & (scores["source_model"].astype(str) == source)]
        col = decisions.get(source, "raw_final_score")
        raw_auc, raw_ap = _auc_ap(real_all["raw_final_score"], fake["raw_final_score"])
        if col == "raw_final_score":
            real_selected = real_all["raw_final_score"]
            fake_selected = fake["raw_final_score"]
        else:
            real_selected = real_all[col]
            fake_selected = fake[col]
        selected_auc, selected_ap = _auc_ap(real_selected, fake_selected)

        for subset_name, frame, selected_scores in (
            ("real", real_all, real_selected),
            ("fake", fake, fake_selected),
        ):
            eval_part = frame[KEY_COLUMNS].copy()
            eval_part["eval_source_model"] = source
            eval_part["eval_subset"] = subset_name
            eval_part["label"] = 1 if subset_name == "real" else 0
            eval_part["raw_score"] = frame["raw_final_score"].astype(float).to_numpy()
            eval_part["consensus_selected_score"] = selected_scores.astype(float).to_numpy()
            eval_part["consensus_selected_col"] = col
            eval_score_frames.append(eval_part)

        per_rows.append(
            {
                "group": args.group,
                "source_model": source,
                "consensus_selected_col": col,
                "consensus_selected_family": col.split("::", 1)[0] if "::" in col else "raw",
                "consensus_selected_weight": col.rsplit("_final_score", 1)[0].split("frequency_add_", 1)[-1]
                if "frequency_add_" in col
                else "raw",
                "raw_auc": raw_auc,
                "raw_ap": raw_ap,
                "selected_auc": selected_auc,
                "selected_ap": selected_ap,
                "delta_auc": selected_auc - raw_auc,
                "delta_ap": selected_ap - raw_ap,
                "n_real": int(len(real_all)),
                "n_fake": int(len(fake)),
            }
        )

    per_source = pd.DataFrame(per_rows)
    summary = pd.DataFrame(
        [
            {
                "group": args.group,
                "n_sources": int(len(per_source)),
                "avg_raw_auc": float(per_source["raw_auc"].mean()),
                "avg_raw_ap": float(per_source["raw_ap"].mean()),
                "avg_selected_auc": float(per_source["selected_auc"].mean()),
                "avg_selected_ap": float(per_source["selected_ap"].mean()),
                "avg_delta_auc": float(per_source["delta_auc"].mean()),
                "avg_delta_ap": float(per_source["delta_ap"].mean()),
                "min_delta_auc": float(per_source["delta_auc"].min()),
                "min_delta_ap": float(per_source["delta_ap"].min()),
                "selected_non_raw_sources": int((per_source["consensus_selected_family"] != "raw").sum()),
                "include_statuses": args.include_statuses,
            }
        ]
    )
    eval_scores = pd.concat(eval_score_frames, ignore_index=True) if eval_score_frames else pd.DataFrame()
    return summary, per_source, scores, eval_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True)
    parser.add_argument("--source-summary-csv", type=Path, required=True)
    parser.add_argument("--long-csv", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True, help="name=full_system_scores.csv")
    parser.add_argument("--include-statuses", default="promote")
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    parser.add_argument("--output-per-source-csv", type=Path, required=True)
    parser.add_argument("--output-scores-csv", type=Path, required=True)
    parser.add_argument("--output-eval-scores-csv", type=Path)
    args = parser.parse_args()

    summary, per_source, scores, eval_scores = run(args)
    args.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary_csv, index=False)
    per_source.to_csv(args.output_per_source_csv, index=False)
    scores.to_csv(args.output_scores_csv, index=False)
    if args.output_eval_scores_csv is not None:
        args.output_eval_scores_csv.parent.mkdir(parents=True, exist_ok=True)
        eval_scores.to_csv(args.output_eval_scores_csv, index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
