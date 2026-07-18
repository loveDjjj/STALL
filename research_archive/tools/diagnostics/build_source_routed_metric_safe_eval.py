#!/usr/bin/env python3
"""Build metric-safe source-routed eval files from frozen and candidate scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


KEY_COLUMNS = ["subset", "source_model", "filename"]


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in KEY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    for col in KEY_COLUMNS:
        df[col] = df[col].astype(str)
    return df


def _metric(real_scores: pd.Series, fake_scores: pd.Series) -> tuple[float, float]:
    y = np.r_[np.ones(len(real_scores), dtype=int), np.zeros(len(fake_scores), dtype=int)]
    s = np.r_[real_scores.to_numpy(float), fake_scores.to_numpy(float)]
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def _parse_sources(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frozen = _read(args.frozen_final_scores_csv)
    if args.frozen_score_col not in frozen.columns:
        raise ValueError(f"{args.frozen_final_scores_csv} missing {args.frozen_score_col}")
    frozen = frozen[[*KEY_COLUMNS, args.frozen_score_col]].rename(columns={args.frozen_score_col: "raw_score"})

    candidate = _read(args.candidate_scores_csv)
    if args.candidate_score_col not in candidate.columns:
        raise ValueError(f"{args.candidate_scores_csv} missing {args.candidate_score_col}")
    candidate = candidate[[*KEY_COLUMNS, args.candidate_score_col]].rename(
        columns={args.candidate_score_col: "candidate_score"}
    )

    promoted = args.promoted_sources
    real_frozen = frozen[frozen["subset"].str.lower() == "real"]
    fake_frozen = frozen[frozen["subset"].str.lower() != "real"]
    real_candidate = candidate[candidate["subset"].str.lower() == "real"]

    rows = []
    eval_frames = []
    for source, fake_raw in fake_frozen.groupby("source_model", sort=True):
        use_candidate = source in promoted
        raw_auc, raw_ap = _metric(real_frozen["raw_score"], fake_raw["raw_score"])
        if use_candidate:
            fake_candidate = candidate[
                (candidate["subset"].str.lower() != "real") & (candidate["source_model"] == source)
            ]
            if len(real_candidate) == 0 or len(fake_candidate) == 0:
                raise ValueError(f"Missing candidate coverage for promoted source {source}")
            selected_auc, selected_ap = _metric(real_candidate["candidate_score"], fake_candidate["candidate_score"])
            real_eval = real_candidate[[*KEY_COLUMNS, "candidate_score"]].rename(
                columns={"candidate_score": "selected_score"}
            )
            fake_eval = fake_candidate[[*KEY_COLUMNS, "candidate_score"]].rename(
                columns={"candidate_score": "selected_score"}
            )
            family = args.candidate_family
        else:
            selected_auc, selected_ap = raw_auc, raw_ap
            real_eval = real_frozen[[*KEY_COLUMNS, "raw_score"]].rename(columns={"raw_score": "selected_score"})
            fake_eval = fake_raw[[*KEY_COLUMNS, "raw_score"]].rename(columns={"raw_score": "selected_score"})
            family = "raw_frozen"

        eval_df = pd.concat([real_eval, fake_eval], ignore_index=True)
        eval_df = eval_df.merge(frozen[[*KEY_COLUMNS, "raw_score"]], on=KEY_COLUMNS, how="left", validate="one_to_one")
        eval_df["eval_source_model"] = source
        eval_df["selected_family"] = family
        eval_df["label"] = (eval_df["subset"].str.lower() == "real").astype(int)
        eval_frames.append(eval_df)

        rows.append(
            {
                "dataset": args.dataset,
                "source_model": source,
                "n_real": int(len(real_candidate if use_candidate else real_frozen)),
                "n_fake": int(len(fake_candidate if use_candidate else fake_raw)),
                "selected_family": family,
                "raw_auc": raw_auc,
                "raw_ap": raw_ap,
                "selected_auc": selected_auc,
                "selected_ap": selected_ap,
                "delta_auc": selected_auc - raw_auc,
                "delta_ap": selected_ap - raw_ap,
            }
        )

    per_source = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "dataset": args.dataset,
                "n_sources": int(len(per_source)),
                "selected_non_raw_sources": int((per_source["selected_family"] != "raw_frozen").sum()),
                "raw_avg_auc": float(per_source["raw_auc"].mean()),
                "raw_avg_ap": float(per_source["raw_ap"].mean()),
                "selected_avg_auc": float(per_source["selected_auc"].mean()),
                "selected_avg_ap": float(per_source["selected_ap"].mean()),
                "avg_delta_auc": float(per_source["delta_auc"].mean()),
                "avg_delta_ap": float(per_source["delta_ap"].mean()),
                "min_delta_auc": float(per_source["delta_auc"].min()),
                "min_delta_ap": float(per_source["delta_ap"].min()),
            }
        ]
    )
    eval_scores = pd.concat(eval_frames, ignore_index=True) if eval_frames else pd.DataFrame()
    return summary, per_source, eval_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--frozen-final-scores-csv", type=Path, required=True)
    parser.add_argument("--candidate-scores-csv", type=Path, required=True)
    parser.add_argument("--promoted-sources", type=_parse_sources, required=True)
    parser.add_argument("--candidate-score-col", required=True)
    parser.add_argument("--frozen-score-col", default="final_score")
    parser.add_argument("--candidate-family", default="fb_cycle_splitminus")
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    parser.add_argument("--output-per-source-csv", type=Path, required=True)
    parser.add_argument("--output-eval-scores-csv", type=Path, required=True)
    args = parser.parse_args()

    summary, per_source, eval_scores = run(args)
    args.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary_csv, index=False)
    per_source.to_csv(args.output_per_source_csv, index=False)
    eval_scores.to_csv(args.output_eval_scores_csv, index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved per-source -> {args.output_per_source_csv}")


if __name__ == "__main__":
    main()
