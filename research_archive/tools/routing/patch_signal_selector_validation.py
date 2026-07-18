#!/usr/bin/env python3
"""Validate source-level patch signal selection with a deterministic split.

This tool evaluates patch-signal selection, not a new global fusion weight. It
uses a calibration split to decide, per fake source, whether a candidate patch
signal should replace the raw patch-system score, then evaluates that decision
on a disjoint split.
"""

from __future__ import annotations

import argparse
import hashlib
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


def _split_value(dataset: str, source_model: str, filename: str, seed: int) -> float:
    key = f"{seed}|{dataset}|{source_model}|{filename}".encode("utf-8")
    digest = hashlib.sha1(key).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def _auc_ap(real_scores: pd.Series, fake_scores: pd.Series) -> tuple[float, float]:
    y = np.concatenate([np.ones(len(real_scores)), np.zeros(len(fake_scores))])
    s = np.concatenate([real_scores.to_numpy(float), fake_scores.to_numpy(float)])
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def _read_candidate(name: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in KEY_COLUMNS + ["raw_final_score"] if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    candidate_cols = [col for col in df.columns if col.startswith("frequency_add_") and col.endswith("_final_score")]
    if not candidate_cols:
        raise ValueError(f"{path} has no frequency_add_*_final_score candidate columns")
    out = df[KEY_COLUMNS + ["raw_final_score"] + candidate_cols].copy()
    for col in KEY_COLUMNS:
        out[col] = out[col].astype(str)
    rename = {col: f"{name}::{col}" for col in candidate_cols}
    return out.rename(columns=rename)


def _merge_specs(specs: list[tuple[str, Path]]) -> tuple[pd.DataFrame, list[str]]:
    merged: pd.DataFrame | None = None
    candidate_cols: list[str] = []
    for name, path in specs:
        df = _read_candidate(name, path)
        cols = [col for col in df.columns if "::" in col]
        candidate_cols.extend(cols)
        if merged is None:
            merged = df
        else:
            df = df.drop(columns=["raw_final_score"])
            merged = merged.merge(df, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if merged is None:
        raise ValueError("No candidate specs provided")
    return merged, candidate_cols


def _source_metric(df: pd.DataFrame, source: str, score_col: str, split_col: str, split_value: str) -> tuple[float, float, int, int]:
    part = df[df[split_col] == split_value]
    real = part[part["subset"].str.lower() == "real"]
    fake = part[(part["subset"].str.lower() != "real") & (part["source_model"] == source)]
    if len(real) < 2 or len(fake) < 2:
        raise ValueError(f"Not enough rows for {source}/{split_value}: real={len(real)} fake={len(fake)}")
    auc, ap = _auc_ap(real[score_col], fake[score_col])
    return auc, ap, int(len(real)), int(len(fake))


def _selection_key(
    objective: str,
    cand_auc: float,
    cand_ap: float,
    raw_auc: float,
    raw_ap: float,
) -> tuple[float, ...]:
    delta_auc = cand_auc - raw_auc
    delta_ap = cand_ap - raw_ap
    if objective == "auc_then_ap":
        return cand_auc, cand_ap
    if objective == "ap_then_auc":
        return cand_ap, cand_auc
    if objective == "delta_sum":
        return delta_auc + delta_ap, delta_ap, delta_auc
    if objective == "delta_min":
        return min(delta_auc, delta_ap), delta_ap + delta_auc
    raise ValueError(f"Unsupported selection objective: {objective}")


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specs = [_parse_spec(value) for value in args.candidate]
    df, candidate_cols = _merge_specs(specs)
    split_key = args.split_key or args.dataset
    split_values = [
        _split_value(split_key, row.source_model, row.filename, args.seed)
        for row in df[KEY_COLUMNS].itertuples(index=False)
    ]
    df["selector_split"] = np.where(np.asarray(split_values) < args.calibration_fraction, "calibration", "eval")

    fake_sources = sorted(df[df["subset"].str.lower() != "real"]["source_model"].unique())
    decision_rows = []
    eval_rows = []
    selected_score_frames = []

    for source in fake_sources:
        raw_cal_auc, raw_cal_ap, cal_real_n, cal_fake_n = _source_metric(
            df, source, "raw_final_score", "selector_split", "calibration"
        )
        best_col = "raw_final_score"
        best_auc = raw_cal_auc
        best_ap = raw_cal_ap
        best_key = _selection_key(args.selection_objective, raw_cal_auc, raw_cal_ap, raw_cal_auc, raw_cal_ap)
        candidate_audits = []
        for col in candidate_cols:
            cand_auc, cand_ap, _, _ = _source_metric(df, source, col, "selector_split", "calibration")
            delta_auc = cand_auc - raw_cal_auc
            delta_ap = cand_ap - raw_cal_ap
            delta_sum = delta_auc + delta_ap
            candidate_audits.append((col, cand_auc, cand_ap, delta_auc, delta_ap))
            if (
                delta_auc >= args.min_cal_delta_auc
                and delta_ap >= args.min_cal_delta_ap
                and delta_sum >= args.min_cal_delta_sum
            ):
                key = _selection_key(args.selection_objective, cand_auc, cand_ap, raw_cal_auc, raw_cal_ap)
                if key > best_key:
                    best_key = key
                    best_col = col
                    best_auc = cand_auc
                    best_ap = cand_ap

        raw_eval_auc, raw_eval_ap, eval_real_n, eval_fake_n = _source_metric(
            df, source, "raw_final_score", "selector_split", "eval"
        )
        sel_eval_auc, sel_eval_ap, _, _ = _source_metric(df, source, best_col, "selector_split", "eval")
        eval_rows.append(
            {
                "dataset": args.dataset,
                "split_key": split_key,
                "source_model": source,
                "selected_col": best_col,
                "selected_family": best_col.split("::", 1)[0] if "::" in best_col else "raw",
                "selected_weight": best_col.rsplit("_final_score", 1)[0].split("frequency_add_", 1)[-1]
                if "frequency_add_" in best_col
                else "raw",
                "cal_raw_auc": raw_cal_auc,
                "cal_raw_ap": raw_cal_ap,
                "cal_selected_auc": best_auc,
                "cal_selected_ap": best_ap,
                "cal_delta_auc": best_auc - raw_cal_auc,
                "cal_delta_ap": best_ap - raw_cal_ap,
                "eval_raw_auc": raw_eval_auc,
                "eval_raw_ap": raw_eval_ap,
                "eval_selected_auc": sel_eval_auc,
                "eval_selected_ap": sel_eval_ap,
                "eval_delta_auc": sel_eval_auc - raw_eval_auc,
                "eval_delta_ap": sel_eval_ap - raw_eval_ap,
                "cal_n_real": cal_real_n,
                "cal_n_fake": cal_fake_n,
                "eval_n_real": eval_real_n,
                "eval_n_fake": eval_fake_n,
            }
        )
        for col, cand_auc, cand_ap, delta_auc, delta_ap in candidate_audits:
            decision_rows.append(
                {
                    "dataset": args.dataset,
                    "split_key": split_key,
                    "source_model": source,
                    "candidate_col": col,
                    "candidate_family": col.split("::", 1)[0],
                    "cal_raw_auc": raw_cal_auc,
                    "cal_raw_ap": raw_cal_ap,
                    "cal_candidate_auc": cand_auc,
                    "cal_candidate_ap": cand_ap,
                    "cal_delta_auc": delta_auc,
                    "cal_delta_ap": delta_ap,
                    "cal_delta_sum": delta_auc + delta_ap,
                    "selected": col == best_col,
                }
            )

        eval_part = df[df["selector_split"] == "eval"].copy()
        real_part = eval_part[eval_part["subset"].str.lower() == "real"][KEY_COLUMNS].copy()
        fake_part = eval_part[
            (eval_part["subset"].str.lower() != "real") & (eval_part["source_model"] == source)
        ][KEY_COLUMNS].copy()
        real_part["selected_score"] = eval_part.loc[real_part.index, best_col].to_numpy(float)
        fake_part["selected_score"] = eval_part.loc[fake_part.index, best_col].to_numpy(float)
        real_part["eval_source"] = source
        fake_part["eval_source"] = source
        selected_score_frames.append(pd.concat([real_part, fake_part], ignore_index=True))

    per_source = pd.DataFrame(eval_rows)
    summary = pd.DataFrame(
        [
            {
                "dataset": args.dataset,
                "split_key": split_key,
                "n_sources": int(len(per_source)),
                "avg_raw_auc": float(per_source["eval_raw_auc"].mean()),
                "avg_raw_ap": float(per_source["eval_raw_ap"].mean()),
                "avg_selected_auc": float(per_source["eval_selected_auc"].mean()),
                "avg_selected_ap": float(per_source["eval_selected_ap"].mean()),
                "avg_delta_auc": float(per_source["eval_delta_auc"].mean()),
                "avg_delta_ap": float(per_source["eval_delta_ap"].mean()),
                "min_delta_auc": float(per_source["eval_delta_auc"].min()),
                "min_delta_ap": float(per_source["eval_delta_ap"].min()),
                "improved_auc_sources": int((per_source["eval_delta_auc"] > 0).sum()),
                "improved_ap_sources": int((per_source["eval_delta_ap"] > 0).sum()),
                "selected_non_raw_sources": int((per_source["selected_family"] != "raw").sum()),
                "calibration_fraction": args.calibration_fraction,
                "min_cal_delta_auc": args.min_cal_delta_auc,
                "min_cal_delta_ap": args.min_cal_delta_ap,
                "min_cal_delta_sum": args.min_cal_delta_sum,
                "selection_objective": args.selection_objective,
            }
        ]
    )
    selected_scores = pd.concat(selected_score_frames, ignore_index=True) if selected_score_frames else pd.DataFrame()
    decisions = pd.DataFrame(decision_rows)
    return summary, per_source, decisions, selected_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-key", help="Stable key for deterministic calibration/eval split; defaults to dataset.")
    parser.add_argument("--candidate", action="append", required=True, help="name=full_system_scores.csv")
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    parser.add_argument("--output-per-source-csv", type=Path, required=True)
    parser.add_argument("--output-decisions-csv", type=Path, required=True)
    parser.add_argument("--output-selected-scores-csv", type=Path)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-cal-delta-auc", type=float, default=0.0)
    parser.add_argument("--min-cal-delta-ap", type=float, default=0.0)
    parser.add_argument("--min-cal-delta-sum", type=float, default=-1e9)
    parser.add_argument(
        "--selection-objective",
        choices=["auc_then_ap", "ap_then_auc", "delta_sum", "delta_min"],
        default="auc_then_ap",
    )
    args = parser.parse_args()

    summary, per_source, decisions, selected_scores = run(args)
    args.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary_csv, index=False)
    per_source.to_csv(args.output_per_source_csv, index=False)
    decisions.to_csv(args.output_decisions_csv, index=False)
    if args.output_selected_scores_csv:
        selected_scores.to_csv(args.output_selected_scores_csv, index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
