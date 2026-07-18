#!/usr/bin/env python3
"""Summarize source-level selector stability across multiple split seeds."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "source_model",
    "selected_family",
    "selected_weight",
    "eval_delta_auc",
    "eval_delta_ap",
    "cal_delta_auc",
    "cal_delta_ap",
]


def _parse_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split("=", 1)
    return name.strip(), Path(path.strip())


def _infer_seed(path: Path, default_seed: int) -> int:
    match = re.search(r"(?:^|_)seed(\d+)(?:_|$)", path.stem)
    if match:
        return int(match.group(1))
    return default_seed


def _read_per_source(group: str, path: Path, default_seed: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    out = df.copy()
    out["stability_group"] = group
    out["seed"] = _infer_seed(path, default_seed)
    out["selected_nonraw"] = out["selected_family"].astype(str) != "raw"
    out["negative_eval"] = (out["eval_delta_auc"].astype(float) < 0.0) | (
        out["eval_delta_ap"].astype(float) < 0.0
    )
    return out


def _status(row: pd.Series, strong_min_seeds: int, exploratory_min_seeds: int) -> str:
    if row["negative_selected_seeds"] > 0:
        return "reject"
    if row["selected_seeds"] >= strong_min_seeds:
        return "promote"
    if row["selected_seeds"] >= exploratory_min_seeds:
        return "exploratory"
    return "reject"


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    for spec in args.per_source:
        group, path = _parse_spec(spec)
        frames.append(_read_per_source(group, path, args.default_seed))
    if not frames:
        raise ValueError("No per-source CSVs provided")

    long_df = pd.concat(frames, ignore_index=True)
    all_seed_counts = (
        long_df.groupby(["stability_group", "source_model"], as_index=False)["seed"]
        .nunique()
        .rename(columns={"seed": "n_seeds"})
    )

    source_rows = []
    for (group, source), part in long_df.groupby(["stability_group", "source_model"], sort=True):
        selected = part[part["selected_nonraw"]].copy()
        selected_seeds = int(selected["seed"].nunique())
        n_seeds = int(part["seed"].nunique())
        negative_selected = selected[selected["negative_eval"]]
        if selected.empty:
            selected_families = "raw"
            selected_weights = "raw"
            mean_selected_eval_delta_auc = 0.0
            mean_selected_eval_delta_ap = 0.0
            min_selected_eval_delta_auc = 0.0
            min_selected_eval_delta_ap = 0.0
        else:
            selected_families = ";".join(
                f"{fam}:{count}"
                for fam, count in selected["selected_family"].astype(str).value_counts().sort_index().items()
            )
            selected_weights = ";".join(
                f"{weight}:{count}"
                for weight, count in selected["selected_weight"].astype(str).value_counts().sort_index().items()
            )
            mean_selected_eval_delta_auc = float(selected["eval_delta_auc"].mean())
            mean_selected_eval_delta_ap = float(selected["eval_delta_ap"].mean())
            min_selected_eval_delta_auc = float(selected["eval_delta_auc"].min())
            min_selected_eval_delta_ap = float(selected["eval_delta_ap"].min())

        row = {
            "stability_group": group,
            "source_model": source,
            "n_seeds": n_seeds,
            "selected_seeds": selected_seeds,
            "raw_seeds": n_seeds - selected_seeds,
            "selection_rate": selected_seeds / n_seeds if n_seeds else 0.0,
            "selected_families": selected_families,
            "selected_weights": selected_weights,
            "negative_selected_seeds": int(negative_selected["seed"].nunique()),
            "mean_selected_eval_delta_auc": mean_selected_eval_delta_auc,
            "mean_selected_eval_delta_ap": mean_selected_eval_delta_ap,
            "min_selected_eval_delta_auc": min_selected_eval_delta_auc,
            "min_selected_eval_delta_ap": min_selected_eval_delta_ap,
        }
        row["status"] = _status(pd.Series(row), args.strong_min_seeds, args.exploratory_min_seeds)
        source_rows.append(row)

    source_summary = pd.DataFrame(source_rows).merge(
        all_seed_counts, on=["stability_group", "source_model"], how="left", suffixes=("", "_check")
    )
    if "n_seeds_check" in source_summary.columns:
        source_summary = source_summary.drop(columns=["n_seeds_check"])

    group_summary = (
        source_summary.groupby("stability_group", as_index=False)
        .agg(
            n_sources=("source_model", "nunique"),
            promoted_sources=("status", lambda s: int((s == "promote").sum())),
            exploratory_sources=("status", lambda s: int((s == "exploratory").sum())),
            rejected_sources=("status", lambda s: int((s == "reject").sum())),
            sources_with_negative_selected_seed=("negative_selected_seeds", lambda s: int((s > 0).sum())),
            mean_selection_rate=("selection_rate", "mean"),
        )
    )
    return source_summary, group_summary, long_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-source", action="append", required=True, help="group=selector_per_source.csv")
    parser.add_argument("--output-source-summary-csv", type=Path, required=True)
    parser.add_argument("--output-group-summary-csv", type=Path, required=True)
    parser.add_argument("--output-long-csv", type=Path)
    parser.add_argument("--default-seed", type=int, default=0)
    parser.add_argument("--strong-min-seeds", type=int, default=4)
    parser.add_argument("--exploratory-min-seeds", type=int, default=3)
    args = parser.parse_args()

    source_summary, group_summary, long_df = run(args)
    args.output_source_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    source_summary.to_csv(args.output_source_summary_csv, index=False)
    group_summary.to_csv(args.output_group_summary_csv, index=False)
    if args.output_long_csv:
        long_df.to_csv(args.output_long_csv, index=False)

    print(source_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print(group_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
