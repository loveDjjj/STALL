#!/usr/bin/env python3
"""Validate score CSV inputs before running fresh sample-fallback validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY_COLUMNS = ["subset", "source_model", "filename"]


def _read(path: Path, score_col: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [*KEY_COLUMNS, score_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} {path} missing columns: {missing}")
    out = df[required].copy()
    for col in KEY_COLUMNS:
        out[col] = out[col].astype(str)
    dup = out.duplicated(KEY_COLUMNS).sum()
    if dup:
        raise ValueError(f"{label} {path} has {dup} duplicated key rows")
    return out.rename(columns={score_col: f"{label}_score"})


def _source_summary(df: pd.DataFrame) -> pd.DataFrame:
    fake = df[df["subset"].str.lower() != "real"]
    rows = []
    for source, group in fake.groupby("source_model", sort=True):
        rows.append({"source_model": source, "n_fake": len(group)})
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_df = _read(args.global_csv, args.global_score_col, "global")
    patch_df = _read(args.raw_patch_csv, args.raw_patch_score_col, "raw_patch")
    persistence_df = _read(args.persistence_csv, args.persistence_score_col, "persistence")

    merged = global_df.merge(patch_df, on=KEY_COLUMNS, how="inner", validate="one_to_one").merge(
        persistence_df,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    counts = {
        "dataset": args.dataset,
        "global_rows": len(global_df),
        "raw_patch_rows": len(patch_df),
        "persistence_rows": len(persistence_df),
        "merged_rows": len(merged),
        "all_inputs_aligned": len(merged) == len(global_df) == len(patch_df) == len(persistence_df),
        "n_real": int((merged["subset"].str.lower() == "real").sum()),
        "n_fake": int((merged["subset"].str.lower() != "real").sum()),
        "n_fake_sources": int(merged.loc[merged["subset"].str.lower() != "real", "source_model"].nunique()),
        "global_score_min": float(merged["global_score"].min()),
        "global_score_max": float(merged["global_score"].max()),
        "raw_patch_score_min": float(merged["raw_patch_score"].min()),
        "raw_patch_score_max": float(merged["raw_patch_score"].max()),
        "persistence_score_min": float(merged["persistence_score"].min()),
        "persistence_score_max": float(merged["persistence_score"].max()),
    }
    checks = []
    checks.append({"check": "all_inputs_aligned", "passed": bool(counts["all_inputs_aligned"]), "detail": str(counts)})
    checks.append({"check": "has_real_rows", "passed": counts["n_real"] > 0, "detail": str(counts["n_real"])})
    checks.append({"check": "has_fake_rows", "passed": counts["n_fake"] > 0, "detail": str(counts["n_fake"])})
    checks.append({"check": "has_fake_sources", "passed": counts["n_fake_sources"] > 0, "detail": str(counts["n_fake_sources"])})
    summary = pd.DataFrame([counts])
    source_summary = _source_summary(merged)
    checks_df = pd.DataFrame(checks)
    if not bool(checks_df["passed"].all()):
        raise ValueError(f"Fresh input validation failed: {checks_df.to_dict(orient='records')}")
    return summary, source_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--global-csv", type=Path, required=True)
    parser.add_argument("--raw-patch-csv", type=Path, required=True)
    parser.add_argument("--persistence-csv", type=Path, required=True)
    parser.add_argument("--persistence-score-col", required=True)
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    parser.add_argument("--output-source-summary-csv", type=Path, required=True)
    parser.add_argument("--global-score-col", default="final_score")
    parser.add_argument("--raw-patch-score-col", default="final_score")
    args = parser.parse_args()

    summary, source_summary = run(args)
    args.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_source_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary_csv, index=False)
    source_summary.to_csv(args.output_source_summary_csv, index=False)
    print(summary.to_string(index=False))
    print(f"Saved validation summary -> {args.output_summary_csv}")
    print(f"Saved source summary -> {args.output_source_summary_csv}")


if __name__ == "__main__":
    main()
