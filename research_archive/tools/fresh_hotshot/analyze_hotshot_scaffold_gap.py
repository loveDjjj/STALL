#!/usr/bin/env python3
"""Analyze why VideoFeedback Hotshot is not scaffold-ready."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY_COLUMNS = ["subset", "source_model", "filename"]


def _read_keys(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in KEY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{label} {path} missing key columns: {missing}")
    out = df[KEY_COLUMNS].copy()
    for col in KEY_COLUMNS:
        out[col] = out[col].astype(str)
    dup = int(out.duplicated(KEY_COLUMNS).sum())
    if dup:
        raise ValueError(f"{label} {path} has duplicated keys: {dup}")
    return out


def _coverage_rows(hotshot: pd.DataFrame, other: pd.DataFrame, label: str) -> tuple[dict[str, object], pd.DataFrame]:
    merged = hotshot.merge(other, on=KEY_COLUMNS, how="left", indicator=True)
    missing = merged[merged["_merge"] == "left_only"].copy()
    present = merged[merged["_merge"] == "both"].copy()
    summary = {
        "target": label,
        "hotshot_rows": int(len(hotshot)),
        "matched_rows": int(len(present)),
        "missing_rows": int(len(missing)),
        "coverage": float(len(present) / len(hotshot)) if len(hotshot) else 0.0,
        "missing_sources": int(missing["source_model"].nunique()) if len(missing) else 0,
    }
    by_source = (
        missing.groupby(["subset", "source_model"], sort=True)
        .size()
        .reset_index(name=f"missing_{label}_rows")
    )
    if by_source.empty:
        by_source = pd.DataFrame(columns=["subset", "source_model", f"missing_{label}_rows"])
    return summary, by_source


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hotshot = _read_keys(args.hotshot_global_csv, "hotshot_global")
    base = _read_keys(args.base_global_csv, "base_global")
    patch = _read_keys(args.raw_patch_csv, "raw_patch")
    persistence = _read_keys(args.persistence_csv, "persistence")

    summaries = []
    source_tables = []
    for label, other in [("base_global", base), ("raw_patch", patch), ("persistence", persistence)]:
        summary, by_source = _coverage_rows(hotshot, other, label)
        summaries.append(summary)
        source_tables.append(by_source)

    source_gap = source_tables[0]
    for table in source_tables[1:]:
        source_gap = source_gap.merge(table, on=["subset", "source_model"], how="outer")
    if source_gap.empty:
        source_gap = pd.DataFrame(columns=["subset", "source_model", "missing_base_global_rows", "missing_raw_patch_rows", "missing_persistence_rows"])
    for col in ["missing_base_global_rows", "missing_raw_patch_rows", "missing_persistence_rows"]:
        if col not in source_gap.columns:
            source_gap[col] = 0
        source_gap[col] = source_gap[col].fillna(0).astype(int)
    source_gap["missing_any_rows"] = source_gap[
        ["missing_base_global_rows", "missing_raw_patch_rows", "missing_persistence_rows"]
    ].max(axis=1)
    source_gap = source_gap.sort_values(["missing_any_rows", "source_model"], ascending=[False, True])

    shared = hotshot.merge(base, on=KEY_COLUMNS, how="inner").merge(patch, on=KEY_COLUMNS, how="inner").merge(persistence, on=KEY_COLUMNS, how="inner")
    decision = pd.DataFrame(
        [
            {
                "candidate": "videofeedback_hotshot",
                "hotshot_rows": int(len(hotshot)),
                "shared_triplet_rows": int(len(shared)),
                "full_hotshot_scaffold_ready": bool(len(shared) == len(hotshot)),
                "shared_subset_scaffold_ready": bool(len(shared) > 0),
                "missing_full_rows": int(len(hotshot) - len(shared)),
                "missing_source_models": int(source_gap["source_model"].nunique()) if len(source_gap) else 0,
                "blocking_source_models": ",".join(source_gap["source_model"].astype(str).tolist()),
                "recommended_next": (
                    "generate raw_patch and persistence for Hotshot-XL before full-hotshot scoring"
                    if len(shared) != len(hotshot)
                    else "full hotshot scaffold is ready"
                ),
            }
        ]
    )
    return pd.DataFrame(summaries), source_gap, decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotshot-global-csv", type=Path, required=True)
    parser.add_argument("--base-global-csv", type=Path, required=True)
    parser.add_argument("--raw-patch-csv", type=Path, required=True)
    parser.add_argument("--persistence-csv", type=Path, required=True)
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    parser.add_argument("--output-source-gap-csv", type=Path, required=True)
    parser.add_argument("--output-decision-csv", type=Path, required=True)
    args = parser.parse_args()

    summary, source_gap, decision = run(args)
    args.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_source_gap_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_decision_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary_csv, index=False)
    source_gap.to_csv(args.output_source_gap_csv, index=False)
    decision.to_csv(args.output_decision_csv, index=False)
    print(summary.to_string(index=False))
    print(source_gap.to_string(index=False))
    print(decision.to_string(index=False))


if __name__ == "__main__":
    main()
