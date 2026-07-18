#!/usr/bin/env python3
"""Merge completed Hotshot-XL rows into full VideoFeedback Hotshot triplets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY_COLUMNS = ["subset", "source_model", "filename"]


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in KEY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing key columns: {missing}")
    dup = int(df.duplicated(KEY_COLUMNS).sum())
    if dup:
        raise ValueError(f"{path} has duplicated keys: {dup}")
    return df


def _concat_like(base: pd.DataFrame, extra: pd.DataFrame, label: str) -> pd.DataFrame:
    missing = [col for col in base.columns if col not in extra.columns]
    if missing:
        raise ValueError(f"{label} extra rows missing columns required by base: {missing}")
    extra_aligned = extra[base.columns].copy()
    out = pd.concat([base, extra_aligned], ignore_index=True)
    dup = int(out.duplicated(KEY_COLUMNS).sum())
    if dup:
        raise ValueError(f"{label} merged rows have duplicated keys: {dup}")
    return out


def _coverage(global_df: pd.DataFrame, patch_df: pd.DataFrame, persistence_df: pd.DataFrame) -> dict[str, object]:
    patch_join = global_df[KEY_COLUMNS].merge(patch_df[KEY_COLUMNS], on=KEY_COLUMNS, how="left", indicator=True)
    pers_join = global_df[KEY_COLUMNS].merge(persistence_df[KEY_COLUMNS], on=KEY_COLUMNS, how="left", indicator=True)
    return {
        "global_rows": int(len(global_df)),
        "patch_rows": int(len(patch_df)),
        "persistence_rows": int(len(persistence_df)),
        "missing_patch_rows": int((patch_join["_merge"] == "left_only").sum()),
        "missing_persistence_rows": int((pers_join["_merge"] == "left_only").sum()),
        "full_triplet_ready": bool((patch_join["_merge"] == "both").all() and (pers_join["_merge"] == "both").all()),
    }


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    planned_patch_exists = args.hotshot_patch_csv.exists()
    planned_persistence_exists = args.hotshot_persistence_csv.exists()
    if not planned_patch_exists or not planned_persistence_exists:
        decision = pd.DataFrame(
            [
                {
                    "decision": "WAITING_FOR_HOTSHOT_COMPLETION",
                    "hotshot_patch_exists": planned_patch_exists,
                    "hotshot_persistence_exists": planned_persistence_exists,
                    "merged_patch_csv": str(args.output_patch_csv),
                    "merged_persistence_csv": str(args.output_persistence_csv),
                    "detail": "Run Hotshot-XL patch and persistence completion before merging.",
                }
            ]
        )
        return decision, pd.DataFrame(
            columns=[
                "global_rows",
                "patch_rows",
                "persistence_rows",
                "missing_patch_rows",
                "missing_persistence_rows",
                "full_triplet_ready",
            ]
        )

    global_df = _read(args.hotshot_global_csv)
    base_patch = _read(args.base_patch_csv)
    base_persistence = _read(args.base_persistence_csv)
    hotshot_patch = _read(args.hotshot_patch_csv)
    hotshot_persistence = _read(args.hotshot_persistence_csv)

    merged_patch = _concat_like(base_patch, hotshot_patch, "patch")
    merged_persistence = _concat_like(base_persistence, hotshot_persistence, "persistence")
    coverage = _coverage(global_df, merged_patch, merged_persistence)

    if coverage["full_triplet_ready"]:
        args.output_patch_csv.parent.mkdir(parents=True, exist_ok=True)
        args.output_persistence_csv.parent.mkdir(parents=True, exist_ok=True)
        merged_patch.to_csv(args.output_patch_csv, index=False)
        merged_persistence.to_csv(args.output_persistence_csv, index=False)
        decision_value = "MERGED_FULL_TRIPLET_READY"
    else:
        decision_value = "MERGED_BUT_INCOMPLETE"

    decision = pd.DataFrame(
        [
            {
                "decision": decision_value,
                "hotshot_patch_exists": True,
                "hotshot_persistence_exists": True,
                "merged_patch_csv": str(args.output_patch_csv),
                "merged_persistence_csv": str(args.output_persistence_csv),
                "detail": "Full triplet ready." if coverage["full_triplet_ready"] else "Merged outputs do not cover every hotshot global key.",
                **coverage,
            }
        ]
    )
    return decision, pd.DataFrame([coverage])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotshot-global-csv", type=Path, required=True)
    parser.add_argument("--base-patch-csv", type=Path, required=True)
    parser.add_argument("--base-persistence-csv", type=Path, required=True)
    parser.add_argument("--hotshot-patch-csv", type=Path, required=True)
    parser.add_argument("--hotshot-persistence-csv", type=Path, required=True)
    parser.add_argument("--output-patch-csv", type=Path, required=True)
    parser.add_argument("--output-persistence-csv", type=Path, required=True)
    parser.add_argument("--output-decision-csv", type=Path, required=True)
    parser.add_argument("--output-coverage-csv", type=Path, required=True)
    args = parser.parse_args()

    decision, coverage = run(args)
    args.output_decision_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_coverage_csv.parent.mkdir(parents=True, exist_ok=True)
    decision.to_csv(args.output_decision_csv, index=False)
    coverage.to_csv(args.output_coverage_csv, index=False)
    print(decision.to_string(index=False))
    if len(coverage):
        print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
