#!/usr/bin/env python3
"""Discover local files that could become fresh-validation candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY_COLUMNS = {"subset", "source_model", "filename"}
GLOBAL_COLUMNS = KEY_COLUMNS | {"final_score"}
VIDEO_MANIFEST_COLUMNS = {"id", "label", "source_model", "original_filename"}


def _csv_columns(path: Path) -> set[str]:
    try:
        return set(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return set()


def _dataset_from_result_name(path: Path) -> str:
    name = path.name
    for suffix in ["_results.csv", "_1s_results.csv"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    if name == "genvideo_1s_results.csv":
        return "genvideo_1s"
    return path.stem


def _has_matching_score_files(root: Path, dataset: str) -> tuple[bool, bool, str, str]:
    patch_hits = sorted(
        p for p in (root / "results").rglob("*.csv")
        if dataset.split("_1s")[0] in p.name and "patch" in str(p).lower()
    )
    persistence_hits = sorted(
        p for p in (root / "results").rglob("*.csv")
        if dataset.split("_1s")[0] in p.name and "persistence" in str(p).lower()
    )
    return (
        bool(patch_hits),
        bool(persistence_hits),
        str(patch_hits[0]) if patch_hits else "",
        str(persistence_hits[0]) if persistence_hits else "",
    )


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    known = set()
    if args.inventory_csv.exists():
        inv = pd.read_csv(args.inventory_csv)
        known = set(inv["candidate"].astype(str))

    for path in sorted((args.root / "results").glob("*_results.csv")):
        columns = _csv_columns(path)
        dataset = _dataset_from_result_name(path)
        patch_exists, persistence_exists, patch_example, persistence_example = _has_matching_score_files(args.root, dataset)
        has_global_schema = GLOBAL_COLUMNS.issubset(columns)
        already_in_inventory = dataset in known
        if already_in_inventory:
            status = "KNOWN_INVENTORY_CANDIDATE"
        elif has_global_schema and patch_exists and persistence_exists:
            status = "POTENTIAL_LOCAL_READY_NEEDS_PROVENANCE"
        elif has_global_schema:
            status = "GLOBAL_ONLY_OR_INCOMPLETE_LOCAL"
        else:
            status = "NOT_SCORE_INPUT"
        rows.append(
            {
                "kind": "global_score_csv",
                "candidate": dataset,
                "path": str(path),
                "status": status,
                "already_in_inventory": already_in_inventory,
                "has_global_schema": has_global_schema,
                "has_patch_candidate": patch_exists,
                "has_persistence_candidate": persistence_exists,
                "patch_example": patch_example,
                "persistence_example": persistence_example,
                "missing_for_fresh": "" if status == "POTENTIAL_LOCAL_READY_NEEDS_PROVENANCE" else "provenance or complete aligned score triplet",
            }
        )

    for path in sorted((args.root / "datasets").glob("*/videos.csv")):
        columns = _csv_columns(path)
        has_video_manifest_schema = VIDEO_MANIFEST_COLUMNS.issubset(columns)
        status = "VIDEO_MANIFEST_ONLY" if has_video_manifest_schema else "UNKNOWN_DATASET_CSV"
        rows.append(
            {
                "kind": "dataset_video_manifest",
                "candidate": path.parent.name,
                "path": str(path),
                "status": status,
                "already_in_inventory": path.parent.name in known,
                "has_global_schema": False,
                "has_patch_candidate": False,
                "has_persistence_candidate": False,
                "patch_example": "",
                "persistence_example": "",
                "missing_for_fresh": "global scores, raw patch scores, persistence scores, and key normalization",
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--inventory-csv", type=Path, default=Path("results/patch_calibrated_persistence/fresh_candidate_inventory.csv"))
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    out = run(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(out.to_string(index=False))
    print(f"Saved local fresh candidate discovery -> {args.output_csv}")


if __name__ == "__main__":
    main()
