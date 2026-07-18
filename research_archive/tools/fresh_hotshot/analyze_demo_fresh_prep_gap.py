#!/usr/bin/env python3
"""Analyze what is missing before demo_dataset can run fresh validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _expected_video_path(root: Path, row: pd.Series) -> Path:
    subset = "real" if str(row["label"]).lower() == "real" else "fake"
    return root / "datasets" / "demo_dataset" / subset / str(row["source_model"]) / f"{int(row['id'])}.mp4"


def _resolve_video_path(root: Path, row: pd.Series) -> tuple[Path, str]:
    expected = _expected_video_path(root, row)
    if expected.exists():
        return expected, "demo_id_path"
    original = str(row["original_filename"])
    matches = sorted((root / "datasets").rglob(original))
    if len(matches) == 1:
        return matches[0], "original_filename_unique_fallback"
    return expected, "missing" if not matches else "ambiguous_original_filename"


def _score_path(root: Path, name: str) -> Path:
    return root / "results" / "patch_calibrated_persistence" / name


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(args.manifest_csv)
    required_cols = {"id", "label", "source_model", "original_filename"}
    missing_cols = sorted(required_cols - set(manifest.columns))
    if missing_cols:
        raise ValueError(f"{args.manifest_csv} missing columns: {missing_cols}")

    rows = []
    for _, row in manifest.iterrows():
        expected_video_path = _expected_video_path(args.root, row)
        video_path, resolution = _resolve_video_path(args.root, row)
        subset = "real" if str(row["label"]).lower() == "real" else "annotated"
        rows.append(
            {
                "subset": subset,
                "source_model": str(row["source_model"]),
                "filename": video_path.name,
                "video_path": str(video_path.relative_to(args.root)),
                "expected_demo_video_path": str(expected_video_path.relative_to(args.root)),
                "path_resolution": resolution,
                "original_filename": str(row["original_filename"]),
                "video_exists": video_path.exists(),
            }
        )
    index_preview = pd.DataFrame(rows)

    global_csv = _score_path(args.root, "demo_dataset_global_scores.csv")
    raw_patch_csv = _score_path(args.root, "demo_dataset_raw_patch_scores.csv")
    persistence_csv = _score_path(args.root, "demo_dataset_persistence_scores.csv")
    all_video_paths_exist = bool(index_preview["video_exists"].all())
    has_real = bool((index_preview["subset"] == "real").any())
    has_fake = bool((index_preview["subset"] != "real").any())
    n_fake_sources = int(index_preview.loc[index_preview["subset"] != "real", "source_model"].nunique())
    can_build_index = all_video_paths_exist and has_real and has_fake and n_fake_sources > 0
    score_triplet_exists = global_csv.exists() and raw_patch_csv.exists() and persistence_csv.exists()

    summary = pd.DataFrame(
        [
            {
                "candidate": "demo_dataset",
                "status": "READY_FOR_SCORE_GENERATION" if can_build_index and not score_triplet_exists else "SCORE_TRIPLET_READY" if score_triplet_exists else "MANIFEST_OR_VIDEO_GAP",
                "manifest_rows": int(len(manifest)),
                "index_rows": int(len(index_preview)),
                "n_real": int((index_preview["subset"] == "real").sum()),
                "n_fake": int((index_preview["subset"] != "real").sum()),
                "n_fake_sources": n_fake_sources,
                "all_video_paths_exist": all_video_paths_exist,
                "can_build_index": can_build_index,
                "global_scores_exist": global_csv.exists(),
                "raw_patch_scores_exist": raw_patch_csv.exists(),
                "persistence_scores_exist": persistence_csv.exists(),
                "score_triplet_exists": score_triplet_exists,
                "required_next": (
                    "generate global scores, raw patch scores, and persistence scores"
                    if can_build_index and not score_triplet_exists
                    else "repair manifest/video paths"
                    if not can_build_index
                    else "run frozen fresh scaffold"
                ),
            }
        ]
    )
    return summary, index_preview


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest-csv", type=Path, default=Path("datasets/demo_dataset/videos.csv"))
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    parser.add_argument("--output-index-preview-csv", type=Path, required=True)
    args = parser.parse_args()

    summary, preview = run(args)
    args.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_index_preview_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary_csv, index=False)
    preview.to_csv(args.output_index_preview_csv, index=False)
    print(summary.to_string(index=False))
    print(f"Saved summary -> {args.output_summary_csv}")
    print(f"Saved index preview -> {args.output_index_preview_csv}")


if __name__ == "__main__":
    main()
