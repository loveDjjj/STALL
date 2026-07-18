#!/usr/bin/env python3
"""Build a score-triplet runbook for the duration=2 filtered demo subset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DATASET = "demo_dataset_duration2_filtered"


def _cmd(parts: list[object]) -> str:
    return " ".join(str(p) for p in parts)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    summary = pd.read_csv(args.filtered_summary_csv)
    if len(summary) != 1:
        raise ValueError(f"{args.filtered_summary_csv} must contain one row")
    row = summary.iloc[0]
    ready = str(row["status"]) == "READY_FOR_DURATION2_SCORE_GENERATION"
    status = "READY" if ready else "BLOCKED"
    caveat = "Filtered duration=2 subset: 34/36 demo rows; excludes HotShot and MoonValley 1-second videos."

    steps = [
        {
            "step": "global_scores",
            "status": status,
            "output": str(args.global_scores_csv),
            "command": _cmd(
                [
                    "python3",
                    "src/eval.py",
                    "--csv",
                    args.filtered_index_csv,
                    "--emb-cache",
                    args.emb_cache,
                    "--output-csv",
                    args.global_scores_csv,
                    "--duration",
                    2,
                    "--compact",
                ]
            ),
            "note": "Generate STALL global final_score on the duration=2 filtered demo subset. " + caveat,
        },
        {
            "step": "patch_cache_prefill",
            "status": status,
            "output": str(args.patch_cache),
            "command": _cmd(
                [
                    "python3",
                    "tools/prefill_patch_cache.py",
                    "--csv",
                    args.filtered_index_csv,
                    "--patch-emb-cache",
                    args.patch_cache,
                    "--duration",
                    2,
                    "--compact",
                    "--execute",
                    "--output-summary-csv",
                    args.patch_cache_summary_csv,
                ]
            ),
            "note": "Populate patch embedding cache before patch-only and persistence scoring. " + caveat,
        },
        {
            "step": "raw_patch_scores",
            "status": status,
            "output": str(args.raw_patch_scores_csv),
            "command": _cmd(
                [
                    "python3",
                    "src/eval_patch_fast.py",
                    "--csv",
                    args.filtered_index_csv,
                    "--patch-emb-cache",
                    args.patch_cache,
                    "--patch-params",
                    args.patch_params,
                    "--output-csv",
                    args.raw_patch_scores_csv,
                    "--duration",
                    2,
                    "--compact",
                    "--patch-temp-mode",
                    "same_grid_second_order",
                    "--patch-spat-weight",
                    0.10,
                    "--patch-temp-weight",
                    0.90,
                    "--aggregation",
                    "mean",
                    "--patch-region-size",
                    1,
                ]
            ),
            "note": "Use the VideoFeedback-compatible region1 mean second-order raw patch default. " + caveat,
        },
        {
            "step": "persistence_scores",
            "status": status,
            "output": str(args.persistence_scores_csv),
            "command": _cmd(
                [
                    "python3",
                    "tools/patch_calibrated_persistence_scores.py",
                    "--csv",
                    args.filtered_index_csv,
                    "--calibration-csv",
                    "cache/indexes/videofeedback.csv",
                    "--calibration-duration",
                    2,
                    "--calibration-patch-emb-cache",
                    "cache/patch_embeddings/videofeedback",
                    "--patch-emb-cache",
                    args.patch_cache,
                    "--patch-params",
                    args.patch_params,
                    "--output-csv",
                    args.persistence_scores_csv,
                    "--duration",
                    2,
                    "--compact",
                    "--patch-temp-mode",
                    "same_grid_second_order",
                    "--patch-region-size",
                    1,
                ]
            ),
            "note": "Use VideoFeedback real calibration for the selected persistence score column. " + caveat,
        },
        {
            "step": "fresh_scaffold",
            "status": status,
            "output": str(args.scaffold_json),
            "command": _cmd(
                [
                    "python3",
                    "tools/build_fresh_validation_scaffold.py",
                    "--manifest-json",
                    "results/patch_calibrated_persistence/sample_fallback_preregistered_manifest.json",
                    "--dataset",
                    DATASET,
                    "--global-csv",
                    args.global_scores_csv,
                    "--raw-patch-csv",
                    args.raw_patch_scores_csv,
                    "--persistence-csv",
                    args.persistence_scores_csv,
                    "--persistence-score-col",
                    "max_frame_mass_thr0p2_real_pct_real",
                    "--output-json",
                    args.scaffold_json,
                    "--output-md",
                    args.scaffold_md,
                ]
            ),
            "note": "Build frozen scorer scaffold after all three score inputs exist. " + caveat,
        },
    ]
    df = pd.DataFrame(steps)
    markdown = "\n".join(
        [
            "# Demo Duration-2 Filtered Score Runbook",
            "",
            f"Dataset: {DATASET}",
            f"Status: {status}",
            "",
            caveat,
            "",
            "Commands:",
            "",
            "```bash",
            *df["command"].astype(str).tolist(),
            "```",
            "",
        ]
    )
    return df, markdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtered-index-csv", type=Path, default=Path("cache/indexes/demo_dataset_duration2_filtered.csv"))
    parser.add_argument("--filtered-summary-csv", type=Path, default=Path("results/patch_calibrated_persistence/demo_dataset_duration2_filtered_summary.csv"))
    parser.add_argument("--emb-cache", type=Path, default=Path("cache/embeddings/demo_dataset_duration2_filtered"))
    parser.add_argument("--patch-cache", type=Path, default=Path("cache/patch_embeddings/demo_dataset_duration2_filtered"))
    parser.add_argument("--patch-cache-summary-csv", type=Path, default=Path("results/patch_calibrated_persistence/demo_dataset_duration2_filtered_patch_cache_prefill_summary.csv"))
    parser.add_argument("--patch-params", type=Path, default=Path("precomputed/patch_params_videofeedback_real_same_grid_second_order_region1_mean_v2.npz"))
    parser.add_argument("--global-scores-csv", type=Path, default=Path("results/patch_calibrated_persistence/demo_dataset_duration2_filtered_global_scores.csv"))
    parser.add_argument("--raw-patch-scores-csv", type=Path, default=Path("results/patch_calibrated_persistence/demo_dataset_duration2_filtered_raw_patch_scores.csv"))
    parser.add_argument("--persistence-scores-csv", type=Path, default=Path("results/patch_calibrated_persistence/demo_dataset_duration2_filtered_persistence_scores.csv"))
    parser.add_argument("--scaffold-json", type=Path, default=Path("results/patch_calibrated_persistence/demo_dataset_duration2_filtered_fresh_validation_scaffold.json"))
    parser.add_argument("--scaffold-md", type=Path, default=Path("results/patch_calibrated_persistence/demo_dataset_duration2_filtered_fresh_validation_scaffold.md"))
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    steps, markdown = run(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    steps.to_csv(args.output_csv, index=False)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(steps.to_string(index=False))
    print(f"Saved runbook CSV -> {args.output_csv}")
    print(f"Saved runbook Markdown -> {args.output_md}")


if __name__ == "__main__":
    main()
