#!/usr/bin/env python3
"""Build a dry-run runbook for generating demo_dataset fresh score triplets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _cmd(parts: list[object]) -> str:
    return " ".join(str(p) for p in parts)


def _duration_window_gap(index_csv: Path, duration: int) -> tuple[int | None, list[str]]:
    if not index_csv.exists():
        return None, []
    index = pd.read_csv(index_csv)
    window_col = f"{duration}_sec_idxs"
    if window_col not in index.columns:
        return len(index), []
    missing = index[index[window_col].isna()]
    return int(len(missing)), missing["video_path"].astype(str).tolist()


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    prep = pd.read_csv(args.prep_summary_csv)
    if len(prep) != 1:
        raise ValueError(f"{args.prep_summary_csv} must contain one summary row")
    row = prep.iloc[0]
    ready = str(row["status"]) == "READY_FOR_SCORE_GENERATION"
    missing_duration_windows, missing_window_paths = _duration_window_gap(args.demo_index_csv, 2)
    duration_ready = missing_duration_windows == 0
    score_status = (
        "READY"
        if ready and duration_ready
        else "BLOCKED_DURATION_WINDOW_GAP"
        if ready and missing_duration_windows not in (None, 0)
        else "WAITING_FOR_ENRICHED_INDEX"
        if ready
        else "BLOCKED"
    )
    score_note_suffix = (
        ""
        if duration_ready
        else f" Blocked for duration=2 full-demo scoring: {missing_duration_windows} rows lack 2_sec_idxs ({'; '.join(missing_window_paths)})."
        if missing_duration_windows not in (None, 0)
        else " Waiting for enriched demo index before checking duration-window coverage."
    )

    steps = [
        {
            "step": "write_enriched_index",
            "status": "READY" if ready else "BLOCKED",
            "output": str(args.demo_index_csv),
            "command": _cmd(
                [
                    "python3",
                    "tools/write_demo_fresh_index.py",
                    "--index-preview-csv",
                    args.index_preview_csv,
                    "--output-csv",
                    args.demo_index_csv,
                ]
            ),
            "note": "Create enriched index with ffprobe metadata and downsampled 1/2/3/4_sec_idxs from resolved demo video paths.",
        },
        {
            "step": "global_scores",
            "status": score_status,
            "output": str(args.global_scores_csv),
            "command": _cmd(
                [
                    "python3",
                    "src/eval.py",
                    "--csv",
                    args.demo_index_csv,
                    "--emb-cache",
                    args.emb_cache,
                    "--output-csv",
                    args.global_scores_csv,
                    "--duration",
                    2,
                    "--compact",
                ]
            ),
            "note": "Generate STALL global final_score for all demo rows." + score_note_suffix,
        },
        {
            "step": "patch_cache_prefill",
            "status": score_status,
            "output": str(args.patch_cache),
            "command": _cmd(
                [
                    "python3",
                    "tools/prefill_patch_cache.py",
                    "--csv",
                    args.demo_index_csv,
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
            "note": "Populate patch embedding cache before patch-only and persistence scoring." + score_note_suffix,
        },
        {
            "step": "raw_patch_scores",
            "status": score_status,
            "output": str(args.raw_patch_scores_csv),
            "command": _cmd(
                [
                    "python3",
                    "src/eval_patch_fast.py",
                    "--csv",
                    args.demo_index_csv,
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
            "note": "Use the VideoFeedback-compatible region1 mean second-order raw patch default." + score_note_suffix,
        },
        {
            "step": "persistence_scores",
            "status": score_status,
            "output": str(args.persistence_scores_csv),
            "command": _cmd(
                [
                    "python3",
                    "tools/patch_calibrated_persistence_scores.py",
                    "--csv",
                    args.demo_index_csv,
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
            "note": "Use VideoFeedback real calibration for the selected persistence score column." + score_note_suffix,
        },
        {
            "step": "fresh_scaffold",
            "status": "READY" if score_status == "READY" else score_status,
            "output": str(args.scaffold_json),
            "command": _cmd(
                [
                    "python3",
                    "tools/build_fresh_validation_scaffold.py",
                    "--manifest-json",
                    "results/patch_calibrated_persistence/sample_fallback_preregistered_manifest.json",
                    "--dataset",
                    "demo_dataset",
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
            "note": "Build frozen scorer scaffold after all three score inputs exist." + score_note_suffix,
        },
    ]

    df = pd.DataFrame(steps)
    markdown = "\n".join(
        [
            "# Demo Fresh Score Triplet Runbook",
            "",
            f"Status: {score_status if ready else 'BLOCKED'}",
            "",
            f"Duration-2 missing-window rows: {missing_duration_windows if missing_duration_windows is not None else 'UNKNOWN'}",
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
    parser.add_argument("--prep-summary-csv", type=Path, default=Path("results/patch_calibrated_persistence/demo_fresh_prep_gap_summary.csv"))
    parser.add_argument("--index-preview-csv", type=Path, default=Path("results/patch_calibrated_persistence/demo_fresh_index_preview.csv"))
    parser.add_argument("--demo-index-csv", type=Path, default=Path("cache/indexes/demo_dataset.csv"))
    parser.add_argument("--emb-cache", type=Path, default=Path("cache/embeddings/demo_dataset"))
    parser.add_argument("--patch-cache", type=Path, default=Path("cache/patch_embeddings/demo_dataset"))
    parser.add_argument("--patch-cache-summary-csv", type=Path, default=Path("results/patch_calibrated_persistence/demo_dataset_patch_cache_prefill_summary.csv"))
    parser.add_argument("--patch-params", type=Path, default=Path("precomputed/patch_params_videofeedback_real_same_grid_second_order_region1_mean_v2.npz"))
    parser.add_argument("--global-scores-csv", type=Path, default=Path("results/patch_calibrated_persistence/demo_dataset_global_scores.csv"))
    parser.add_argument("--raw-patch-scores-csv", type=Path, default=Path("results/patch_calibrated_persistence/demo_dataset_raw_patch_scores.csv"))
    parser.add_argument("--persistence-scores-csv", type=Path, default=Path("results/patch_calibrated_persistence/demo_dataset_persistence_scores.csv"))
    parser.add_argument("--scaffold-json", type=Path, default=Path("results/patch_calibrated_persistence/demo_dataset_fresh_validation_scaffold.json"))
    parser.add_argument("--scaffold-md", type=Path, default=Path("results/patch_calibrated_persistence/demo_dataset_fresh_validation_scaffold.md"))
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
