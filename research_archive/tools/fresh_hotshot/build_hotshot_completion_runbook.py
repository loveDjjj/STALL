#!/usr/bin/env python3
"""Build the Hotshot-XL completion manifest and runbook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


KEY_COLUMNS = ["subset", "source_model", "filename"]


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in KEY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df


def _cache_path(cache_root: Path, subset: str, source_model: str, filename: str, duration: int, compact: bool) -> Path:
    stem = Path(filename).stem
    suffix = f"{duration}s_compact.pt" if compact else f"{duration}s.pt"
    return cache_root / subset / source_model / f"{stem}_{suffix}"


def _normalize_video_path(value: object, strip_prefix: str) -> str:
    text = str(value)
    prefix = strip_prefix.strip("/")
    if not prefix:
        return text
    prefix = f"{prefix}/"
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    hotshot = _read(args.hotshot_global_csv)
    patch = _read(args.raw_patch_csv)
    persistence = _read(args.persistence_csv)
    index = pd.read_csv(args.index_csv)
    index["filename"] = index["video_path"].map(lambda value: Path(str(value)).name)
    missing_index_cols = [col for col in [*KEY_COLUMNS, "video_path", f"{args.duration}_sec_idxs"] if col not in index.columns]
    if missing_index_cols:
        raise ValueError(f"{args.index_csv} missing columns needed for patch cache prefill: {missing_index_cols}")
    merged = hotshot[KEY_COLUMNS].merge(patch[KEY_COLUMNS], on=KEY_COLUMNS, how="left", indicator="patch_merge")
    merged = merged.merge(persistence[KEY_COLUMNS], on=KEY_COLUMNS, how="left", indicator="persistence_merge")
    missing = merged[(merged["patch_merge"] == "left_only") | (merged["persistence_merge"] == "left_only")].copy()
    targets = hotshot.merge(missing[KEY_COLUMNS], on=KEY_COLUMNS, how="inner")
    if args.source_model:
        targets = targets[targets["source_model"].astype(str) == args.source_model].copy()
    target_keys = targets[KEY_COLUMNS].copy()
    targets = index.merge(target_keys, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    targets["video_path"] = targets["video_path"].map(
        lambda value: _normalize_video_path(value, args.strip_video_path_prefix)
    )
    targets = targets.sort_values(KEY_COLUMNS).reset_index(drop=True)

    cache_rows = []
    for _, row in targets.iterrows():
        cache_path = _cache_path(
            args.patch_emb_cache,
            str(row["subset"]),
            str(row["source_model"]),
            str(row["filename"]),
            args.duration,
            args.compact,
        )
        cache_rows.append(
            {
                **{col: row[col] for col in KEY_COLUMNS},
                "cache_path": str(cache_path),
                "cache_exists": cache_path.exists(),
            }
        )
    cache_status = pd.DataFrame(cache_rows)
    n_cache_exists = int(cache_status["cache_exists"].sum()) if len(cache_status) else 0

    hotshot_duration_note = (
        f"Hotshot-XL target rows use duration={args.duration}; this may differ from the existing "
        "VideoFeedback 2-second full-cache protocol and must be reported as a local-variant caveat."
    )
    commands = {
        "prefill_patch_cache": (
            "python3 tools/prefill_patch_cache.py "
            f"--csv {args.target_csv_placeholder} "
            f"--patch-emb-cache {args.patch_emb_cache} "
            f"--duration {args.duration} "
            f"{'--compact ' if args.compact else ''}"
            "--num-workers 8 "
            "--video-batch 4 "
            "--frame-batch 32 "
            "--device cuda "
            "--execute "
            "--output-summary-csv results/patch_calibrated_persistence/hotshot_xl_patch_cache_prefill_summary.csv"
        ),
        "score_raw_patch": (
            "python3 src/eval_patch_fast.py "
            f"--csv {args.target_csv_placeholder} "
            f"--patch-emb-cache {args.patch_emb_cache} "
            f"--patch-params {args.patch_params} "
            f"--output-csv {args.output_patch_csv} "
            f"--duration {args.duration} "
            f"{'--compact ' if args.compact else ''}"
            "--score-device cuda "
            "--score-batch-size 32 "
            "--patch-temp-mode same_grid_second_order "
            "--patch-spat-weight 0.10 "
            "--patch-temp-weight 0.90"
        ),
        "score_persistence": (
            "python3 tools/patch_calibrated_persistence_scores.py "
            f"--csv {args.target_csv_placeholder} "
            f"--patch-emb-cache {args.patch_emb_cache} "
            f"--patch-params {args.patch_params} "
            f"--output-csv {args.output_persistence_csv} "
            f"--duration {args.duration} "
            f"{'--compact ' if args.compact else ''}"
            "--patch-temp-mode same_grid_second_order "
            "--patch-region-size 1 "
            "--score-device cuda "
            "--score-batch-size 32 "
            "--thresholds 0.05,0.10,0.20"
        ),
    }
    summary = pd.DataFrame(
        [
            {
                "candidate": "videofeedback_hotshot_completion",
            "source_model": args.source_model,
            "duration": args.duration,
            "target_rows": int(len(targets)),
                "target_sources": int(targets["source_model"].nunique()) if len(targets) else 0,
                "patch_cache_rows_existing": n_cache_exists,
                "patch_cache_rows_missing": int(len(targets) - n_cache_exists),
                "target_csv": str(args.output_target_csv),
                "cache_status_csv": str(args.output_cache_status_csv),
                "output_patch_csv": str(args.output_patch_csv),
                "output_persistence_csv": str(args.output_persistence_csv),
                "patch_params": str(args.patch_params),
                "video_path_prefix_stripped": args.strip_video_path_prefix,
            "commands_json": json.dumps(commands, sort_keys=True),
            "duration_note": hotshot_duration_note,
        }
        ]
    )

    md = "\n".join(
        [
            "# Hotshot-XL Completion Runbook",
            "",
            "Purpose: generate the missing raw-patch and persistence rows needed for full `videofeedback_hotshot` local-variant scoring.",
            "",
            "Scope:",
            "",
            f"- source_model = {args.source_model}",
            f"- duration = {args.duration}",
            f"- target_rows = {len(targets)}",
            f"- existing patch cache rows = {n_cache_exists}",
            f"- missing patch cache rows = {len(targets) - n_cache_exists}",
            f"- target CSV = {args.output_target_csv}",
            f"- target video paths normalized for STALL working directory by stripping prefix `{args.strip_video_path_prefix}/`",
            "",
            "Frozen-compatible scoring config:",
            "",
            "- patch_temp_mode = same_grid_second_order",
            "- patch_region_size = 1",
            "- patch_spat_weight = 0.10",
            "- patch_temp_weight = 0.90",
            "- persistence score column for final scaffold = max_frame_mass_thr0p2_real_pct_real",
            f"- caveat: {hotshot_duration_note}",
            "",
            "Commands:",
            "",
            "```bash",
            commands["prefill_patch_cache"],
            commands["score_raw_patch"],
            commands["score_persistence"],
            "```",
            "",
            "After completion:",
            "",
            "1. concatenate these Hotshot-XL patch/persistence rows with the existing VideoFeedback patch/persistence files;",
            "2. build a concrete `videofeedback_hotshot` scaffold using `results/videofeedback_hotshot_results.csv`;",
            "3. run scaffold verification and `tools/run_fresh_validation_scaffold.py`;",
            "4. label the result as a local variant unless provenance is proven independent enough for fresh validation.",
            "",
        ]
    )
    return targets, cache_status, summary, md


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotshot-global-csv", type=Path, required=True)
    parser.add_argument("--index-csv", type=Path, required=True)
    parser.add_argument("--raw-patch-csv", type=Path, required=True)
    parser.add_argument("--persistence-csv", type=Path, required=True)
    parser.add_argument("--patch-emb-cache", type=Path, required=True)
    parser.add_argument("--patch-params", type=Path, required=True)
    parser.add_argument("--source-model", default="Hotshot-XL")
    parser.add_argument("--duration", type=int, default=2)
    parser.add_argument("--compact", action="store_true", default=True)
    parser.add_argument("--strip-video-path-prefix", default="STALL")
    parser.add_argument("--output-target-csv", type=Path, required=True)
    parser.add_argument("--output-cache-status-csv", type=Path, required=True)
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    parser.add_argument("--output-runbook-md", type=Path, required=True)
    parser.add_argument("--output-patch-csv", type=Path, required=True)
    parser.add_argument("--output-persistence-csv", type=Path, required=True)
    args = parser.parse_args()

    args.target_csv_placeholder = args.output_target_csv
    targets, cache_status, summary, md = run(args)
    args.output_target_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_cache_status_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_runbook_md.parent.mkdir(parents=True, exist_ok=True)
    targets.to_csv(args.output_target_csv, index=False)
    cache_status.to_csv(args.output_cache_status_csv, index=False)
    summary.to_csv(args.output_summary_csv, index=False)
    args.output_runbook_md.write_text(md, encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Saved target CSV -> {args.output_target_csv}")
    print(f"Saved cache status -> {args.output_cache_status_csv}")
    print(f"Saved runbook -> {args.output_runbook_md}")


if __name__ == "__main__":
    main()
