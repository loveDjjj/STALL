#!/usr/bin/env python3
"""Reuse exact duration-aware raw windows and split remaining original-K1 tasks."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CUSTOM_SIZE = {"comgenvid": 600, "videofeedback": 400, "genvideo": 1500}
RAW_COLUMNS = [
    "global_spatial_raw",
    "global_t1_raw",
    "patch_spatial_raw",
    "patch_d2_raw",
]


def normalized_indices(value: object) -> str:
    return json.dumps([int(item) for item in json.loads(str(value))], separators=(",", ":"))


def load_existing(raw_dir: Path, custom_dir: Path) -> pd.DataFrame:
    raw_paths = sorted(Path(path) for path in glob.glob(str(raw_dir / "*.csv")))
    custom_paths = sorted(Path(path) for path in glob.glob(str(custom_dir / "*.csv")))
    raw = pd.concat(
        [pd.read_csv(path, float_precision="round_trip") for path in raw_paths],
        ignore_index=True,
    )
    custom = pd.concat(
        [pd.read_csv(path, float_precision="round_trip") for path in custom_paths],
        ignore_index=True,
    )
    custom_parts = []
    for dataset, frame in custom.groupby("dataset", sort=False):
        size = CUSTOM_SIZE[str(dataset)]
        custom_parts.append(
            frame[
                ["task_id", "window_id", f"patch_spatial__n{size}", f"patch_d2__n{size}"]
            ].rename(
                columns={
                    f"patch_spatial__n{size}": "patch_spatial_raw",
                    f"patch_d2__n{size}": "patch_d2_raw",
                }
            )
        )
    selected = pd.concat(custom_parts, ignore_index=True)
    merged = raw.merge(selected, on=["task_id", "window_id"], validate="one_to_one")
    if len(merged) != len(raw):
        raise ValueError("custom raw score coverage is incomplete")
    merged["frame_indices"] = merged["frame_indices"].map(normalized_indices)
    keys = ["video_id", "protocol_duration_sec", "frame_indices"]
    duplicated = merged[merged.duplicated(keys, keep=False)]
    if not duplicated.empty:
        spread = duplicated.groupby(keys)[RAW_COLUMNS].agg(lambda values: values.max() - values.min())
        if float(spread.to_numpy().max()) > 1e-10:
            raise ValueError("duplicate exact windows have inconsistent raw scores")
    return merged.drop_duplicates(keys).reset_index(drop=True)


def run(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest, float_precision="round_trip")
    manifest["frame_indices"] = manifest["frame_indices"].map(normalized_indices)
    existing = load_existing(args.raw_dir, args.custom_raw_dir)
    keys = ["video_id", "protocol_duration_sec", "frame_indices"]
    reusable = manifest.merge(
        existing[keys + RAW_COLUMNS], on=keys, how="left", validate="one_to_one"
    )
    matched = reusable[reusable[RAW_COLUMNS].notna().all(axis=1)].copy()
    matched["score_source"] = "existing_k3_raw_exact_window"
    pending = reusable[reusable[RAW_COLUMNS].isna().any(axis=1)][manifest.columns].copy()
    cache = pending[pending["compact_cache_status"].eq("exists_unverified")].copy()
    dino = pending[~pending["compact_cache_status"].eq("exists_unverified")].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "original_k1_raw_reused.csv": matched,
        "original_k1_pending_cache.csv": cache,
        "original_k1_pending_dino.csv": dino,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False)
    if len(matched) + len(cache) + len(dino) != len(manifest):
        raise ValueError("K1 reuse partition is incomplete")
    summary = pd.DataFrame(
        [
            {"partition": "reused", "tasks": len(matched)},
            {"partition": "cache", "tasks": len(cache)},
            {"partition": "dino", "tasks": len(dino)},
            {"partition": "total", "tasks": len(manifest)},
        ]
    )
    summary.to_csv(args.output_dir / "original_k1_reuse_summary.csv", index=False)
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results/full_coverage_paper_protocol/original_k1_tasks.csv",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "results/duration_aware_23source/raw",
    )
    parser.add_argument(
        "--custom-raw-dir",
        type=Path,
        default=ROOT / "results/duration_aware_23source/curve_raw/selected_realonly",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/full_coverage_paper_protocol",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
