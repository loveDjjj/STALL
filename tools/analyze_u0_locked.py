#!/usr/bin/env python3
"""Calibrate locked U0 raw windows and write release video scores and metrics."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import sklearn
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.calibration import (
    cdf_with_positive_infinity,
)
from alpha_stalled.u0_analysis import (
    aggregate_and_calibrate_videos,
    calibrate_windows,
)
from alpha_stalled.metrics import metric_tables
from alpha_stalled.parameters import global_references
from alpha_stalled.release_io import sha256_file, write_json
from alpha_stalled.u0_protocol import (
    EXPECTED_EVALUATION,
    KEY_COLUMNS as U0_KEY_COLUMNS,
    WINDOW_KEYS as U0_WINDOW_KEYS,
    load_calibration_references,
    load_raw_windows,
    selected_calibration_means,
    verify_calibration_reference_windows,
)


KEY_COLUMNS = list(U0_KEY_COLUMNS)
WINDOW_KEYS = list(U0_WINDOW_KEYS)


def run(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    windows = load_raw_windows(args.raw_dir, args.num_shards)
    calibration_references = load_calibration_references(
        args.calibration_raw_dir, args.num_shards
    )
    verify_calibration_reference_windows(calibration_references, args.release_dir)
    scored_windows, local_references = calibrate_windows(
        windows, calibration_references, config
    )
    evaluation, video_references = aggregate_and_calibrate_videos(scored_windows)
    counts = evaluation.groupby("dataset").size().astype(int).to_dict()
    if counts != EXPECTED_EVALUATION:
        raise ValueError(f"locked evaluation counts {counts} != {EXPECTED_EVALUATION}")

    dataset_metrics, generator_metrics = metric_tables(
        evaluation,
        seed=int(config["release"]["random_seed"]),
        score_columns=("S",),
        config_names={"S": "U0_locked"},
    )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    scored_windows.to_csv(output / "per_window_scores.csv", index=False)
    evaluation.to_csv(output / "final_video_scores.csv", index=False)
    local_references.to_csv(output / "local_window_calibration.csv", index=False)
    video_references.to_csv(output / "video_cdf_audit.csv", index=False)
    dataset_metrics.to_csv(output / "dataset_metrics.csv", index=False)
    generator_metrics.to_csv(output / "generator_metrics.csv", index=False)
    args.release_dir.mkdir(parents=True, exist_ok=True)
    evaluation.to_csv(args.release_dir / "final_video_scores.csv", index=False)
    raw_files = sorted(args.raw_dir.glob("*_shard*_of_*.csv"))
    calibration_raw_files = sorted(
        args.calibration_raw_dir.glob("*_shard*_of_*.csv")
    )
    macro = dataset_metrics[dataset_metrics["dataset"] == "Macro-3"].iloc[0]
    metadata = {
        "schema_version": "u0_reproduction_v1",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.relative_to(ROOT)),
        "raw_directory": str(args.raw_dir),
        "calibration_raw_directory": str(args.calibration_raw_dir),
        "num_shards_per_dataset": args.num_shards,
        "raw_window_count": len(scored_windows),
        "evaluation_video_count": len(evaluation),
        "macro_auc": float(macro["auc"]),
        "macro_real_positive_ap": float(macro["ap"]),
        "raw_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in raw_files
        },
        "calibration_raw_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in calibration_raw_files
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": str(torch.__version__),
            "sklearn": sklearn.__version__,
        },
        "protocol_correction": "PatchSpatial and PatchD2 both recomputed with region1 mean",
    }
    write_json(args.release_dir / "reproduction_metadata.json", metadata)
    hashes_path = args.release_dir / "config_and_checkpoint_hashes.json"
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    hashes["release_outputs"] = {
        "final_video_scores.csv": {
            "path": "release/u0/final_video_scores.csv",
            "sha256": sha256_file(args.release_dir / "final_video_scores.csv"),
        },
        "reproduction_metadata.json": {
            "path": "release/u0/reproduction_metadata.json",
            "sha256": sha256_file(args.release_dir / "reproduction_metadata.json"),
        },
    }
    write_json(hashes_path, hashes)
    print(dataset_metrics.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/alpha_stalled_u0_locked.yaml"
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--calibration-raw-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, default=ROOT / "release/u0")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
