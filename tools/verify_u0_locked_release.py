#!/usr/bin/env python3
"""Validate locked U0 manifests, raw reproduction, scores, hashes, and metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.u0_protocol import (
    EXPECTED_EVALUATION,
    load_calibration_references,
    load_raw_windows,
    verify_calibration_reference_windows,
)
from alpha_stalled.metrics import metric_tables
from alpha_stalled.release_io import sha256_file

def require(condition: bool, message: str, checks: list[dict]) -> None:
    checks.append({"check": message, "passed": bool(condition)})
    if not condition:
        raise ValueError(message)


def run(args: argparse.Namespace) -> None:
    checks: list[dict] = []
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    hashes = json.loads(
        (args.release_dir / "config_and_checkpoint_hashes.json").read_text()
    )
    for name, item in hashes["input_files"].items():
        path = ROOT / item["path"]
        require(path.is_file(), f"input exists: {name}", checks)
        require(
            sha256_file(path) == item["sha256"],
            f"input SHA256 matches: {name}",
            checks,
        )
    for name, item in hashes["release_manifests"].items():
        path = ROOT / item["path"]
        require(path.is_file(), f"release manifest exists: {name}", checks)
        require(
            sha256_file(path) == item["sha256"],
            f"release manifest SHA256 matches: {name}",
            checks,
        )

    calibration = json.loads(
        (args.release_dir / "calibration_manifest.json").read_text()
    )
    evaluation = json.loads(
        (args.release_dir / "evaluation_manifest.json").read_text()
    )
    indices = json.loads((args.release_dir / "frame_indices.json").read_text())
    calibration_ids = {row["video_id"] for row in calibration["videos"]}
    evaluation_ids = {row["video_id"] for row in evaluation["videos"]}
    require(len(calibration_ids) == 600, "600 unique calibration videos", checks)
    require(len(evaluation_ids) == 21421, "21,421 unique evaluation videos", checks)
    require(not calibration_ids & evaluation_ids, "zero calibration/evaluation overlap", checks)
    require(
        {row["subset"] for row in calibration["videos"]} == {"real"},
        "calibration contains real videos only",
        checks,
    )
    require(
        set(indices["videos"]) == calibration_ids | evaluation_ids,
        "frame-index IDs exactly match both manifests",
        checks,
    )
    expected_windows = sum(len(value) for value in indices["videos"].values())
    require(expected_windows == 58496, "locked manifest contains 58,496 windows", checks)

    raw = load_raw_windows(args.raw_dir, args.num_shards)
    require(len(raw) == expected_windows, "raw window row count matches manifest", checks)
    require(set(raw["video_id"]) == calibration_ids | evaluation_ids, "raw IDs complete", checks)
    frame_lookup = {
        (video_id, window_id): json.dumps(window, separators=(",", ":"))
        for video_id, windows in indices["videos"].items()
        for window_id, window in enumerate(windows)
    }
    actual_frames = {
        (row.video_id, int(row.window_id)): row.frame_indices
        for row in raw[["video_id", "window_id", "frame_indices"]].itertuples(index=False)
    }
    require(actual_frames == frame_lookup, "every raw row uses its locked frame indices", checks)
    calibration_raw = load_calibration_references(
        args.calibration_raw_dir, args.num_shards
    )
    require(
        set(calibration_raw["video_id"]) == calibration_ids,
        "calibration raw IDs match calibration manifest",
        checks,
    )
    verify_calibration_reference_windows(calibration_raw, args.release_dir)
    require(
        True,
        "every calibration raw row uses its locked K1 reference indices",
        checks,
    )
    failure_files = [
        *args.raw_dir.parent.glob("checkpoints/**/failures.csv"),
        *args.raw_dir.parent.glob("calibration_reference_checkpoints/**/failures.csv"),
    ]
    nonempty_failures = []
    for path in failure_files:
        try:
            if len(pd.read_csv(path)):
                nonempty_failures.append(path)
        except pd.errors.EmptyDataError:
            pass
    require(not nonempty_failures, "no decode/scoring failure rows", checks)

    final_path = args.release_dir / "final_video_scores.csv"
    final = pd.read_csv(final_path, float_precision="round_trip")
    require(len(final) == 21421, "final score file has 21,421 rows", checks)
    require(final["video_id"].is_unique, "final score video IDs are unique", checks)
    require(set(final["video_id"]) == evaluation_ids, "final score IDs match evaluation manifest", checks)
    counts = final.groupby("dataset").size().astype(int).to_dict()
    require(counts == EXPECTED_EVALUATION, "final score dataset counts match protocol", checks)
    required_scores = ["G_raw", "L_raw", "G", "L", "S"]
    require(
        np.isfinite(final[required_scores].to_numpy()).all(),
        "all final score fields are finite",
        checks,
    )
    require(
        ((final[["G", "L", "S"]] >= 0) & (final[["G", "L", "S"]] <= 1)).all().all(),
        "calibrated branch and final scores are in [0,1]",
        checks,
    )
    require(
        np.max(
            np.abs(
                final["S"].to_numpy()
                - (0.6 * final["G"].to_numpy() + 0.4 * final["L"].to_numpy())
            )
        )
        <= 1e-15,
        "stored final score exactly matches 0.6G+0.4L",
        checks,
    )

    dataset_metrics, _ = metric_tables(
        final,
        seed=int(config["release"]["random_seed"]),
        score_columns=("S",),
        config_names={"S": "U0_locked"},
    )
    macro = dataset_metrics[dataset_metrics["dataset"] == "Macro-3"].iloc[0]
    metadata = json.loads(
        (args.release_dir / "reproduction_metadata.json").read_text()
    )
    raw_files = sorted(args.raw_dir.glob("*_shard*_of_*.csv"))
    calibration_raw_files = sorted(
        args.calibration_raw_dir.glob("*_shard*_of_*.csv")
    )
    require(
        set(metadata["raw_files"]) == {path.name for path in raw_files},
        "metadata enumerates every K3 raw shard",
        checks,
    )
    require(
        set(metadata["calibration_raw_files"])
        == {path.name for path in calibration_raw_files},
        "metadata enumerates every K1 calibration raw shard",
        checks,
    )
    for path in raw_files:
        require(
            sha256_file(path) == metadata["raw_files"][path.name]["sha256"],
            f"K3 raw SHA256 matches metadata: {path.name}",
            checks,
        )
    for path in calibration_raw_files:
        require(
            sha256_file(path)
            == metadata["calibration_raw_files"][path.name]["sha256"],
            f"K1 calibration raw SHA256 matches metadata: {path.name}",
            checks,
        )
    require(
        abs(float(macro["auc"]) - metadata["macro_auc"]) < 1e-15,
        "recomputed Macro AUC matches reproduction metadata",
        checks,
    )
    require(
        abs(float(macro["ap"]) - metadata["macro_real_positive_ap"]) < 1e-15,
        "recomputed Macro real-positive AP matches reproduction metadata",
        checks,
    )
    for name, item in hashes["release_outputs"].items():
        require(
            sha256_file(ROOT / item["path"]) == item["sha256"],
            f"release output SHA256 matches: {name}",
            checks,
        )

    payload = {
        "schema_version": "u0_release_validation_v1",
        "passed": True,
        "checks": checks,
        "macro_auc": float(macro["auc"]),
        "macro_real_positive_ap": float(macro["ap"]),
    }
    output = args.release_dir / "validation.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"locked U0 release validation passed: checks={len(checks)} "
        f"Macro={macro['auc']:.6f}/{macro['ap']:.6f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/alpha_stalled_u0_locked.yaml"
    )
    parser.add_argument("--release-dir", type=Path, default=ROOT / "release/u0")
    parser.add_argument(
        "--raw-dir", type=Path, default=ROOT / "results/u0_locked_reproduction/raw"
    )
    parser.add_argument(
        "--calibration-raw-dir",
        type=Path,
        default=ROOT / "results/u0_locked_reproduction/calibration_raw",
    )
    parser.add_argument("--num-shards", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
