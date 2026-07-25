#!/usr/bin/env python3
"""Calibrate locked U0 raw windows and write release video scores and metrics."""

from __future__ import annotations

import argparse
import hashlib
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
for directory in (ROOT / "src", ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from build_multi_order_baselines import metric_tables
from stable_whitening import empirical_cdf_right_inclusive, stable_sorted


KEY_COLUMNS = [
    "video_id",
    "dataset",
    "protocol_split",
    "subset",
    "source_model",
    "filename",
]
WINDOW_KEYS = [*KEY_COLUMNS, "window_id"]
EXPECTED_EVALUATION = {
    "comgenvid": 4298,
    "videofeedback": 3500,
    "genvideo": 13623,
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_raw_windows(raw_dir: Path, num_shards: int) -> pd.DataFrame:
    frames = []
    for dataset in EXPECTED_EVALUATION:
        for shard in range(num_shards):
            path = raw_dir / f"{dataset}_shard{shard:02d}_of_{num_shards:02d}.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            frames.append(pd.read_csv(path, float_precision="round_trip"))
    windows = pd.concat(frames, ignore_index=True)
    if windows.duplicated(WINDOW_KEYS).any():
        raise ValueError("duplicate locked raw window keys")
    if not np.isfinite(
        windows[
            [
                "global_spatial_raw",
                "patch_spatial_raw",
                "patch_d2_raw",
            ]
        ].to_numpy()
    ).all():
        raise ValueError("non-finite locked raw score")
    # Positive infinity is the declared representation for an all-zero global
    # temporal window and is valid under right-inclusive CDF calibration.
    if np.isnan(windows["global_t1_raw"].to_numpy()).any():
        raise ValueError("NaN global temporal raw score")
    return windows.sort_values(WINDOW_KEYS).reset_index(drop=True)


def global_references(config: dict) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(ROOT / config["global_branch"]["params"], allow_pickle=True)
    return (
        stable_sorted(np.max(data["calib_ll_spat"].astype(np.float64), axis=1)),
        stable_sorted(np.min(data["calib_ll_temp"].astype(np.float64), axis=1)),
    )


def cdf_with_positive_infinity(
    values: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Map declared +inf scores to one while rejecting NaN and -inf."""
    scores = np.asarray(values, dtype=np.float64)
    if np.isnan(scores).any() or np.isneginf(scores).any():
        raise ValueError("CDF values contain NaN or negative infinity")
    result = np.ones(len(scores), dtype=np.float64)
    finite = np.isfinite(scores)
    result[finite] = empirical_cdf_right_inclusive(scores[finite], reference)
    return result


def load_calibration_references(raw_dir: Path, num_shards: int) -> pd.DataFrame:
    frames = []
    for dataset in EXPECTED_EVALUATION:
        for shard in range(num_shards):
            path = raw_dir / f"{dataset}_shard{shard:02d}_of_{num_shards:02d}.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            frames.append(pd.read_csv(path, float_precision="round_trip"))
    references = pd.concat(frames, ignore_index=True)
    if len(references) != 600 or references["video_id"].nunique() != 600:
        raise ValueError(
            f"expected 600 unique calibration references, found rows={len(references)} "
            f"videos={references['video_id'].nunique()}"
        )
    if set(references["protocol_split"]) != {"calibration"}:
        raise ValueError("local CDF references must be calibration videos")
    if set(references["subset"]) != {"real"}:
        raise ValueError("local CDF references must contain real videos only")
    if set(references["effective_k"].astype(int)) != {1}:
        raise ValueError("local CDF references must each contain exactly one K1 window")
    return references


def verify_calibration_reference_windows(
    references: pd.DataFrame, release_dir: Path
) -> None:
    payload = json.loads((release_dir / "frame_indices.json").read_text())
    expected = {
        video_id: json.dumps(window, separators=(",", ":"))
        for video_id, window in payload["calibration_reference_windows"].items()
    }
    actual = dict(zip(references["video_id"], references["frame_indices"]))
    if actual != expected:
        raise ValueError("scored local CDF references differ from locked K1 windows")


def calibrate_windows(
    windows: pd.DataFrame,
    calibration_references: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_spatial_ref, global_t1_ref = global_references(config)
    output = []
    references = []
    for dataset, frame in windows.groupby("dataset", sort=False):
        target = frame.copy()
        calibration = calibration_references[
            calibration_references["dataset"] == dataset
        ]
        if len(calibration) != 200:
            raise ValueError(f"{dataset}: expected 200 local calibration windows")
        patch_spatial_ref = stable_sorted(calibration["patch_spatial_raw"].to_numpy())
        patch_d2_ref = stable_sorted(calibration["patch_d2_raw"].to_numpy())
        target["global_spatial"] = empirical_cdf_right_inclusive(
            target["global_spatial_raw"].to_numpy(), global_spatial_ref
        )
        target["global_t1"] = cdf_with_positive_infinity(
            target["global_t1_raw"].to_numpy(), global_t1_ref
        )
        target["patch_spatial"] = empirical_cdf_right_inclusive(
            target["patch_spatial_raw"].to_numpy(), patch_spatial_ref
        )
        target["patch_d2"] = empirical_cdf_right_inclusive(
            target["patch_d2_raw"].to_numpy(), patch_d2_ref
        )
        target["G_k"] = 0.5 * target["global_spatial"] + 0.5 * target["global_t1"]
        target["L_k"] = 0.1 * target["patch_spatial"] + 0.9 * target["patch_d2"]
        target["S_k"] = 0.6 * target["G_k"] + 0.4 * target["L_k"]
        output.append(target)
        for branch, values in (
            ("patch_spatial", patch_spatial_ref),
            ("patch_d2", patch_d2_ref),
        ):
            references.extend(
                {
                    "dataset": dataset,
                    "branch": branch,
                    "rank": rank,
                    "raw_score": value,
                }
                for rank, value in enumerate(values)
            )
    return pd.concat(output, ignore_index=True), pd.DataFrame(references)


def selected_calibration_means(
    calibration_windows: pd.DataFrame, target_k: int
) -> pd.DataFrame:
    rows = []
    for video_id, frame in calibration_windows.groupby("video_id", sort=False):
        ordered = frame.sort_values("window_id")
        if len(ordered) < target_k:
            continue
        positions = (
            np.array([(len(ordered) - 1) // 2], dtype=int)
            if target_k == 1
            else np.rint(np.linspace(0, len(ordered) - 1, target_k)).astype(int)
        )
        selected = ordered.iloc[np.unique(positions)]
        if len(selected) != target_k:
            continue
        rows.append(
            {
                "video_id": video_id,
                "G_raw": float(selected["G_k"].mean()),
                "L_raw": float(selected["L_k"].mean()),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) < 2:
        raise ValueError(f"insufficient effective-K={target_k} calibration videos")
    return result


def aggregate_and_calibrate_videos(windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_video = (
        windows.groupby(KEY_COLUMNS, sort=False, observed=True)
        .agg(
            video_path=("video_path", "first"),
            duration_seconds=("duration_seconds", "first"),
            effective_k=("effective_k", "first"),
            window_count=("window_id", "size"),
            G_raw=("G_k", "mean"),
            L_raw=("L_k", "mean"),
        )
        .reset_index()
    )
    if not (per_video["effective_k"] == per_video["window_count"]).all():
        raise ValueError("effective_k does not match raw window count")
    frames = []
    reference_rows = []
    for dataset, dataset_videos in per_video.groupby("dataset", sort=False):
        dataset_windows = windows[windows["dataset"] == dataset]
        calibration_windows = dataset_windows[
            dataset_windows["protocol_split"] == "calibration"
        ]
        evaluation = dataset_videos[
            dataset_videos["protocol_split"] == "evaluation"
        ].copy()
        for effective_k, target in evaluation.groupby("effective_k", sort=True):
            reference = selected_calibration_means(
                calibration_windows, int(effective_k)
            )
            g_reference = stable_sorted(reference["G_raw"].to_numpy())
            l_reference = stable_sorted(reference["L_raw"].to_numpy())
            target = target.copy()
            target["G"] = empirical_cdf_right_inclusive(
                target["G_raw"].to_numpy(), g_reference
            )
            target["L"] = empirical_cdf_right_inclusive(
                target["L_raw"].to_numpy(), l_reference
            )
            target["S"] = 0.6 * target["G"] + 0.4 * target["L"]
            frames.append(target)
            reference_rows.append(
                {
                    "dataset": dataset,
                    "effective_k": int(effective_k),
                    "calibration_videos": len(reference),
                    "global_min": float(g_reference.min()),
                    "global_max": float(g_reference.max()),
                    "local_min": float(l_reference.min()),
                    "local_max": float(l_reference.max()),
                }
            )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(reference_rows)


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
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
        "config": str(args.config.relative_to(ROOT)),
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
