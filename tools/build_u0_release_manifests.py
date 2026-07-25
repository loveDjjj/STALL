#!/usr/bin/env python3
"""Build deterministic immutable manifests for the locked U0 release."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
IDENTITY_COLUMNS = ["dataset", "subset", "source_model", "filename"]
EXPECTED_EVALUATION = {
    "comgenvid": 4298,
    "videofeedback": 3500,
    "genvideo": 13623,
}
EXPECTED_CALIBRATION = {name: 200 for name in EXPECTED_EVALUATION}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def video_id(row: pd.Series) -> str:
    identity = "|".join(str(row[column]) for column in IDENTITY_COLUMNS)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def repository_relative(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        try:
            return str(path.relative_to(ROOT.parent))
        except ValueError:
            return str(path)
    return str(path)


def resolve_video(value: str) -> Path:
    path = Path(value)
    candidates = (
        path,
        ROOT / path,
        ROOT.parent / path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return (ROOT.parent / path).resolve()


def manifest_payload(frame: pd.DataFrame, split: str) -> dict:
    videos = []
    for _, row in frame.sort_values(["dataset", *IDENTITY_COLUMNS[1:]]).iterrows():
        videos.append(
            {
                "video_id": row["video_id"],
                "dataset": row["dataset"],
                "protocol_split": split,
                "subset": row["subset"],
                "source_model": row["source_model"],
                "filename": row["filename"],
                "video_path": repository_relative(str(row["video_path"])),
                "duration_seconds": float(row["duration_seconds"]),
                "effective_k": int(row["effective_K3_uniform"]),
            }
        )
    counts = frame.groupby("dataset", sort=True).size().astype(int).to_dict()
    return {
        "schema_version": "u0_manifest_v1",
        "protocol_split": split,
        "video_count": len(videos),
        "dataset_counts": counts,
        "videos": videos,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def git_value(args: list[str], cwd: Path = ROOT) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = pd.read_csv(args.source_manifest, float_precision="round_trip")
    required = {
        *IDENTITY_COLUMNS,
        "protocol_split",
        "video_path",
        "duration_seconds",
        "effective_K3_uniform",
        "indices_K3_uniform",
        "indices_K1_current",
    }
    missing_columns = sorted(required.difference(source.columns))
    if missing_columns:
        raise ValueError(f"source manifest missing columns: {missing_columns}")
    if source.duplicated(["protocol_split", *IDENTITY_COLUMNS]).any():
        raise ValueError("source manifest has duplicate video identities")

    source["video_id"] = source.apply(video_id, axis=1)
    if source["video_id"].duplicated().any():
        raise ValueError("calibration/evaluation identity overlap or duplicate video")

    calibration = source[source["protocol_split"] == "calibration"].copy()
    evaluation = source[source["protocol_split"] == "evaluation"].copy()
    if set(calibration["subset"]) != {"real"}:
        raise ValueError("locked calibration manifest must contain real videos only")
    calibration_counts = calibration.groupby("dataset").size().astype(int).to_dict()
    evaluation_counts = evaluation.groupby("dataset").size().astype(int).to_dict()
    if calibration_counts != EXPECTED_CALIBRATION:
        raise ValueError(f"calibration counts {calibration_counts} != {EXPECTED_CALIBRATION}")
    if evaluation_counts != EXPECTED_EVALUATION:
        raise ValueError(f"evaluation counts {evaluation_counts} != {EXPECTED_EVALUATION}")
    overlap = set(calibration["video_id"]) & set(evaluation["video_id"])
    if overlap:
        raise ValueError(f"calibration/evaluation overlap: {len(overlap)}")

    frame_indices: dict[str, list[list[int]]] = {}
    calibration_reference_windows: dict[str, list[int]] = {}
    invalid = []
    missing_videos = []
    for _, row in source.iterrows():
        windows = json.loads(row["indices_K3_uniform"])
        windows = [[int(index) for index in window] for window in windows]
        reasons = []
        if len(windows) != int(row["effective_K3_uniform"]):
            reasons.append("effective_k_mismatch")
        if not windows or any(len(window) != 16 for window in windows):
            reasons.append("not_exactly_16_frames_per_window")
        if len({tuple(window) for window in windows}) != len(windows):
            reasons.append("duplicate_window")
        if any(len(set(window)) != 16 for window in windows):
            reasons.append("duplicate_frame_within_window")
        if any(any(index < 0 for index in window) for window in windows):
            reasons.append("negative_frame_index")
        if reasons:
            invalid.append({"video_id": row["video_id"], "reasons": reasons})
        frame_indices[row["video_id"]] = windows
        if row["protocol_split"] == "calibration":
            k1_windows = json.loads(row["indices_K1_current"])
            if len(k1_windows) != 1 or len(k1_windows[0]) != 16:
                raise ValueError(
                    f"invalid K1 calibration reference for {row['video_id']}"
                )
            reference = [int(index) for index in k1_windows[0]]
            if len(set(reference)) != 16 or any(index < 0 for index in reference):
                raise ValueError(
                    f"invalid K1 calibration frame indices for {row['video_id']}"
                )
            calibration_reference_windows[row["video_id"]] = reference
        resolved = resolve_video(str(row["video_path"]))
        if not resolved.is_file():
            missing_videos.append(
                {
                    "video_id": row["video_id"],
                    "video_path": str(row["video_path"]),
                }
            )
    if invalid:
        raise ValueError(f"invalid frame-index entries: {len(invalid)}")

    output = args.output_dir
    write_json(output / "calibration_manifest.json", manifest_payload(calibration, "calibration"))
    write_json(output / "evaluation_manifest.json", manifest_payload(evaluation, "evaluation"))
    write_json(
        output / "frame_indices.json",
        {
            "schema_version": "u0_frame_indices_v1",
            "sampling": "deterministic_uniform_K3",
            "video_count": len(frame_indices),
            "videos": dict(sorted(frame_indices.items())),
            "calibration_reference_count": len(calibration_reference_windows),
            "calibration_reference_windows": dict(
                sorted(calibration_reference_windows.items())
            ),
        },
    )
    write_json(
        output / "missing_videos.json",
        {
            "schema_version": "u0_missing_videos_v1",
            "missing_count": len(missing_videos),
            "videos": missing_videos,
        },
    )

    tracked_files = {
        "locked_config": args.config,
        "source_manifest": args.source_manifest,
        "exclusions": ROOT / config["sampling"]["explicit_exclusions"],
        "dino_checkpoint": ROOT / config["backbone"]["checkpoint"],
        "global_params": ROOT / config["global_branch"]["params"],
    }
    for dataset, item in config["local_branch"]["params_by_dataset"].items():
        tracked_files[f"local_params_{dataset}"] = ROOT / item["path"]
    file_hashes = {
        name: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
        for name, path in tracked_files.items()
    }
    expected_hashes = {
        "source_manifest": config["sampling"]["source_manifest_sha256"],
        "exclusions": config["sampling"]["exclusions_sha256"],
        "dino_checkpoint": config["backbone"]["checkpoint_sha256"],
        "global_params": config["global_branch"]["params_sha256"],
        **{
            f"local_params_{dataset}": item["sha256"]
            for dataset, item in config["local_branch"]["params_by_dataset"].items()
        },
    }
    mismatched_hashes = {
        name: {"expected": expected, "actual": file_hashes[name]["sha256"]}
        for name, expected in expected_hashes.items()
        if file_hashes[name]["sha256"] != expected
    }
    if mismatched_hashes:
        raise ValueError(f"locked input hash mismatch: {mismatched_hashes}")

    manifest_paths = (
        output / "calibration_manifest.json",
        output / "evaluation_manifest.json",
        output / "frame_indices.json",
        output / "missing_videos.json",
    )
    release_hashes = {
        path.name: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
        for path in manifest_paths
    }
    worktree_dirty = bool(git_value(["status", "--porcelain"]))
    write_json(
        output / "config_and_checkpoint_hashes.json",
        {
            "schema_version": "u0_hashes_v1",
            "code_git_commit": git_value(["rev-parse", "HEAD"]),
            "code_worktree_dirty_at_lock": worktree_dirty,
            "dinov3_git_commit": git_value(["rev-parse", "HEAD"], ROOT / "dinov3"),
            "input_files": file_hashes,
            "release_manifests": release_hashes,
        },
    )

    print(
        f"wrote locked manifests: calibration={len(calibration)} "
        f"evaluation={len(evaluation)} missing_files={len(missing_videos)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "results/multi_window_joint_typicality/multi_window_manifest.csv",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/alpha_stalled_u0_locked.yaml",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "release/u0"
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
