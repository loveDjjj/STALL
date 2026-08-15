#!/usr/bin/env python3
"""Lock the confirmatory GenVidBench Pair1 calibration and evaluation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from alpha_stalled.release_io import resolve_video, sha256_file, write_json
from alpha_stalled.sampling import uniform_windows


DATASET = "genvidbench_pair1"


def external_video_id(row: pd.Series) -> str:
    identity = "|".join(
        (
            DATASET,
            str(row["protocol_split"]),
            str(row["subset"]),
            str(row["source_model"]),
            str(row["filename"]),
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def cache_path(row: pd.Series, cache_root: Path) -> Path:
    return (
        cache_root
        / str(row["subset"])
        / str(row["source_model"])
        / f"{Path(str(row['filename'])).stem}_2s.pt"
    )


def prepare_rows(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    output = frame.copy()
    output["dataset"] = DATASET
    output["protocol_split"] = split
    output["filename"] = output["video_path"].map(lambda value: Path(str(value)).name)
    output["video_id"] = output.apply(external_video_id, axis=1)
    output["k1_window"] = output["2_sec_idxs"].map(
        lambda value: [int(index) for index in json.loads(str(value))]
    )
    output["k3_windows"] = output["downsample_idxs"].map(
        lambda value: uniform_windows(
            [int(index) for index in json.loads(str(value))], 3
        )
    )
    output["effective_k"] = output["k3_windows"].map(len)
    return output


def validate_rows(frame: pd.DataFrame, cache_root: Path) -> None:
    if frame["video_id"].duplicated().any():
        raise ValueError("external manifest contains duplicate video IDs")
    for row in frame.itertuples(index=False):
        if len(row.k1_window) != 16 or len(set(row.k1_window)) != 16:
            raise ValueError(f"invalid K1 strict window: {row.video_path}")
        if not row.k3_windows:
            raise ValueError(f"no K3 window: {row.video_path}")
        if len({tuple(window) for window in row.k3_windows}) != len(row.k3_windows):
            raise ValueError(f"duplicate K3 window: {row.video_path}")
        if any(len(window) != 16 or len(set(window)) != 16 for window in row.k3_windows):
            raise ValueError(f"invalid K3 strict window: {row.video_path}")
        if not resolve_video(str(row.video_path)).is_file():
            raise FileNotFoundError(row.video_path)
        expected_cache = cache_path(pd.Series(row._asdict()), cache_root)
        if not expected_cache.is_file():
            raise FileNotFoundError(expected_cache)


def manifest_payload(frame: pd.DataFrame, split: str) -> dict:
    videos = []
    for row in frame.sort_values(["subset", "source_model", "filename"]).itertuples(
        index=False
    ):
        videos.append(
            {
                "video_id": row.video_id,
                "dataset": DATASET,
                "protocol_split": split,
                "subset": row.subset,
                "source_model": row.source_model,
                "filename": row.filename,
                "video_path": str(row.video_path),
                "duration_seconds": float(row.duration_seconds),
                "native_fps": float(row.fps),
                "native_frames": int(row.num_frames),
                "effective_k": int(row.effective_k),
            }
        )
    return {
        "schema_version": "u0_external_genvidbench_manifest_v1",
        "dataset": DATASET,
        "protocol_split": split,
        "video_count": len(videos),
        "subset_counts": frame["subset"].value_counts().sort_index().astype(int).to_dict(),
        "source_counts": frame["source_model"].value_counts().sort_index().astype(int).to_dict(),
        "videos": videos,
    }


def probe_t2vz(fake_root: Path) -> dict:
    videos = sorted(fake_root.rglob("*.mp4"))
    fps_values = []
    unreadable = []
    for path in videos:
        capture = cv2.VideoCapture(str(path))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if fps <= 0 or frames <= 0:
            unreadable.append(str(path))
        else:
            fps_values.append(fps)
    return {
        "source_model": "t2vz",
        "file_count": len(videos),
        "readable_count": len(fps_values),
        "unreadable_files": unreadable,
        "native_fps_min": min(fps_values) if fps_values else None,
        "native_fps_median": float(pd.Series(fps_values).median()) if fps_values else None,
        "native_fps_max": max(fps_values) if fps_values else None,
        "eligible_fake_count": 0,
        "exclusion_reason": (
            "native 4 FPS cannot provide 16 distinct frames in a strict 2-second, "
            "8 FPS window without frame duplication or a protocol change"
        ),
    }


def run(args: argparse.Namespace) -> None:
    calibration_source = pd.read_csv(args.calibration_index, float_precision="round_trip")
    ms_source = pd.read_csv(args.ms_index, float_precision="round_trip")
    pika_source = pd.read_csv(args.pika_index, float_precision="round_trip")

    calibration = prepare_rows(calibration_source, "calibration")
    evaluation_source = pd.concat(
        [
            ms_source[ms_source["subset"].eq("real")],
            ms_source[ms_source["subset"].eq("annotated")],
            pika_source[pika_source["subset"].eq("annotated")],
        ],
        ignore_index=True,
    )
    evaluation = prepare_rows(evaluation_source, "evaluation")
    validate_rows(pd.concat([calibration, evaluation], ignore_index=True), args.cache_root)

    calibration_real_names = set(calibration["filename"])
    evaluation_real_names = set(evaluation[evaluation["subset"].eq("real")]["filename"])
    overlap = calibration_real_names & evaluation_real_names
    if overlap:
        raise ValueError(f"calibration/evaluation real filename overlap: {len(overlap)}")
    if len(calibration) != 199:
        raise ValueError(f"expected 199 indexed calibration videos, got {len(calibration)}")
    expected_evaluation = {"real": 300, "annotated": 600}
    actual_evaluation = evaluation["subset"].value_counts().astype(int).to_dict()
    if actual_evaluation != expected_evaluation:
        raise ValueError(f"evaluation counts {actual_evaluation} != {expected_evaluation}")

    all_calibration_files = {
        path.name for path in args.calibration_video_root.rglob("*.mp4")
    }
    indexed_calibration_files = set(calibration["filename"])
    unindexed = sorted(all_calibration_files - indexed_calibration_files)

    frame_indices = {
        row.video_id: row.k3_windows
        for row in pd.concat([calibration, evaluation], ignore_index=True).itertuples(
            index=False
        )
    }
    calibration_reference_windows = {
        row.video_id: row.k1_window for row in calibration.itertuples(index=False)
    }
    write_json(args.output_dir / "calibration_manifest.json", manifest_payload(calibration, "calibration"))
    write_json(args.output_dir / "evaluation_manifest.json", manifest_payload(evaluation, "evaluation"))
    write_json(
        args.output_dir / "frame_indices.json",
        {
            "schema_version": "u0_external_genvidbench_frames_v1",
            "sampling": "locked deterministic uniform K3, strict 2 seconds/8 FPS/16 distinct frames",
            "video_count": len(frame_indices),
            "videos": dict(sorted(frame_indices.items())),
            "calibration_reference_count": len(calibration_reference_windows),
            "calibration_reference_windows": dict(sorted(calibration_reference_windows.items())),
        },
    )

    t2vz = probe_t2vz(args.t2vz_fake_root)
    audit = {
        "schema_version": "u0_external_genvidbench_audit_v1",
        "locked_before_u0_metrics": True,
        "calibration_indexed_real": len(calibration),
        "calibration_directory_files": len(all_calibration_files),
        "unindexed_calibration_files": unindexed,
        "evaluation_unique_real": int(evaluation["subset"].eq("real").sum()),
        "evaluation_fake_by_generator": evaluation[
            evaluation["subset"].eq("annotated")
        ]["source_model"].value_counts().sort_index().astype(int).to_dict(),
        "calibration_evaluation_overlap_count": 0,
        "t2vz": t2vz,
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (args.calibration_index, args.ms_index, args.pika_index)
        },
    }
    write_json(args.output_dir / "data_audit.json", audit)
    print(
        f"locked external GenVidBench: calibration={len(calibration)} "
        f"evaluation={len(evaluation)} effective_k={evaluation.effective_k.value_counts().sort_index().to_dict()} "
        f"unindexed_calibration={len(unindexed)} t2vz_excluded={t2vz['file_count']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-index",
        type=Path,
        default=ROOT / "cache/indexes/genvidbench_pair1_ms_vript_calib.csv",
    )
    parser.add_argument(
        "--ms-index",
        type=Path,
        default=ROOT / "cache/indexes/genvidbench_pair1_ms_vript_eval.csv",
    )
    parser.add_argument(
        "--pika-index",
        type=Path,
        default=ROOT / "cache/indexes/genvidbench_pair1_pika_vript_eval.csv",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "cache/patch_embeddings/genvidbench_pair1_ms_vript",
    )
    parser.add_argument(
        "--calibration-video-root",
        type=Path,
        default=ROOT / "datasets/genvidbench_pair1_ms_vript_calib",
    )
    parser.add_argument(
        "--t2vz-fake-root",
        type=Path,
        default=ROOT / "datasets/genvidbench_pair1_t2vz_vript_eval/fake",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "release/u0_external_genvidbench",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
