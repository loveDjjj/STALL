"""Locked U0 manifest, shard, and calibration-reference protocol primitives."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .aggregation import selected_video_means
from .artifacts import read_expected_shards
from .release_io import video_id


U0_DATASETS = ("comgenvid", "videofeedback", "genvideo")
EXPECTED_EVALUATION = {
    "comgenvid": 4298,
    "videofeedback": 3500,
    "genvideo": 13623,
}
KEY_COLUMNS = (
    "video_id",
    "dataset",
    "protocol_split",
    "subset",
    "source_model",
    "filename",
)
WINDOW_KEYS = (*KEY_COLUMNS, "window_id")
FINITE_RAW_COLUMNS = (
    "global_spatial_raw",
    "patch_spatial_raw",
    "patch_d2_raw",
)


def load_release_rows(
    release_dir: Path,
) -> tuple[pd.DataFrame, dict[str, list[list[int]]]]:
    """Load the two locked manifests and their exact K3 frame-index mapping."""

    records: list[dict] = []
    for split in ("calibration", "evaluation"):
        path = release_dir / f"{split}_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol_split") != split:
            raise ValueError(f"{path.name} has wrong protocol_split")
        if payload.get("video_count") != len(payload.get("videos", [])):
            raise ValueError(f"{path.name} video_count does not match videos")
        for row in payload["videos"]:
            if row.get("protocol_split") != split:
                raise ValueError(f"{path.name} row has wrong protocol_split")
            if row.get("video_id") != video_id(row):
                raise ValueError(f"{path.name} video_id does not match identity fields")
            records.append(row)
    frame_payload = json.loads(
        (release_dir / "frame_indices.json").read_text(encoding="utf-8")
    )
    frame_indices = frame_payload["videos"]
    rows = pd.DataFrame(records)
    if rows["video_id"].duplicated().any():
        raise ValueError("duplicate video_id in locked release manifests")
    if set(rows["video_id"]) != set(frame_indices):
        raise ValueError("release manifests and frame-index keys differ")
    return rows, frame_indices


def load_raw_windows(raw_dir: Path, num_shards: int) -> pd.DataFrame:
    """Read the complete locked raw K3 shard set and validate score fields."""

    windows = read_expected_shards(raw_dir, U0_DATASETS, num_shards)
    if windows.duplicated(list(WINDOW_KEYS)).any():
        raise ValueError("duplicate locked raw window keys")
    if not np.isfinite(windows[list(FINITE_RAW_COLUMNS)].to_numpy()).all():
        raise ValueError("non-finite locked raw score")
    # Positive infinity is the declared representation for an all-zero global
    # temporal window and maps to one under right-inclusive CDF calibration.
    if np.isnan(windows["global_t1_raw"].to_numpy()).any():
        raise ValueError("NaN global temporal raw score")
    return windows.sort_values(list(WINDOW_KEYS)).reset_index(drop=True)


def load_calibration_references(raw_dir: Path, num_shards: int) -> pd.DataFrame:
    """Read the 600 real, identity-disjoint locked K1 reference windows."""

    references = read_expected_shards(raw_dir, U0_DATASETS, num_shards)
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
    """Require scored K1 reference indices to equal the locked release mapping."""

    payload = json.loads(
        (release_dir / "frame_indices.json").read_text(encoding="utf-8")
    )
    expected = {
        identity: json.dumps(window, separators=(",", ":"))
        for identity, window in payload["calibration_reference_windows"].items()
    }
    actual = dict(zip(references["video_id"], references["frame_indices"]))
    if actual != expected:
        raise ValueError("scored local CDF references differ from locked K1 windows")


def selected_calibration_means(
    calibration_windows: pd.DataFrame, target_k: int
) -> pd.DataFrame:
    """Select deterministic effective-K windows and average calibrated G/L."""

    result = selected_video_means(
        calibration_windows,
        target_k,
        {"G_raw": "G_k", "L_raw": "L_k"},
    )
    if len(result) < 2:
        raise ValueError(f"insufficient effective-K={target_k} calibration videos")
    return result


__all__ = [
    "EXPECTED_EVALUATION",
    "FINITE_RAW_COLUMNS",
    "KEY_COLUMNS",
    "U0_DATASETS",
    "WINDOW_KEYS",
    "load_calibration_references",
    "load_raw_windows",
    "load_release_rows",
    "selected_calibration_means",
    "verify_calibration_reference_windows",
]
