"""Protocol construction for the duration-aware 23-source extension.

This module owns stable data identities, calibration-size design, and task
construction.  Executable tools may re-export these names for compatibility,
but protocol consumers must import them from here instead of another CLI.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .release_io import (
    REPOSITORY_ROOT,
    repository_relative,
    resolve_video,
    video_id,
)
from .sampling import parse_indices, uniform_windows


DATASET_SPECS = {
    "comgenvid": {
        "index": REPOSITORY_ROOT / "cache/indexes/comgenvid.csv",
        "calibration": REPOSITORY_ROOT / "cache/indexes/comgenvid_calib_real800.csv",
        "calibration_size": 800,
    },
    "videofeedback": {
        "index": REPOSITORY_ROOT / "cache/indexes/videofeedback.csv",
        "calibration": REPOSITORY_ROOT
        / "cache/indexes/videofeedback_small_calib_real500.csv",
        "calibration_size": 500,
    },
    "genvideo": {
        "index": REPOSITORY_ROOT / "cache/indexes/genvideo.csv",
        "calibration": REPOSITORY_ROOT / "cache/indexes/genvideo_calib_real2000.csv",
        "calibration_size": 2000,
    },
}

# Historical tools and result files use DATASETS for this mapping.  Keep the
# alias while exposing a less ambiguous name to new consumers.
DATASETS = DATASET_SPECS

N200_INDEX = {
    "comgenvid": REPOSITORY_ROOT / "cache/indexes/comgenvid_calib_real200.csv",
    "videofeedback": REPOSITORY_ROOT
    / "cache/indexes/videofeedback_small_calib_real200.csv",
    "genvideo": REPOSITORY_ROOT / "cache/indexes/genvideo_calib_real200.csv",
}
SHORT_DATASETS = frozenset({"videofeedback", "genvideo"})
CALIBRATION_SIZES = {
    "comgenvid": (200, 400, 600, 800),
    "videofeedback": (200, 300, 400, 500),
    "genvideo": (200, 500, 1000, 1500, 2000),
}
EXISTING_SIZE_INDEX = {
    ("comgenvid", 200): N200_INDEX["comgenvid"],
    ("comgenvid", 400): REPOSITORY_ROOT
    / "cache/indexes/comgenvid_calib_real400.csv",
    ("comgenvid", 800): DATASET_SPECS["comgenvid"]["calibration"],
    ("videofeedback", 200): N200_INDEX["videofeedback"],
    ("videofeedback", 500): DATASET_SPECS["videofeedback"]["calibration"],
    ("genvideo", 200): N200_INDEX["genvideo"],
    ("genvideo", 500): REPOSITORY_ROOT
    / "cache/indexes/genvideo_calib_real500.csv",
    ("genvideo", 1000): REPOSITORY_ROOT
    / "cache/indexes/genvideo_calib_real1000.csv",
    ("genvideo", 2000): DATASET_SPECS["genvideo"]["calibration"],
}


def load_k3_exclusions(
    path: Path | None = None,
) -> dict[tuple[str, str, str, str], set[tuple[int, ...]]]:
    """Load declared K3 decode exclusions keyed by canonical video identity."""

    source = path or REPOSITORY_ROOT / "configs/multi_window_exclusions.csv"
    frame = pd.read_csv(source)
    frame = frame[frame["sampling"].eq("K3_uniform")]
    output: dict[tuple[str, str, str, str], set[tuple[int, ...]]] = {}
    for row in frame.itertuples(index=False):
        key = (
            str(row.dataset),
            str(row.subset),
            str(row.source_model),
            str(row.filename),
        )
        output.setdefault(key, set()).add(tuple(parse_indices(row.frame_indices)))
    return output


K3_EXCLUSIONS = load_k3_exclusions()


def stable_size_rank(dataset: str, video_id_value: str) -> str:
    """Return the frozen deterministic rank for nested calibration curves."""

    return hashlib.sha256(
        f"duration-aware-calibration-size-v1\0{dataset}\0{video_id_value}".encode(
            "utf-8"
        )
    ).hexdigest()


def proportional_counts(frame: pd.DataFrame, size: int) -> dict[str, int]:
    """Allocate an exact target size proportionally across real sources."""

    counts = frame["source_model"].value_counts().sort_index()
    exact = counts * (float(size) / float(counts.sum()))
    allocated = exact.astype(int)
    remainder = size - int(allocated.sum())
    order = sorted(
        counts.index,
        key=lambda key: (-(exact[key] - allocated[key]), key),
    )
    for key in order[:remainder]:
        allocated[key] += 1
    return allocated.astype(int).to_dict()


def add_identity(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Attach the canonical dataset, filename, and video identity columns."""

    output = frame.copy()
    output["dataset"] = dataset
    output["filename"] = output["video_path"].map(
        lambda value: Path(str(value)).name
    )
    output["video_id"] = output.apply(video_id, axis=1)
    return output


def calibration_size_splits(
    dataset: str, maximum: pd.DataFrame
) -> dict[int, set[str]]:
    """Build the frozen nested real-only calibration-size memberships."""

    sizes = CALIBRATION_SIZES[dataset]
    maximum_ids = set(maximum["video_id"])
    splits: dict[int, set[str]] = {}
    previous: set[str] = set()
    for size in sizes:
        existing = EXISTING_SIZE_INDEX.get((dataset, size))
        if existing is not None:
            frame = add_identity(
                pd.read_csv(existing, float_precision="round_trip"), dataset
            )
            selected = set(frame["video_id"])
            if len(selected) != size or not selected <= maximum_ids:
                raise ValueError(
                    f"{dataset}/N={size}: invalid existing calibration index"
                )
        else:
            selected = set(previous)
            allocation = proportional_counts(maximum, size)
            for source_model, target_count in allocation.items():
                current = maximum[
                    maximum["source_model"].eq(source_model)
                    & maximum["video_id"].isin(selected)
                ]["video_id"].nunique()
                needed = target_count - int(current)
                if needed < 0:
                    raise ValueError(
                        f"{dataset}/N={size}: previous source allocation exceeds target"
                    )
                candidates = maximum[
                    maximum["source_model"].eq(source_model)
                    & ~maximum["video_id"].isin(selected)
                ].copy()
                candidates["_rank"] = candidates["video_id"].map(
                    lambda value: stable_size_rank(dataset, str(value))
                )
                selected.update(
                    candidates.sort_values("_rank").head(needed)["video_id"]
                )
        if len(selected) != size or not previous <= selected:
            raise ValueError(f"{dataset}/N={size}: calibration sizes are not nested")
        splits[size] = selected
        previous = selected
    if splits[sizes[-1]] != maximum_ids:
        raise ValueError(
            f"{dataset}: largest size does not equal maximum calibration pool"
        )
    return splits


def task_id(
    video_id_value: str, duration_sec: int, sampling: str, split: str
) -> str:
    """Build a stable identity for one duration/sampling protocol task."""

    value = f"{video_id_value}\0{duration_sec}\0{sampling}\0{split}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_window(value: object, frames: int) -> list[int]:
    """Parse a valid distinct-frame window or return an empty sentinel."""

    if pd.isna(value):
        return []
    window = parse_indices(value)
    if len(window) != frames or len(set(window)) != frames:
        return []
    return window


def record(
    row: pd.Series,
    split: str,
    duration_sec: int,
    sampling: str,
    windows: list[list[int]],
) -> dict:
    """Serialize one validated protocol task."""

    if not windows:
        raise ValueError(f"empty windows for {row['video_id']}")
    expected_frames = duration_sec * 8
    if any(
        len(window) != expected_frames or len(set(window)) != expected_frames
        for window in windows
    ):
        raise ValueError(f"invalid {duration_sec}s windows for {row['video_id']}")
    return {
        "task_id": task_id(str(row["video_id"]), duration_sec, sampling, split),
        "video_id": str(row["video_id"]),
        "dataset": str(row["dataset"]),
        "protocol_split": split,
        "subset": str(row["subset"]),
        "source_model": str(row["source_model"]),
        "filename": str(row["filename"]),
        "video_path": repository_relative(str(row["video_path"])),
        "duration_seconds": float(row["duration_seconds"]),
        "protocol_duration_sec": duration_sec,
        "sampling": sampling,
        "effective_k": len(windows),
        "frame_indices": json.dumps(windows, separators=(",", ":")),
    }


def duration_two_windows(row: pd.Series) -> list[list[int]]:
    """Build valid duration-aware K3 windows after declared exclusions."""

    if pd.isna(row["downsample_idxs"]):
        return []
    windows = uniform_windows(parse_indices(row["downsample_idxs"]), 3)
    key = (
        str(row["dataset"]),
        str(row["subset"]),
        str(row["source_model"]),
        str(row["filename"]),
    )
    excluded = K3_EXCLUSIONS.get(key, set())
    return [window for window in windows if tuple(window) not in excluded]


def build_dataset(dataset: str, spec: dict) -> tuple[pd.DataFrame, dict]:
    """Build all calibration and evaluation tasks for one canonical dataset."""

    full = add_identity(
        pd.read_csv(spec["index"], float_precision="round_trip"), dataset
    )
    calibration_index = add_identity(
        pd.read_csv(spec["calibration"], float_precision="round_trip"), dataset
    )
    calibration = calibration_index[calibration_index["subset"].eq("real")].copy()
    if len(calibration) != spec["calibration_size"]:
        raise ValueError(f"{dataset}: invalid calibration count {len(calibration)}")
    calibration_ids = set(calibration["video_id"])
    if calibration["video_id"].duplicated().any():
        raise ValueError(f"{dataset}: duplicate calibration identities")

    evaluation = full[~full["video_id"].isin(calibration_ids)].copy()
    expected_real = int((full["subset"] == "real").sum()) - len(calibration)
    if int((evaluation["subset"] == "real").sum()) != expected_real:
        raise ValueError(
            f"{dataset}: real calibration/evaluation partition is incomplete"
        )
    if int((evaluation["subset"] == "annotated").sum()) != int(
        (full["subset"] == "annotated").sum()
    ):
        raise ValueError(f"{dataset}: generated evaluation videos were dropped")

    rows: list[dict] = []
    for _, source in calibration.iterrows():
        one_2s = parse_window(source["2_sec_idxs"], 16)
        k3 = duration_two_windows(source)
        rows.append(record(source, "calibration", 2, "k1", [one_2s]))
        rows.append(record(source, "calibration", 2, "k3_uniform", k3))
        if dataset in SHORT_DATASETS:
            one_1s = parse_window(source["1_sec_idxs"], 8)
            rows.append(record(source, "calibration", 1, "k1", [one_1s]))

    excluded_real_too_short = 0
    for _, source in evaluation.iterrows():
        if source["subset"] == "real":
            k3 = duration_two_windows(source)
            if not k3:
                excluded_real_too_short += 1
                continue
            rows.append(record(source, "evaluation", 2, "k3_uniform", k3))
            if dataset in SHORT_DATASETS:
                one_1s = parse_window(source["1_sec_idxs"], 8)
                rows.append(record(source, "evaluation", 1, "k1", [one_1s]))
            continue

        k3 = duration_two_windows(source)
        if k3:
            rows.append(record(source, "evaluation", 2, "k3_uniform", k3))
        else:
            one_1s = parse_window(source["1_sec_idxs"], 8)
            if not one_1s:
                raise ValueError(
                    f"{dataset}: generated video supports neither 1s nor 2s: "
                    f"{source['video_id']}"
                )
            rows.append(record(source, "evaluation", 1, "k1", [one_1s]))

    tasks = pd.DataFrame(rows)
    summary = {
        "calibration_real": len(calibration),
        "evaluation_real": expected_real - excluded_real_too_short,
        "excluded_real_too_short": excluded_real_too_short,
        "evaluation_fake": int((evaluation["subset"] == "annotated").sum()),
        "physical_evaluation_videos": len(evaluation) - excluded_real_too_short,
        "evaluation_tasks": int((tasks["protocol_split"] == "evaluation").sum()),
        "calibration_tasks": int((tasks["protocol_split"] == "calibration").sum()),
    }
    return tasks, summary


def validate(tasks: pd.DataFrame) -> None:
    """Validate task identities, split isolation, windows, and source files."""

    if tasks["task_id"].duplicated().any():
        raise ValueError("duplicate task IDs")
    calibration_ids = set(
        tasks[tasks["protocol_split"].eq("calibration")]["video_id"]
    )
    evaluation_ids = set(
        tasks[tasks["protocol_split"].eq("evaluation")]["video_id"]
    )
    if calibration_ids & evaluation_ids:
        raise ValueError("calibration/evaluation identity overlap")
    if set(tasks[tasks["protocol_split"].eq("calibration")]["subset"]) != {
        "real"
    }:
        raise ValueError("calibration contains generated videos")
    for row in tasks.itertuples(index=False):
        windows = json.loads(row.frame_indices)
        expected = int(row.protocol_duration_sec) * 8
        if len(windows) != int(row.effective_k):
            raise ValueError(f"effective_k mismatch: {row.task_id}")
        if any(
            len(window) != expected or len(set(window)) != expected
            for window in windows
        ):
            raise ValueError(f"frame protocol mismatch: {row.task_id}")
        if not resolve_video(str(row.video_path)).is_file():
            raise FileNotFoundError(row.video_path)


__all__ = [
    "CALIBRATION_SIZES",
    "DATASETS",
    "DATASET_SPECS",
    "EXISTING_SIZE_INDEX",
    "K3_EXCLUSIONS",
    "N200_INDEX",
    "SHORT_DATASETS",
    "add_identity",
    "build_dataset",
    "calibration_size_splits",
    "duration_two_windows",
    "load_k3_exclusions",
    "parse_window",
    "proportional_counts",
    "record",
    "stable_size_rank",
    "task_id",
    "validate",
]
