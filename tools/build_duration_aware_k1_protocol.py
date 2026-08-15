#!/usr/bin/env python3
"""Build the original fixed-seed K1 window manifest for the full 23-source pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from alpha_stalled.release_io import video_id  # noqa: E402


INDEXES = {
    "comgenvid": ROOT / "cache/indexes/comgenvid.csv",
    "videofeedback": ROOT / "cache/indexes/videofeedback.csv",
    "genvideo": ROOT / "cache/indexes/genvideo.csv",
}
CACHE_ROOTS = {
    dataset: ROOT / f"cache/patch_embeddings/{dataset}" for dataset in INDEXES
}


def k1_task_id(video_id_value: str, duration: int, split: str) -> str:
    value = f"duration-aware-original-k1-v1\0{video_id_value}\0{duration}\0{split}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_window(value: object, frames: int) -> list[int]:
    if pd.isna(value):
        return []
    values = json.loads(str(value)) if isinstance(value, str) else list(value)
    output = [int(item) for item in values]
    if len(output) != frames or len(set(output)) != frames:
        return []
    return output


def load_index_lookup() -> pd.DataFrame:
    frames = []
    for dataset, path in INDEXES.items():
        frame = pd.read_csv(path, float_precision="round_trip")
        frame["dataset"] = dataset
        frame["filename"] = frame["video_path"].map(lambda value: Path(str(value)).name)
        frame["video_id"] = frame.apply(video_id, axis=1)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if result["video_id"].duplicated().any():
        raise ValueError("duplicate video identities in source indexes")
    return result


def compact_cache_status(row: pd.Series, window: list[int]) -> tuple[str, str]:
    path = (
        CACHE_ROOTS[str(row["dataset"])]
        / str(row["subset"])
        / str(row["source_model"])
        / f"{Path(str(row['filename'])).stem}_{int(row['protocol_duration_sec'])}s.pt"
    )
    if not path.is_file():
        return str(path.relative_to(ROOT)), "missing"
    # Exact frame validation happens when a cache is actually consumed. Avoid
    # reading hundreds of gigabytes merely to build the protocol manifest.
    return str(path.relative_to(ROOT)), "exists_unverified"


def existing_raw_windows(raw_dir: Path, custom_dir: Path) -> set[tuple[str, int, tuple[int, ...]]]:
    raw_paths = sorted(raw_dir.glob("*.csv"))
    custom_paths = sorted(custom_dir.glob("*.csv"))
    if not raw_paths or not custom_paths:
        raise FileNotFoundError("duration-aware raw score inputs are incomplete")
    raw = pd.concat(
        [
            pd.read_csv(
                path,
                usecols=["video_id", "protocol_duration_sec", "frame_indices"],
            )
            for path in raw_paths
        ],
        ignore_index=True,
    )
    custom = pd.concat(
        [pd.read_csv(path, usecols=["task_id", "window_id"]) for path in custom_paths],
        ignore_index=True,
    )
    raw_keys = pd.concat(
        [pd.read_csv(path, usecols=["task_id", "window_id"]) for path in raw_paths],
        ignore_index=True,
    )
    if len(raw_keys.merge(custom, on=["task_id", "window_id"])) != len(raw):
        raise ValueError("custom raw scores do not cover all duration-aware windows")
    return {
        (str(row.video_id), int(row.protocol_duration_sec), tuple(json.loads(row.frame_indices)))
        for row in raw.itertuples(index=False)
    }


def run(args: argparse.Namespace) -> None:
    tasks = pd.read_csv(args.tasks, float_precision="round_trip")
    base = tasks[
        tasks["sampling"].eq("k1")
        | tasks["protocol_split"].eq("evaluation")
    ].drop_duplicates(["video_id", "protocol_duration_sec", "protocol_split"])
    indexes = load_index_lookup()
    source_columns = [
        "video_id",
        "dataset",
        "subset",
        "source_model",
        "filename",
        "video_path",
        "duration_seconds",
        "1_sec_idxs",
        "2_sec_idxs",
    ]
    base = base.drop(columns=["video_path", "duration_seconds"]).merge(
        indexes[source_columns],
        on=["video_id", "dataset", "subset", "source_model", "filename"],
        validate="many_to_one",
    )
    existing = existing_raw_windows(args.raw_dir, args.custom_raw_dir)
    rows = []
    for _, source in base.iterrows():
        duration = int(source["protocol_duration_sec"])
        window = parse_window(source[f"{duration}_sec_idxs"], duration * 8)
        if not window:
            raise ValueError(f"missing original K1 window: {source['video_id']}/{duration}s")
        cache_path, cache_status = compact_cache_status(source, window)
        raw_status = (
            "exact"
            if (str(source["video_id"]), duration, tuple(window)) in existing
            else "missing"
        )
        rows.append(
            {
                "k1_task_id": k1_task_id(str(source["video_id"]), duration, str(source["protocol_split"])),
                "video_id": str(source["video_id"]),
                "dataset": str(source["dataset"]),
                "protocol_split": str(source["protocol_split"]),
                "subset": str(source["subset"]),
                "source_model": str(source["source_model"]),
                "filename": str(source["filename"]),
                "video_path": str(source["video_path"]),
                "duration_seconds": float(source["duration_seconds"]),
                "protocol_duration_sec": duration,
                "sampling": "original_fixed_seed_k1",
                "frame_indices": json.dumps(window, separators=(",", ":")),
                "existing_k3_raw_status": raw_status,
                "compact_cache_path": cache_path,
                "compact_cache_status": cache_status,
            }
        )
    output = pd.DataFrame(rows).sort_values(
        ["dataset", "protocol_split", "subset", "source_model", "filename", "protocol_duration_sec"]
    )
    if output["k1_task_id"].duplicated().any():
        raise ValueError("duplicate K1 task identities")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    summary = (
        output.groupby(
            ["dataset", "protocol_split", "existing_k3_raw_status", "compact_cache_status"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "tasks"})
    )
    summary.to_csv(args.summary, index=False)
    print(summary.to_string(index=False))
    print(f"wrote {len(output)} K1 tasks -> {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=ROOT / "results/duration_aware_23source/protocol_tasks.csv",
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
        "--output",
        type=Path,
        default=ROOT / "results/full_coverage_paper_protocol/original_k1_tasks.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "results/full_coverage_paper_protocol/original_k1_cache_audit.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
