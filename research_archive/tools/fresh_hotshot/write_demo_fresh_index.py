#!/usr/bin/env python3
"""Write an enriched demo_dataset index from the resolved index preview."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.video_index import compute_windows, downsample_frames, get_video_metadata


def _json_or_none(value: object) -> str | None:
    return json.dumps(value) if value is not None else None


def build_enriched_index(
    index_preview_csv: Path,
    output_csv: Path,
    target_fps: float = 8.0,
    root: Path = Path("."),
) -> pd.DataFrame:
    preview = pd.read_csv(index_preview_csv)
    required = ["video_path", "subset", "source_model"]
    missing = [col for col in required if col not in preview.columns]
    if missing:
        raise ValueError(f"{index_preview_csv} missing columns: {missing}")

    rows: list[dict[str, object]] = []
    for _, item in preview.iterrows():
        rel_video_path = str(item["video_path"])
        video_path = root / rel_video_path
        meta = get_video_metadata(str(video_path))
        fps = meta["fps"]
        duration_seconds = meta["duration_seconds"]
        num_frames = meta["num_frames"]
        if fps is None or duration_seconds is None or num_frames is None:
            warnings.warn(f"Skipping {rel_video_path}: ffprobe metadata failed")
            continue
        if fps < target_fps or duration_seconds < 1.0:
            warnings.warn(
                f"Skipping {rel_video_path}: fps={fps}, duration={duration_seconds}"
            )
            continue

        downsample_idxs = downsample_frames(int(num_frames), float(fps), target_fps)
        windows = compute_windows(downsample_idxs, target_fps=target_fps)
        row = {
            "video_path": rel_video_path,
            "subset": str(item["subset"]),
            "source_model": str(item["source_model"]),
            "fps": fps,
            "duration_seconds": duration_seconds,
            "num_frames": num_frames,
            "downsample_idxs": json.dumps(downsample_idxs),
        }
        for dur_sec in (1, 2, 3, 4):
            row[f"{dur_sec}_sec_idxs"] = _json_or_none(windows[f"{dur_sec}_sec_idxs"])
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No demo videos survived metadata filtering")

    cols = [
        "video_path",
        "subset",
        "source_model",
        "fps",
        "duration_seconds",
        "num_frames",
        "downsample_idxs",
        "1_sec_idxs",
        "2_sec_idxs",
        "3_sec_idxs",
        "4_sec_idxs",
    ]
    out = out[cols]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-preview-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--target-fps", type=float, default=8.0)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()

    out = build_enriched_index(
        index_preview_csv=args.index_preview_csv,
        output_csv=args.output_csv,
        target_fps=args.target_fps,
        root=args.root,
    )
    print(f"Saved enriched demo index ({len(out)} rows) -> {args.output_csv}")


if __name__ == "__main__":
    main()
