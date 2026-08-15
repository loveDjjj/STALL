"""Shared task identity and decoding for duration-aware scoring tools."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import pandas as pd

from .release_io import resolve_required_video
from .video_io import decode_selected_frames


MAX_CALIBRATION_SIZE = {
    "comgenvid": 800,
    "videofeedback": 500,
    "genvideo": 2000,
}

# Compatibility alias used by the original duration-aware scorer CLI.
MAX_SIZE = MAX_CALIBRATION_SIZE

KEY_COLUMNS = [
    "task_id",
    "video_id",
    "dataset",
    "protocol_split",
    "subset",
    "source_model",
    "filename",
    "protocol_duration_sec",
    "sampling",
]
WINDOW_KEYS = ["task_id", "window_id"]


def decode_video(
    rows: pd.DataFrame,
    seek_gap: int,
    attempts: int,
    *,
    decoder: Callable[[Any, list[int], int], list] | None = None,
    resolver: Callable[[str], Any] | None = None,
    retry_delay: float = 0.25,
) -> dict:
    """Decode the union of frames for all duration tasks of one video.

    ``decoder`` and ``resolver`` are injectable only for deterministic tests;
    production callers use the shared strict video resolver and native-frame
    decoder.
    """

    if rows.empty:
        raise ValueError("cannot decode an empty duration-aware task group")
    if attempts < 1:
        raise ValueError("decode attempts must be positive")

    tasks = []
    for row in rows.to_dict("records"):
        windows = [
            [int(value) for value in window]
            for window in json.loads(row["frame_indices"])
        ]
        if len(windows) != int(row["effective_k"]):
            raise ValueError(f"effective_k mismatch: {row['task_id']}")
        for window_id, window in enumerate(windows):
            tasks.append((row, window_id, window))
    if not tasks:
        raise ValueError("duration-aware task group contains no windows")

    unique = sorted({index for _, _, window in tasks for index in window})
    decode = decoder or decode_selected_frames
    resolve = resolver or resolve_required_video
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            frames = decode(
                resolve(str(rows.iloc[0]["video_path"])), unique, seek_gap
            )
            positions = {value: index for index, value in enumerate(unique)}
            return {
                "rows": rows,
                "tasks": tasks,
                "positions": [
                    [positions[value] for value in window]
                    for _, _, window in tasks
                ],
                "frames": frames,
                "unique_indices": unique,
            }
        except Exception as caught:
            error = caught
            if attempt + 1 < attempts:
                time.sleep(retry_delay * (attempt + 1))
    raise RuntimeError(
        f"decode failed after {attempts} attempts: {error}"
    ) from error


__all__ = [
    "KEY_COLUMNS",
    "MAX_CALIBRATION_SIZE",
    "MAX_SIZE",
    "WINDOW_KEYS",
    "decode_video",
]
