"""Global+Local 方法使用的确定性帧索引采样。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd


WINDOW_FRAMES = 16


def parse_indices(value: object) -> list[int]:
    """Parse a JSON/list-like frame-index value into Python integers."""
    if isinstance(value, str):
        parsed = json.loads(value)
    elif isinstance(value, (list, tuple, np.ndarray)):
        parsed = list(value)
    else:
        raise ValueError(f"invalid frame-index value: {value!r}")
    return [int(item) for item in parsed]


def deduplicate_windows(windows: list[list[int]]) -> list[list[int]]:
    """Keep the first occurrence of each exact frame-index window."""
    unique: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for window in windows:
        key = tuple(window)
        if key not in seen:
            seen.add(key)
            unique.append(window)
    return unique


def uniform_windows(
    downsample_indices: list[int],
    requested_k: int,
    window_frames: int = WINDOW_FRAMES,
) -> list[list[int]]:
    """Return up to K uniformly spaced contiguous windows, deduplicating ties."""
    if requested_k < 1:
        raise ValueError("requested_k must be positive")
    if window_frames < 1:
        raise ValueError("window_frames must be positive")
    max_start = len(downsample_indices) - window_frames
    if max_start < 0:
        return []
    starts = np.rint(np.linspace(0, max_start, requested_k)).astype(int)
    windows = [
        downsample_indices[start : start + window_frames]
        for start in starts.tolist()
    ]
    return deduplicate_windows(windows)


def nonoverlap_windows(
    downsample_indices: list[int],
    window_frames: int = WINDOW_FRAMES,
) -> list[list[int]]:
    """Return complete non-overlapping windows and exclude an incomplete tail."""
    if window_frames < 1:
        raise ValueError("window_frames must be positive")
    if len(downsample_indices) < window_frames:
        return []
    return [
        downsample_indices[start : start + window_frames]
        for start in range(
            0,
            len(downsample_indices) - window_frames + 1,
            window_frames,
        )
    ]


def current_window(
    value: object,
    window_frames: int = WINDOW_FRAMES,
) -> list[list[int]]:
    """Return one frozen current window when it has the required frame count."""
    if window_frames < 1:
        raise ValueError("window_frames must be positive")
    if pd.isna(value):
        return []
    window = parse_indices(value)
    return [window] if len(window) == window_frames else []
