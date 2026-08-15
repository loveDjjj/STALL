"""Deterministic native-frame decoding shared by Alpha-STALLED scorers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def _open_video_capture(video_path: str | Path) -> cv2.VideoCapture:
    """Open with explicit FFMPEG first, then the platform default backend."""

    capture = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(str(video_path))
    if capture.isOpened():
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def decode_all_frames(video_path: str | Path, *, require_open: bool = False) -> np.ndarray:
    """Sequentially decode a complete video in native BGR order."""

    capture = _open_video_capture(video_path)
    if not capture.isOpened():
        capture.release()
        if require_open:
            raise ValueError(f"cannot open video: {video_path}")
        return np.array([])
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    return np.asarray(frames)


def decode_indexed_frames(
    video_path: str | Path,
    frame_indices: Iterable[int],
    *,
    require_all: bool = False,
) -> np.ndarray:
    """Random-access frames while preserving requested order and duplicates.

    With ``require_all=False`` this exactly preserves the original STALL cache
    behavior: missing frames are omitted. Strict cache producers set
    ``require_all=True`` so metadata can never claim indices absent from the
    tensor payload.
    """

    requested = [int(index) for index in frame_indices]
    if any(index < 0 for index in requested):
        raise ValueError("frame indices must be non-negative")
    if not requested:
        return np.array([])
    capture = _open_video_capture(video_path)
    if not capture.isOpened():
        capture.release()
        if require_all:
            raise ValueError(f"cannot open video: {video_path}")
        return np.array([])
    decoded: dict[int, np.ndarray] = {}
    try:
        for index in sorted(set(requested)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if ok:
                decoded[index] = frame
    finally:
        capture.release()
    missing = [index for index in requested if index not in decoded]
    if require_all and missing:
        unique_missing = list(dict.fromkeys(missing))
        raise ValueError(
            f"missing {len(unique_missing)} requested frames, first={unique_missing[:5]}"
        )
    return np.asarray([decoded[index] for index in requested if index in decoded])


def load_video_frames(video_path: str | Path, frame_indices=None) -> np.ndarray:
    """Original STALL-compatible full or indexed video loader."""

    if frame_indices is None:
        return decode_all_frames(video_path, require_open=False)
    return decode_indexed_frames(video_path, frame_indices, require_all=False)


def decode_spans(targets: list[int], seek_gap: int) -> list[tuple[int, int]]:
    """Group sorted target indices into nearby sequential decode spans."""
    if not targets:
        return []
    spans: list[tuple[int, int]] = []
    start = previous = targets[0]
    for index in targets[1:]:
        if index - previous > seek_gap:
            spans.append((start, previous))
            start = index
        previous = index
    spans.append((start, previous))
    return spans


def decode_selected_frames(
    video_path: str | Path,
    frame_indices: Iterable[int],
    seek_gap: int = 64,
) -> np.ndarray:
    """Decode sorted unique native frames with one pass per nearby span."""
    targets = sorted(set(int(index) for index in frame_indices))
    if not targets:
        raise ValueError("frame_indices is empty")
    if targets[0] < 0:
        raise ValueError("frame indices must be non-negative")
    if seek_gap < 0:
        raise ValueError("seek_gap must be non-negative")
    target_set = set(targets)
    decoded: dict[int, np.ndarray] = {}
    cap = _open_video_capture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    try:
        for start, end in decode_spans(targets, seek_gap):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for index in range(start, end + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                if index in target_set:
                    decoded[index] = frame
    finally:
        cap.release()
    missing = [index for index in targets if index not in decoded]
    if missing:
        raise ValueError(f"missing {len(missing)} decoded frames, first={missing[:5]}")
    return np.stack([decoded[index] for index in targets])


__all__ = [
    "decode_all_frames",
    "decode_indexed_frames",
    "decode_selected_frames",
    "decode_spans",
    "load_video_frames",
]
