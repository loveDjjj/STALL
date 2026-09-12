"""Global+Local 方法使用的确定性原始帧解码。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def video_metadata(path):
    """沿既有avg_frame_rate与stream duration规则探测，失败显式拒绝。"""
    import json
    import math
    import subprocess
    from fractions import Fraction

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration,avg_frame_rate",
            "-print_format",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError("视频缺少可用流")
    fps = float(Fraction(streams[0].get("avg_frame_rate", "0/1")))
    duration = float(streams[0].get("duration", 0))
    if not math.isfinite(fps) or not math.isfinite(duration) or fps <= 0 or duration <= 0:
        raise ValueError("视频帧率或时长无效")
    return dict(fps=fps, duration_seconds=duration, num_frames=round(fps * duration))


def downsample_indices(num_frames, current_fps, target_fps=8):
    import math

    if (
        num_frames < 1
        or not math.isfinite(current_fps)
        or not math.isfinite(target_fps)
        or target_fps <= 0
        or current_fps < target_fps
    ):
        raise ValueError("不能复制帧上采样，帧率和帧数必须有效")
    ratio = current_fps / target_fps
    indices = []
    position = 0
    while True:
        index = round(ratio * position)
        if index >= num_frames:
            break
        indices.append(index)
        position += 1
    return indices


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

    当 ``require_all=False`` 时，缺失帧会被忽略；严格缓存生产者使用
    ``require_all=True``，以确保元数据不会声明 tensor 中不存在的帧索引。
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


def decode_bounded(video_path: str | Path, frame_indices: Iterable[int]) -> np.ndarray:
    """严格随机定位失败时顺序恢复，只保留所需帧，保持原协议的BGR和请求次序。"""
    indices = list(frame_indices)
    if not indices:
        raise ValueError("解码索引为空")
    try:
        return decode_indexed_frames(video_path, indices, require_all=True)
    except ValueError:
        capture = _open_video_capture(video_path)
        selected = {}
        targets = set(indices)
        try:
            for index in range(max(indices) + 1):
                ok, frame = capture.read()
                if not ok:
                    break
                if index in targets:
                    selected[index] = frame
        finally:
            capture.release()
        if targets - selected.keys():
            raise ValueError(f"严格解码帧不足：{video_path}")
        return np.stack([selected[index] for index in indices])


__all__ = [
    "decode_all_frames",
    "decode_indexed_frames",
    "load_video_frames",
    "decode_bounded",
]
