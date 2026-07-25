"""Deterministic pixel and temporal perturbations for the locked U0 audit."""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


CONDITIONS = (
    "R0_original",
    "R1_h264_crf23",
    "R2_h264_crf35",
    "R3_resize_half_restore",
    "R4_drop10",
    "R5_drop25",
    "R6_repeat10",
    "R7_repeat25",
    "R8_scene_cut",
    "R9_4fps",
)


def _ranked_positions(length: int, count: int, key: str, first: int = 0) -> list[int]:
    if not 0 <= first < length:
        raise ValueError(f"invalid first position {first} for length {length}")
    candidates = range(first, length)
    ranked = sorted(
        candidates,
        key=lambda position: hashlib.sha256(
            f"{key}\0{length}\0{position}".encode("utf-8")
        ).hexdigest(),
    )
    if count > len(ranked):
        raise ValueError(f"requested {count} of {len(ranked)} positions")
    return sorted(ranked[:count])


def deterministic_count(length: int, fraction: float) -> int:
    if length < 1 or not 0.0 < fraction < 1.0:
        raise ValueError("length and fraction must be positive, with fraction below one")
    return max(1, int(np.rint(length * fraction)))


def temporal_perturbation(frames: np.ndarray, condition: str, key: str) -> np.ndarray:
    """Apply a frozen temporal stress operator to one sampled window."""
    values = np.asarray(frames)
    if values.ndim != 4 or len(values) < 3:
        raise ValueError(f"expected [T,H,W,C] with T>=3, got {values.shape}")
    if condition == "R0_original":
        return values.copy()
    if condition in {"R4_drop10", "R5_drop25"}:
        fraction = 0.10 if condition.endswith("10") else 0.25
        count = deterministic_count(len(values), fraction)
        dropped = set(_ranked_positions(len(values), count, f"{key}\0{condition}"))
        return values[[index not in dropped for index in range(len(values))]].copy()
    if condition in {"R6_repeat10", "R7_repeat25"}:
        fraction = 0.10 if condition.endswith("10") else 0.25
        count = deterministic_count(len(values), fraction)
        positions = _ranked_positions(
            len(values), count, f"{key}\0{condition}", first=1
        )
        output = values.copy()
        for position in positions:
            output[position] = output[position - 1]
        return output
    if condition == "R9_4fps":
        return values[::2].copy()
    raise ValueError(f"not a temporal perturbation: {condition}")


def resize_half_restore(frames: np.ndarray) -> np.ndarray:
    """Downscale each frame by 0.5 and restore its exact original dimensions."""
    output = []
    for frame in np.asarray(frames):
        height, width = frame.shape[:2]
        reduced = cv2.resize(
            frame,
            (max(1, width // 2), max(1, height // 2)),
            interpolation=cv2.INTER_AREA,
        )
        output.append(
            cv2.resize(reduced, (width, height), interpolation=cv2.INTER_CUBIC)
        )
    return np.stack(output)


def insert_scene_cut(target: np.ndarray, donor: np.ndarray) -> np.ndarray:
    """Replace the second half of a target window with a resized donor window."""
    target_values = np.asarray(target)
    donor_values = np.asarray(donor)
    if target_values.ndim != 4 or donor_values.ndim != 4:
        raise ValueError("scene-cut inputs must be frame sequences")
    if len(target_values) != len(donor_values) or len(target_values) < 4:
        raise ValueError("scene-cut inputs must have the same temporal length >=4")
    height, width = target_values.shape[1:3]
    resized = np.stack(
        [cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in donor_values]
    )
    boundary = len(target_values) // 2
    return np.concatenate((target_values[:boundary], resized[boundary:]), axis=0)


def h264_roundtrip(
    frames: np.ndarray,
    crf: int,
    ffmpeg: str = "ffmpeg",
    fps: int = 8,
) -> np.ndarray:
    """Encode and decode one sampled frame sequence with libx264 at fixed CRF."""
    values = np.ascontiguousarray(frames, dtype=np.uint8)
    if values.ndim != 4 or values.shape[-1] != 3 or len(values) < 1:
        raise ValueError(f"expected uint8 BGR [T,H,W,3], got {values.shape}")
    height, width = values.shape[1:3]
    with tempfile.TemporaryDirectory(prefix="u0_h264_") as directory:
        encoded = Path(directory) / "variant.mkv"
        encode = subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                str(int(crf)),
                "-pix_fmt",
                "yuv420p",
                "-y",
                str(encoded),
            ],
            input=values.tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if encode.returncode != 0:
            raise RuntimeError(encode.stderr.decode("utf-8", errors="replace"))
        capture = cv2.VideoCapture(str(encoded))
        decoded = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded.append(frame)
        capture.release()
    if len(decoded) != len(values):
        raise ValueError(f"H.264 roundtrip changed frame count {len(values)} -> {len(decoded)}")
    result = np.stack(decoded)[:, :height, :width]
    if result.shape != values.shape:
        raise ValueError(f"H.264 roundtrip changed shape {values.shape} -> {result.shape}")
    return result
