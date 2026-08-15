"""Strict decoding and raw Global/Local scoring for locked U0 windows."""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from .global_branch import score_global_raw
from .local_branch import score_local_raw
from .release_io import resolve_required_video
from .u0_protocol import KEY_COLUMNS
from .video_io import decode_selected_frames
from .whitening import StableGaussianParams
from stall_patch import PatchSTALL


def score_raw_components(
    global_batch: torch.Tensor,
    patch_batch: torch.Tensor,
    params: dict[str, StableGaussianParams],
    device: str,
) -> dict[str, np.ndarray]:
    """Score the four locked U0 components from cached window embeddings."""

    global_raw = score_global_raw(
        global_batch,
        params["global_spatial"],
        params["global_t1"],
        device=device,
    )
    local_raw = score_local_raw(
        patch_batch,
        params["patch_spatial"],
        params["patch_d2"],
        device=device,
    )
    return {
        "global_spatial_raw": global_raw.spatial,
        "global_t1_raw": global_raw.temporal_t1,
        "patch_spatial_raw": local_raw.patch_spatial,
        "patch_d2_raw": local_raw.patch_temporal,
    }


# Historical K1 scorer name retained as the same function object.
score_raw_batch = score_raw_components


def decode_row(
    row: pd.Series,
    windows: list[list[int]],
    seek_gap: int,
    attempts: int,
) -> dict[str, Any]:
    """Decode the unique native frames for one locked video with retries."""

    if attempts < 1:
        raise ValueError("decode attempts must be positive")
    if len(windows) != int(row["effective_k"]):
        raise ValueError(f"effective_k mismatch for {row['video_id']}")
    unique_indices = sorted({int(index) for window in windows for index in window})
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            frames = decode_selected_frames(
                resolve_required_video(str(row["video_path"])),
                unique_indices,
                seek_gap,
            )
            positions = {index: position for position, index in enumerate(unique_indices)}
            return {
                "row": row,
                "windows": windows,
                "positions": [
                    [positions[index] for index in window] for window in windows
                ],
                "frames": frames,
                "unique_indices": unique_indices,
            }
        except Exception as caught:
            error = caught
            if attempt + 1 < attempts:
                time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"decode failed after {attempts} attempts: {error}") from error


@torch.inference_mode()
def score_batch(
    decoded: list[dict[str, Any]],
    extractor: PatchSTALL,
    params: dict[str, StableGaussianParams],
    score_device: str,
    frame_batch_size: int,
) -> list[dict[str, Any]]:
    """Extract and score all locked windows from a decoded outer video batch."""

    if not decoded:
        return []
    # DINO frame grouping is part of the numerical protocol. Each video has a
    # separate extraction call so the final frame batch cannot depend on the
    # number of neighboring videos in the I/O batch.
    extracted = [
        extractor.frames_to_global_patch_embeddings(
            [item["frames"]], batch_size=frame_batch_size
        )[0]
        for item in decoded
    ]
    global_windows = []
    patch_windows = []
    metadata = []
    for item, output in zip(decoded, extracted):
        if tuple(output["grid_size"]) != (14, 14):
            raise ValueError(f"unexpected patch grid: {output['grid_size']}")
        for window_id, positions in enumerate(item["positions"]):
            global_windows.append(output["global"][positions])
            patch_windows.append(output["patch"][positions])
            metadata.append((item, window_id))
    global_batch = torch.from_numpy(np.stack(global_windows).astype(np.float32))
    patch_batch = torch.from_numpy(np.stack(patch_windows).astype(np.float32))

    raw = score_raw_components(global_batch, patch_batch, params, score_device)

    rows: list[dict[str, Any]] = []
    for index, (item, window_id) in enumerate(metadata):
        source = item["row"]
        rows.append(
            {
                **{column: source[column] for column in KEY_COLUMNS},
                "video_path": source["video_path"],
                "duration_seconds": float(source["duration_seconds"]),
                "effective_k": int(source["effective_k"]),
                "unique_frame_count": len(item["unique_indices"]),
                "window_id": window_id,
                "frame_indices": json.dumps(
                    item["windows"][window_id], separators=(",", ":")
                ),
                "global_spatial_raw": raw["global_spatial_raw"][index],
                "global_t1_raw": raw["global_t1_raw"][index],
                "patch_spatial_raw": raw["patch_spatial_raw"][index],
                "patch_d2_raw": raw["patch_d2_raw"][index],
            }
        )
    return rows


__all__ = [
    "decode_row",
    "score_batch",
    "score_raw_batch",
    "score_raw_components",
]
