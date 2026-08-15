"""Shared infrastructure for the historical pre-release multi-window scorer.

This module preserves the old five-column identity, calibrated legacy scorers,
and environment-specific parameter defaults. It is intentionally separate from
the locked-U0 protocol modules.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .artifacts import checkpoint_completed_keys
from .release_io import REPOSITORY_ROOT
from .video_io import decode_selected_frames


KEY_COLUMNS = ["dataset", "protocol_split", "subset", "source_model", "filename"]
SAMPLINGS = ("K1_current", "K3_uniform", "K5_uniform", "all_nonoverlap")
LOCAL_PARAMS = {
    "comgenvid": Path("/tmp/alpha_stalled_local_d2_work/comgenvid_L0.npz"),
    "videofeedback": Path("/tmp/alpha_stalled_local_d2_work/videofeedback_L0.npz"),
    "genvideo": Path("/tmp/alpha_stalled_local_d2_work/genvideo_L0.npz"),
}


def stable_shard(row: pd.Series, num_shards: int) -> int:
    """Apply the historical five-column SHA-256 shard rule."""

    key = "|".join(str(row[column]) for column in KEY_COLUMNS)
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16) % num_shards


def video_key(row: pd.Series | dict) -> tuple[str, ...]:
    """Return the historical five-column video identity tuple."""

    return tuple(str(row[column]) for column in KEY_COLUMNS)


def resolve_video_path(
    value: str, repository_root: Path = REPOSITORY_ROOT
) -> Path:
    """Resolve paths with the exact historical cwd/repository search order."""

    path = Path(value)
    if path.is_absolute():
        return path
    candidates = (
        Path.cwd() / path,
        repository_root / path,
        repository_root.parent / path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"video not found: {value}")


def load_windows(row: pd.Series, sampling: str) -> list[list[int]]:
    """Parse and validate one historical manifest sampling column."""

    windows = json.loads(row[f"indices_{sampling}"])
    parsed = [[int(index) for index in window] for window in windows]
    if not parsed or any(len(window) != 16 for window in parsed):
        raise ValueError(f"invalid {sampling} windows for {video_key(row)}")
    if len({tuple(window) for window in parsed}) != len(parsed):
        raise ValueError(f"duplicate {sampling} windows for {video_key(row)}")
    return parsed


def decode_manifest_row(row: pd.Series, sampling: str, seek_gap: int) -> dict:
    """Decode unique native frames and map historical windows to positions."""

    windows = load_windows(row, sampling)
    unique_indices = sorted({index for window in windows for index in window})
    frames = decode_selected_frames(
        resolve_video_path(str(row["video_path"])), unique_indices, seek_gap
    )
    positions = {index: position for position, index in enumerate(unique_indices)}
    return {
        "row": row,
        "windows": windows,
        "window_positions": [
            [positions[index] for index in window] for window in windows
        ],
        "unique_indices": unique_indices,
        "frames": frames,
    }


def decode_manifest_row_with_retries(
    row: pd.Series,
    sampling: str,
    seek_gap: int,
    attempts: int,
    retry_delay: float = 0.25,
    *,
    decoder: Callable[[pd.Series, str, int], dict] | None = None,
) -> dict:
    """Retry the historical decoder without changing its linear backoff."""

    if attempts < 1:
        raise ValueError("decode attempts must be positive")
    decode = decoder or decode_manifest_row
    for attempt in range(attempts):
        try:
            return decode(row, sampling, seek_gap)
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(retry_delay * (attempt + 1))
    raise AssertionError("unreachable")


def load_completed(checkpoint_dir: Path) -> tuple[set[tuple[str, ...]], list[Path]]:
    """Load completed historical video keys from checkpoint CSV parts."""

    return checkpoint_completed_keys(checkpoint_dir, KEY_COLUMNS)


def score_batch(
    decoded: list[dict],
    extractor: Any,
    global_scorer: Any,
    local_scorer: Any,
    frame_batch_size: int,
) -> list[dict]:
    """Score decoded windows with the exact historical calibrated model path."""

    outputs = extractor.frames_to_global_patch_embeddings(
        [item["frames"] for item in decoded],
        batch_size=frame_batch_size,
    )
    global_windows: list[np.ndarray] = []
    patch_windows: list[np.ndarray] = []
    metadata: list[tuple[dict, int]] = []
    for item, output in zip(decoded, outputs):
        global_emb = output["global"]
        patch_emb = output["patch"]
        for window_id, positions in enumerate(item["window_positions"]):
            global_windows.append(global_emb[positions])
            patch_windows.append(patch_emb[positions])
            metadata.append((item, window_id))
    global_batch = np.stack(global_windows)
    patch_batch = np.stack(patch_windows)
    global_scores = global_scorer._scores_from_embs(global_batch)
    local_scores = local_scorer.score_batch(
        patch_batch,
        patch_temp_mode="same_grid_second_order",
        patch_spat_weight=0.1,
        patch_temp_weight=0.9,
        aggregation=local_scorer.aggregation_config.get("mode", "bottomk_mean"),
        bottomk_ratio=local_scorer.params_bottomk_ratio,
        temporal_run_length=local_scorer.params_temporal_run_length,
        patch_region_size=local_scorer.params_patch_region_size,
        global_batch=global_batch,
    )
    rows: list[dict] = []
    for index, (item, window_id) in enumerate(metadata):
        source = item["row"]
        global_spatial = float(global_scores["spat_percentile"][index])
        global_t1 = float(global_scores["temp_percentile"][index])
        global_final = float(global_scores["final_score"][index])
        patch_spatial = float(local_scores["patch_spat_percentile"][index])
        patch_d2 = float(local_scores["patch_temp_percentile"][index])
        local_final = float(local_scores["patch_final_score"][index])
        rows.append(
            {
                **{column: source[column] for column in KEY_COLUMNS},
                "video_path": source["video_path"],
                "duration_seconds": source["duration_seconds"],
                "sampling": source["active_sampling"],
                "effective_k": len(item["windows"]),
                "unique_frame_count": len(item["unique_indices"]),
                "window_id": window_id,
                "frame_indices": json.dumps(
                    item["windows"][window_id], separators=(",", ":")
                ),
                "global_spatial": global_spatial,
                "global_t1": global_t1,
                "G_k": global_final,
                "patch_spatial": patch_spatial,
                "patch_d2": patch_d2,
                "L_k": local_final,
                "S_k": 0.6 * global_final + 0.4 * local_final,
            }
        )
    return rows


__all__ = [
    "KEY_COLUMNS",
    "LOCAL_PARAMS",
    "SAMPLINGS",
    "decode_manifest_row",
    "decode_manifest_row_with_retries",
    "load_completed",
    "load_windows",
    "resolve_video_path",
    "score_batch",
    "stable_shard",
    "video_key",
]
