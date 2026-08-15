"""Shared real-only calibration infrastructure for U0 extension studies.

The locked U0 release does not depend on this module. It centralizes the
predeclared reserve design and the repeated candidate-calibration path used by
calibration-size, cross-domain, independent-real, and OAS analyses.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import read_checkpoint_parts
from .calibration import (
    U0VideoReferences,
    U0WindowReferences,
    calibrate_u0_video_branches,
    calibrate_u0_window_components,
)
from .release_io import REPOSITORY_ROOT
from .u0_analysis import effective_k_reference
from .whitening import stable_sorted


DATASETS = ("comgenvid", "videofeedback", "genvideo")
CALIBRATION_SEEDS = (17, 29, 43, 71, 101)
CALIBRATION_SIZES = (25, 50, 100, 200)
CALIBRATION_RESERVE_SPECS = {
    "comgenvid": (
        REPOSITORY_ROOT / "cache/indexes/comgenvid.csv",
        REPOSITORY_ROOT / "cache/patch_embeddings/comgenvid",
    ),
    "videofeedback": (
        REPOSITORY_ROOT / "cache/indexes/videofeedback.csv",
        REPOSITORY_ROOT / "cache/patch_embeddings/videofeedback",
    ),
    "genvideo": (
        REPOSITORY_ROOT / "cache/indexes/genvideo.csv",
        REPOSITORY_ROOT / "cache/patch_embeddings/genvideo",
    ),
}

# Historical names retained for CLI compatibility.
SEEDS = CALIBRATION_SEEDS
SIZES = CALIBRATION_SIZES


def candidate(seed: int, size: int) -> str:
    """Return the stable column suffix for a calibration split candidate."""

    return f"s{seed}_n{size}"


def load_parts(root: Path, split: str) -> pd.DataFrame:
    """Load and validate one U0 calibration-experiment raw-score split."""

    frame, _ = read_checkpoint_parts(
        root,
        f"checkpoints/{split}/*/shard_*/part_*.csv",
        empty_error=f"no {split} candidate score parts",
    )
    keys = ["video_id", "sampling", "window_id"]
    if frame.duplicated(keys).any():
        raise ValueError(f"duplicate {split} score keys")
    return frame


def _calibrate_windows(
    frame: pd.DataFrame,
    candidate_name: str,
    global_spatial_ref: np.ndarray,
    global_t1_ref: np.ndarray,
    patch_spatial_ref: np.ndarray,
    patch_d2_ref: np.ndarray,
) -> pd.DataFrame:
    output = frame.copy()
    calibrated = calibrate_u0_window_components(
        output["global_spatial_raw"].to_numpy(),
        output["global_t1_raw"].to_numpy(),
        output[f"patch_spatial__{candidate_name}"].to_numpy(),
        output[f"patch_d2__{candidate_name}"].to_numpy(),
        U0WindowReferences(
            global_spatial_ref,
            global_t1_ref,
            patch_spatial_ref,
            patch_d2_ref,
        ),
    )
    for column, values in calibrated.items():
        if column == "S_k":
            continue
        output["patch_d2" if column == "patch_temporal" else column] = values
    return output


def _calibrate_named_candidate(
    evaluation: pd.DataFrame,
    calibration: pd.DataFrame,
    selected_ids: set[str],
    candidate_name: str,
    global_spatial_ref: np.ndarray,
    global_t1_ref: np.ndarray,
    *,
    include_effective_k: bool,
    bank_error: str,
    coverage_error: str,
) -> pd.DataFrame:
    k1 = calibration[
        calibration["sampling"].eq("k1")
        & calibration["video_id"].isin(selected_ids)
    ]
    if len(k1) != len(selected_ids):
        raise ValueError(f"{candidate_name}: {bank_error}")
    patch_spatial_ref = stable_sorted(
        k1[f"patch_spatial__{candidate_name}"].to_numpy()
    )
    patch_d2_ref = stable_sorted(k1[f"patch_d2__{candidate_name}"].to_numpy())

    evaluation_scored = _calibrate_windows(
        evaluation,
        candidate_name,
        global_spatial_ref,
        global_t1_ref,
        patch_spatial_ref,
        patch_d2_ref,
    )
    calibration_k3 = _calibrate_windows(
        calibration[calibration["sampling"].eq("k3")],
        candidate_name,
        global_spatial_ref,
        global_t1_ref,
        patch_spatial_ref,
        patch_d2_ref,
    )
    video = (
        evaluation_scored.groupby("video_id", sort=False)
        .agg(
            effective_k=("window_id", "size"),
            G_raw=("G_k", "mean"),
            L_raw=("L_k", "mean"),
        )
        .reset_index()
    )
    pieces = []
    for effective_k, target in video.groupby("effective_k", sort=True):
        target = target.copy()
        calibrated = calibrate_u0_video_branches(
            target["G_raw"].to_numpy(),
            target["L_raw"].to_numpy(),
            U0VideoReferences(
                effective_k_reference(
                    calibration_k3,
                    int(effective_k),
                    "G_k",
                    selected_video_ids=selected_ids,
                ),
                effective_k_reference(
                    calibration_k3,
                    int(effective_k),
                    "L_k",
                    selected_video_ids=selected_ids,
                ),
            ),
        )
        for column, values in calibrated.items():
            target[column] = values
        pieces.append(target)
    result = pd.concat(pieces, ignore_index=True)
    if (
        len(result) != evaluation["video_id"].nunique()
        or result["video_id"].duplicated().any()
    ):
        raise ValueError(f"{candidate_name}: {coverage_error}")
    columns = ["video_id", "G", "L", "S"]
    if include_effective_k:
        columns.insert(1, "effective_k")
    return result[columns]


def calibrate_candidate(
    evaluation: pd.DataFrame,
    reserve: pd.DataFrame,
    selected_ids: set[str],
    name: str,
    global_spatial_ref: np.ndarray,
    global_t1_ref: np.ndarray,
) -> pd.DataFrame:
    """Calibrate one size/seed candidate and retain effective-K diagnostics."""

    return _calibrate_named_candidate(
        evaluation,
        reserve,
        selected_ids,
        name,
        global_spatial_ref,
        global_t1_ref,
        include_effective_k=True,
        bank_error="K1 reserve count mismatch",
        coverage_error="incomplete candidate video scores",
    )


def calibrate_cross(
    evaluation: pd.DataFrame,
    calibration: pd.DataFrame,
    selected_ids: set[str],
    candidate_name: str,
    global_spatial_ref: np.ndarray,
    global_t1_ref: np.ndarray,
) -> pd.DataFrame:
    """Calibrate one cross-domain/OAS candidate using a named real bank."""

    return _calibrate_named_candidate(
        evaluation,
        calibration,
        selected_ids,
        candidate_name,
        global_spatial_ref,
        global_t1_ref,
        include_effective_k=False,
        bank_error="K1 bank count mismatch",
        coverage_error="incomplete cross calibration",
    )


__all__ = [
    "CALIBRATION_RESERVE_SPECS",
    "CALIBRATION_SEEDS",
    "CALIBRATION_SIZES",
    "DATASETS",
    "SEEDS",
    "SIZES",
    "calibrate_candidate",
    "calibrate_cross",
    "candidate",
    "load_parts",
]
