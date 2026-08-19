"""我们方法使用的窗口级和视频级真实数据校准工具。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from branches.global_branch import (
    GLOBAL_SPATIAL_WEIGHT,
    fuse_global_components,
)
from branches.local_branch import (
    LOCAL_SPATIAL_WEIGHT,
    fuse_local_components,
)
from math_utils import empirical_cdf_right_inclusive, stable_sorted


GLOBAL_BRANCH_WEIGHT = 0.6


def empirical_cdf(values: np.ndarray, calibration: np.ndarray) -> np.ndarray:
    """Right-inclusive empirical CDF with deterministic reference sorting."""

    return empirical_cdf_right_inclusive(values, stable_sorted(calibration))


@dataclass(frozen=True)
class WindowReferences:
    """四个窗口原始分量对应的已排序真实视频参考分数。"""

    global_spatial: np.ndarray
    global_t1: np.ndarray
    patch_spatial: np.ndarray
    patch_temporal: np.ndarray


@dataclass(frozen=True)
class VideoReferences:
    """视频级 Global 与 Local 分支对应的 effective-K 真实参考分数。"""

    global_branch: np.ndarray
    local_branch: np.ndarray


def cdf_with_positive_infinity(
    values: np.ndarray, reference_sorted: np.ndarray
) -> np.ndarray:
    """Apply the right-inclusive CDF while mapping declared ``+inf`` to one."""
    scores = np.asarray(values, dtype=np.float64)
    if np.isnan(scores).any() or np.isneginf(scores).any():
        raise ValueError("CDF values contain NaN or negative infinity")
    result = np.ones(len(scores), dtype=np.float64)
    finite = np.isfinite(scores)
    result[finite] = empirical_cdf_right_inclusive(
        scores[finite], reference_sorted
    )
    return result


def _weighted_pair(
    first: np.ndarray,
    second: np.ndarray,
    first_weight: float,
) -> np.ndarray:
    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    if first_values.shape != second_values.shape:
        raise ValueError(
            f"score shapes differ: {first_values.shape} != {second_values.shape}"
        )
    return first_weight * first_values + (1.0 - first_weight) * second_values


def fuse_global_local(
    global_score: np.ndarray,
    local_score: np.ndarray,
    global_weight: float = GLOBAL_BRANCH_WEIGHT,
) -> np.ndarray:
    """Fuse calibrated video-level Global and Local branch scores."""
    return _weighted_pair(global_score, local_score, global_weight)


def calibrate_window_components(
    global_spatial_raw: np.ndarray,
    global_t1_raw: np.ndarray,
    patch_spatial_raw: np.ndarray,
    patch_temporal_raw: np.ndarray,
    references: WindowReferences,
) -> dict[str, np.ndarray]:
    """校准并融合一个窗口的四个原始分量。"""
    global_spatial = empirical_cdf_right_inclusive(
        global_spatial_raw, references.global_spatial
    )
    global_t1 = cdf_with_positive_infinity(global_t1_raw, references.global_t1)
    patch_spatial = empirical_cdf_right_inclusive(
        patch_spatial_raw, references.patch_spatial
    )
    patch_temporal = empirical_cdf_right_inclusive(
        patch_temporal_raw, references.patch_temporal
    )
    global_score = fuse_global_components(global_spatial, global_t1)
    local_score = fuse_local_components(patch_spatial, patch_temporal)
    return {
        "global_spatial": global_spatial,
        "global_t1": global_t1,
        "patch_spatial": patch_spatial,
        "patch_temporal": patch_temporal,
        "G_k": global_score,
        "L_k": local_score,
        "S_k": fuse_global_local(global_score, local_score),
    }


def calibrate_video_branches(
    global_raw: np.ndarray,
    local_raw: np.ndarray,
    references: VideoReferences,
) -> dict[str, np.ndarray]:
    """校准 effective-K 视频均值并执行 Global/Local 融合。"""
    global_score = empirical_cdf_right_inclusive(
        global_raw, references.global_branch
    )
    local_score = empirical_cdf_right_inclusive(local_raw, references.local_branch)
    return {
        "G": global_score,
        "L": local_score,
        "S": fuse_global_local(global_score, local_score),
    }


__all__ = [
    "GLOBAL_BRANCH_WEIGHT",
    "GLOBAL_SPATIAL_WEIGHT",
    "LOCAL_SPATIAL_WEIGHT",
    "WindowReferences",
    "VideoReferences",
    "calibrate_video_branches",
    "calibrate_window_components",
    "cdf_with_positive_infinity",
    "empirical_cdf",
    "fuse_global_components",
    "fuse_global_local",
    "fuse_local_components",
]
