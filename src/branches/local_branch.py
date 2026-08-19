"""我们方法的 Local 分支基础计算。

The formal Local branch uses unpooled same-grid patch tokens, feature-wise L2
normalization, mean Gaussian likelihood aggregation, and a fixed spatial weight.
Matching, multi-lag, higher-order, region-pooling, and residual variants are
historical experiments and deliberately do not appear in this public module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from math_utils import (
    StableGaussianParams,
    l2_normalized_patch_first_order,
    l2_normalized_second_order,
    score_gaussian_aggregate_float64,
)


LOCAL_SPATIAL_WEIGHT = 0.1
LOCAL_TEMPORAL_ORDER = 2
LOCAL_AGGREGATION = "mean"


@dataclass(frozen=True)
class LocalRawScores:
    """Raw mean-likelihood scores before real-only ECDF calibration."""

    patch_spatial: np.ndarray
    patch_temporal: np.ndarray


def local_d1_features(patch: torch.Tensor) -> torch.Tensor:
    """Controlled first-order Local temporal feature used by the D1 ablation."""

    return l2_normalized_patch_first_order(patch)


def local_d2_features(patch: torch.Tensor) -> torch.Tensor:
    """Locked same-grid second-order Local temporal feature."""

    return l2_normalized_second_order(patch)


def same_grid_finite_difference(patch_sequence: np.ndarray, order: int) -> np.ndarray:
    """计算同一 patch 网格上的一阶或二阶局部时间差分。"""

    if patch_sequence.ndim != 3:
        raise ValueError(f"expected patch_sequence [T, P, D], got {patch_sequence.shape}")
    if order not in {1, 2}:
        raise ValueError(f"local temporal order must be 1 or 2, got {order}")
    if len(patch_sequence) <= order:
        raise ValueError(f"not enough frames for order={order}: T={len(patch_sequence)}")
    coefficients = np.array(
        [((-1) ** (order - index)) * math.comb(order, index) for index in range(order + 1)],
        dtype=np.float32,
    )
    difference = np.zeros_like(patch_sequence[order:], dtype=np.float32)
    for index, coefficient in enumerate(coefficients):
        difference += coefficient * patch_sequence[index : index + len(difference)]
    norms = np.linalg.norm(difference, axis=-1, keepdims=True)
    return difference / np.where(norms == 0, 1.0, norms)


def patch_temporal_delta(
    patch_sequence: np.ndarray,
    grid_size: tuple[int, int],
    mode: str = "same_grid_second_order",
    region_size: int = 1,
    **unused: object,
) -> np.ndarray:
    """为参数拟合提供同网格 D1/D2 特征的统一入口。"""

    del grid_size, unused
    if region_size != 1:
        raise ValueError("Alpha STALL 的 Local 分支固定使用 region_size=1")
    if mode in {"same_grid", "same_grid_lag1"}:
        return same_grid_finite_difference(patch_sequence, order=1)
    if mode == "same_grid_second_order":
        return same_grid_finite_difference(patch_sequence, order=2)
    raise ValueError(f"unsupported patch temporal mode: {mode}")


def fuse_local_components(
    spatial: np.ndarray,
    temporal: np.ndarray,
    spatial_weight: float = LOCAL_SPATIAL_WEIGHT,
) -> np.ndarray:
    """Fuse calibrated Patch Spatial and Local Temporal percentiles."""

    spatial_values = np.asarray(spatial, dtype=np.float64)
    temporal_values = np.asarray(temporal, dtype=np.float64)
    if spatial_values.shape != temporal_values.shape:
        raise ValueError(
            f"score shapes differ: {spatial_values.shape} != {temporal_values.shape}"
        )
    if not 0.0 <= spatial_weight <= 1.0:
        raise ValueError("local spatial weight must be in [0,1]")
    return spatial_weight * spatial_values + (1.0 - spatial_weight) * temporal_values


@torch.inference_mode()
def score_local_raw(
    patch: torch.Tensor | np.ndarray,
    spatial_params: StableGaussianParams,
    temporal_params: StableGaussianParams,
    *,
    device: str | torch.device = "cuda:0",
    temporal_order: int = LOCAL_TEMPORAL_ORDER,
) -> LocalRawScores:
    """Score the formal Local branch before real-only ECDF calibration.

    ``temporal_order=1`` 仅用于受控 D1 消融；默认方法使用二阶定义。
    """

    if temporal_order == 1:
        temporal_features = local_d1_features(torch.as_tensor(patch))
    elif temporal_order == 2:
        temporal_features = local_d2_features(torch.as_tensor(patch))
    else:
        raise ValueError("formal Local temporal order must be 1 or 2")

    patch_spatial, _ = score_gaussian_aggregate_float64(
        patch,
        spatial_params,
        aggregation=LOCAL_AGGREGATION,
        device=device,
        compute_percentile=False,
    )
    patch_temporal, _ = score_gaussian_aggregate_float64(
        temporal_features,
        temporal_params,
        aggregation=LOCAL_AGGREGATION,
        device=device,
        compute_percentile=False,
    )
    return LocalRawScores(
        patch_spatial=patch_spatial,
        patch_temporal=patch_temporal,
    )


__all__ = [
    "LOCAL_AGGREGATION",
    "LOCAL_SPATIAL_WEIGHT",
    "LOCAL_TEMPORAL_ORDER",
    "LocalRawScores",
    "fuse_local_components",
    "local_d1_features",
    "local_d2_features",
    "patch_temporal_delta",
    "score_local_raw",
    "same_grid_finite_difference",
]
