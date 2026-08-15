"""Canonical Local branch primitives for locked Alpha-STALLED protocols.

The formal Local branch uses unpooled same-grid patch tokens, feature-wise L2
normalization, mean Gaussian likelihood aggregation, and a fixed spatial weight.
Matching, multi-lag, higher-order, region-pooling, and residual variants are
historical experiments and deliberately do not appear in this public module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .whitening import (
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

    ``temporal_order=1`` exists only for the controlled D1 ablation. The locked
    U0 and duration-aware protocols use the default second-order definition.
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
    "score_local_raw",
]
