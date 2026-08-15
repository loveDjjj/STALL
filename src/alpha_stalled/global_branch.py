"""Canonical Global Spatial/T1 primitives for locked Alpha-STALLED protocols.

The formal Global branch is the original STALL definition: frame-level spatial
likelihood with max aggregation, normalized lag-1 temporal likelihood with min
aggregation, and equal fusion after real-only ECDF calibration. D3, volatility,
and higher-order Global variants deliberately do not appear in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .whitening import (
    StableGaussianParams,
    l2_normalized_first_order,
    score_gaussian_aggregate_float64,
)


GLOBAL_SPATIAL_WEIGHT = 0.5
GLOBAL_SPATIAL_AGGREGATION = "max"
GLOBAL_TEMPORAL_AGGREGATION = "min"
GLOBAL_TEMPORAL_ORDER = 1


@dataclass(frozen=True)
class GlobalRawScores:
    """Raw Global likelihood scores before real-only ECDF calibration."""

    spatial: np.ndarray
    temporal_t1: np.ndarray


def global_t1_features(
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized lag-1 differences and the exact-zero mask."""

    return l2_normalized_first_order(features)


def fuse_global_components(
    spatial: np.ndarray,
    temporal: np.ndarray,
    spatial_weight: float = GLOBAL_SPATIAL_WEIGHT,
) -> np.ndarray:
    """Fuse calibrated Global Spatial and Global T1 percentiles."""

    spatial_values = np.asarray(spatial, dtype=np.float64)
    temporal_values = np.asarray(temporal, dtype=np.float64)
    if spatial_values.shape != temporal_values.shape:
        raise ValueError(
            f"score shapes differ: {spatial_values.shape} != {temporal_values.shape}"
        )
    if not 0.0 <= spatial_weight <= 1.0:
        raise ValueError("global spatial weight must be in [0,1]")
    return spatial_weight * spatial_values + (1.0 - spatial_weight) * temporal_values


@torch.inference_mode()
def score_global_raw(
    features: torch.Tensor | np.ndarray,
    spatial_params: StableGaussianParams,
    temporal_params: StableGaussianParams,
    *,
    device: str | torch.device = "cuda:0",
) -> GlobalRawScores:
    """Score the locked Global branch before real-only ECDF calibration."""

    tensor = torch.as_tensor(features)
    spatial, _ = score_gaussian_aggregate_float64(
        tensor,
        spatial_params,
        aggregation=GLOBAL_SPATIAL_AGGREGATION,
        device=device,
        compute_percentile=False,
    )
    temporal_features, zero_mask = global_t1_features(tensor)
    temporal, _ = score_gaussian_aggregate_float64(
        temporal_features,
        temporal_params,
        aggregation=GLOBAL_TEMPORAL_AGGREGATION,
        device=device,
        invalid_mask=zero_mask,
        compute_percentile=False,
    )
    return GlobalRawScores(spatial=spatial, temporal_t1=temporal)


__all__ = [
    "GLOBAL_SPATIAL_AGGREGATION",
    "GLOBAL_SPATIAL_WEIGHT",
    "GLOBAL_TEMPORAL_AGGREGATION",
    "GLOBAL_TEMPORAL_ORDER",
    "GlobalRawScores",
    "fuse_global_components",
    "global_t1_features",
    "score_global_raw",
]
