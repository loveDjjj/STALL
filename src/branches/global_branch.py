"""我们方法中的 Global Spatial 与 Global T1 证据分支。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from math_utils import (
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
    """Global 分支在真实视频 CDF 校准前的原始似然分数。"""

    spatial: np.ndarray
    temporal_t1: np.ndarray


def fuse_global_components(
    spatial: np.ndarray,
    temporal: np.ndarray,
    spatial_weight: float = GLOBAL_SPATIAL_WEIGHT,
) -> np.ndarray:
    """融合已校准的 Global Spatial 与 Global T1 分数。"""

    spatial_values = np.asarray(spatial, dtype=np.float64)
    temporal_values = np.asarray(temporal, dtype=np.float64)
    if spatial_values.shape != temporal_values.shape:
        raise ValueError(f"score shapes differ: {spatial_values.shape} != {temporal_values.shape}")
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
    """计算我们方法 Global 分支的空间和一阶时间原始分数。"""

    tensor = torch.as_tensor(features)
    spatial, _ = score_gaussian_aggregate_float64(
        tensor, spatial_params, aggregation=GLOBAL_SPATIAL_AGGREGATION,
        device=device, compute_percentile=False,
    )
    temporal_features, zero_mask = l2_normalized_first_order(tensor)
    temporal, _ = score_gaussian_aggregate_float64(
        temporal_features, temporal_params, aggregation=GLOBAL_TEMPORAL_AGGREGATION,
        device=device, invalid_mask=zero_mask, compute_percentile=False,
    )
    return GlobalRawScores(spatial=spatial, temporal_t1=temporal)


__all__ = [
    "GLOBAL_SPATIAL_WEIGHT", "GLOBAL_SPATIAL_AGGREGATION",
    "GLOBAL_TEMPORAL_AGGREGATION", "GLOBAL_TEMPORAL_ORDER", "GlobalRawScores",
    "fuse_global_components", "score_global_raw",
]
