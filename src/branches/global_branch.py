"""我们方法中的 Global Spatial 与 Global T1 证据分支。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from math_utils import (
    StableGaussianParams,
    l2_normalized_first_order,
    score_gaussian_aggregate_float64,
    stable_sorted,
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


def load_official_stall_parameters(path: Path) -> dict[str, StableGaussianParams]:
    """读取官方 STALL 发布的 VATEX Global 参数与视频级 CDF 参考。

    官方文件保存每条 VATEX 视频的逐帧或逐转移似然；STALL 分别以
    max/min 聚合后再做 CDF。这里在加载时完成同一聚合，避免误用目标域
    Local 校准视频作为 Global 的窗口参考。
    """

    required = {
        "mu_spat",
        "W_spat",
        "calib_ll_spat",
        "mu_temp",
        "W_temp",
        "calib_ll_temp",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"官方 STALL 参数缺少字段：{sorted(missing)}")
        return {
            "global_spatial": StableGaussianParams(
                mean=np.asarray(data["mu_spat"], dtype=np.float64),
                whitening=np.asarray(data["W_spat"], dtype=np.float64),
                calibration_raw=stable_sorted(
                    np.max(np.asarray(data["calib_ll_spat"], dtype=np.float64), axis=1)
                ),
            ),
            "global_t1": StableGaussianParams(
                mean=np.asarray(data["mu_temp"], dtype=np.float64),
                whitening=np.asarray(data["W_temp"], dtype=np.float64),
                calibration_raw=stable_sorted(
                    np.min(np.asarray(data["calib_ll_temp"], dtype=np.float64), axis=1)
                ),
            ),
        }


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
        tensor,
        spatial_params,
        aggregation=GLOBAL_SPATIAL_AGGREGATION,
        device=device,
        compute_percentile=False,
    )
    temporal_features, zero_mask = l2_normalized_first_order(tensor)
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
    "GLOBAL_SPATIAL_WEIGHT",
    "GLOBAL_SPATIAL_AGGREGATION",
    "GLOBAL_TEMPORAL_AGGREGATION",
    "GLOBAL_TEMPORAL_ORDER",
    "GlobalRawScores",
    "fuse_global_components",
    "load_official_stall_parameters",
    "score_global_raw",
]
