"""我们方法的窗口级 Global/Local 原始分数计算。"""

from __future__ import annotations

import numpy as np
import torch

from math_utils import StableGaussianParams
from branches.global_branch import score_global_raw
from branches.local_branch import score_local_raw


def score_window_components(
    global_features: torch.Tensor | np.ndarray,
    patch_features: torch.Tensor | np.ndarray,
    parameters: dict[str, StableGaussianParams],
    *,
    device: str,
    local_enabled: bool = True,
    local_spatial_enabled: bool = True,
    local_temporal_enabled: bool = True,
    temporal_order: int = 2,
) -> dict[str, np.ndarray]:
    """计算一批窗口的原始分数，所有输出均是一维窗口数组。

    Global 始终返回空间与一阶时间分量。Local 分支可通过配置分别关闭空间或
    时序证据；被关闭的分量以 ``NaN`` 表示，避免被误认为真实模型输出。
    """

    global_raw = score_global_raw(
        global_features,
        parameters["global_spatial"],
        parameters["global_t1"],
        device=device,
    )
    count = len(global_raw.spatial)
    unavailable = np.full(count, np.nan, dtype=np.float64)
    result = {
        "global_spatial_raw": global_raw.spatial,
        "global_t1_raw": global_raw.temporal_t1,
        "patch_spatial_raw": unavailable.copy(),
        "patch_temporal_raw": unavailable.copy(),
    }
    if not local_enabled:
        return result

    local_raw = score_local_raw(
        patch_features,
        parameters["patch_spatial"],
        parameters["patch_temporal"],
        device=device,
        temporal_order=temporal_order,
    )
    if local_spatial_enabled:
        result["patch_spatial_raw"] = local_raw.patch_spatial
    if local_temporal_enabled:
        result["patch_temporal_raw"] = local_raw.patch_temporal
    return result


__all__ = ["score_window_components"]
