"""把 patch 对应转换为可由真实视频拟合的局部动力学证据。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from correspondence.local import align_local_velocity
from math_utils import l2_normalized_patch_first_order, l2_normalized_second_order


@dataclass(frozen=True)
class LocalDynamicsResult:
    """局部动力学特征和可选的 likelihood 聚合权重。"""

    features: torch.Tensor
    aggregation_weights: torch.Tensor | None


def _correspondence_config(local_config: dict) -> dict:
    """为历史 resolved config 提供与旧主方法完全一致的默认值。"""

    configured = local_config.get("correspondence", {})
    return {
        "type": str(configured.get("type", "same_grid")),
        "radius": int(configured.get("radius", 1)),
        "temperature": float(configured.get("temperature", 0.07)),
        "spatial_penalty": float(configured.get("spatial_penalty", 0.05)),
        "confidence": str(configured.get("confidence", "none")),
    }


@torch.inference_mode()
def build_local_dynamics(
    patch: torch.Tensor,
    *,
    grid_size: tuple[int, int],
    local_config: dict,
    device: str | torch.device,
) -> LocalDynamicsResult:
    """按配置构造 D1/D2，calibration 与 evaluation 必须共用此入口。"""

    values = torch.as_tensor(patch, dtype=torch.float32, device=device)
    order = int(local_config["temporal_order"])
    correspondence = _correspondence_config(local_config)
    mode = correspondence["type"]
    confidence_mode = correspondence["confidence"]
    if confidence_mode not in {"none", "aggregation"}:
        raise ValueError("correspondence.confidence 只能是 none 或 aggregation")

    # C0 直接调用历史函数，保证相同输入下逐位复现旧 same-grid 路径。
    if mode == "same_grid":
        if confidence_mode != "none":
            raise ValueError("same_grid 不产生 correspondence confidence")
        features = (
            l2_normalized_patch_first_order(values)
            if order == 1
            else l2_normalized_second_order(values)
        )
        return LocalDynamicsResult(features=features.cpu(), aggregation_weights=None)

    aligned = align_local_velocity(
        values,
        grid_size=grid_size,
        mode=mode,
        radius=correspondence["radius"],
        temperature=correspondence["temperature"],
        spatial_penalty=correspondence["spatial_penalty"],
    )
    if order == 1:
        features = F.normalize(aligned.velocity, p=2, dim=-1, eps=1e-12)
        confidence = aligned.confidence
    elif order == 2:
        acceleration = aligned.velocity[:, 1:] - aligned.velocity[:, :-1]
        features = F.normalize(acceleration, p=2, dim=-1, eps=1e-12)
        # 一个 D2 位置依赖两条相邻匹配边，保守取两者较小置信度。
        confidence = (
            torch.minimum(aligned.confidence[:, 1:], aligned.confidence[:, :-1])
            if aligned.confidence is not None
            else None
        )
    else:
        raise ValueError("method.local.temporal_order 只能是 1 或 2")

    weights = confidence if confidence_mode == "aggregation" else None
    if confidence_mode == "aggregation" and weights is None:
        raise ValueError("当前 correspondence 模式不提供置信度")
    return LocalDynamicsResult(
        features=features.cpu(),
        aggregation_weights=weights.cpu() if weights is not None else None,
    )


__all__ = ["LocalDynamicsResult", "build_local_dynamics"]
