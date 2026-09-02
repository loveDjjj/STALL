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


def _trajectory_geometry(velocity: torch.Tensor) -> dict[str, torch.Tensor]:
    """从相邻速度构造对齐到 `[B,T-2,P]` 的低维轨迹几何量。"""

    if velocity.ndim != 4 or velocity.shape[1] < 2:
        raise ValueError(
            f"轨迹几何要求 velocity [B,T-1,P,D] 且 T>=3，实际 {tuple(velocity.shape)}"
        )
    previous = velocity[:, :-1]
    current = velocity[:, 1:]
    previous_speed = torch.linalg.vector_norm(previous, dim=-1)
    current_speed = torch.linalg.vector_norm(current, dim=-1)
    epsilon = 1e-6
    valid_turn = (previous_speed > epsilon) & (current_speed > epsilon)
    cosine = F.cosine_similarity(previous, current, dim=-1, eps=epsilon)
    curvature = torch.where(valid_turn, 1.0 - cosine, torch.zeros_like(cosine))
    speed_ratio = torch.log(
        (current_speed + epsilon) / (previous_speed + epsilon)
    )
    chord = torch.linalg.vector_norm(previous + current, dim=-1)
    path_chord = (
        (previous_speed + current_speed) / (chord + epsilon) - 1.0
    )
    log_speed = 0.5 * (
        torch.log(previous_speed + epsilon) + torch.log(current_speed + epsilon)
    )
    return {
        "log_speed": log_speed,
        "curvature": curvature,
        "speed_ratio": speed_ratio,
        "path_chord": path_chord,
    }


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
    """按配置构造有限差分或轨迹几何，校准与评测共用此入口。"""

    values = torch.as_tensor(patch, dtype=torch.float32, device=device)
    order = int(local_config["temporal_order"])
    correspondence = _correspondence_config(local_config)
    mode = correspondence["type"]
    confidence_mode = correspondence["confidence"]
    dynamics = str(local_config.get("dynamics", "finite_difference"))
    if confidence_mode not in {"none", "aggregation"}:
        raise ValueError("correspondence.confidence 只能是 none 或 aggregation")

    # C0 直接调用历史函数，保证相同输入下逐位复现旧 same-grid 路径。
    if mode == "same_grid" and dynamics == "finite_difference":
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
    if dynamics == "finite_difference" and order == 1:
        features = F.normalize(aligned.velocity, p=2, dim=-1, eps=1e-12)
        confidence = aligned.confidence
    elif dynamics == "finite_difference" and order == 2:
        acceleration = aligned.velocity[:, 1:] - aligned.velocity[:, :-1]
        features = F.normalize(acceleration, p=2, dim=-1, eps=1e-12)
        # 一个 D2 位置依赖两条相邻匹配边，保守取两者较小置信度。
        confidence = (
            torch.minimum(aligned.confidence[:, 1:], aligned.confidence[:, :-1])
            if aligned.confidence is not None
            else None
        )
    elif dynamics == "finite_difference":
        raise ValueError("method.local.temporal_order 只能是 1 或 2")
    else:
        geometry = _trajectory_geometry(aligned.velocity)
        if dynamics in {"curvature", "speed_ratio", "path_chord"}:
            features = geometry[dynamics].unsqueeze(-1)
        elif dynamics == "d2_curvature":
            acceleration = aligned.velocity[:, 1:] - aligned.velocity[:, :-1]
            normalized_d2 = F.normalize(
                acceleration, p=2, dim=-1, eps=1e-12
            )
            features = torch.cat(
                [normalized_d2, geometry["curvature"].unsqueeze(-1)], dim=-1
            )
        elif dynamics == "geometry":
            features = torch.stack(
                [
                    geometry["log_speed"],
                    geometry["curvature"],
                    geometry["speed_ratio"],
                    geometry["path_chord"],
                ],
                dim=-1,
            )
        else:
            raise ValueError(f"不支持的 Local dynamics：{dynamics}")
        confidence = (
            torch.minimum(aligned.confidence[:, 1:], aligned.confidence[:, :-1])
            if aligned.confidence is not None
            else None
        )

    weights = confidence if confidence_mode == "aggregation" else None
    if confidence_mode == "aggregation" and weights is None:
        raise ValueError("当前 correspondence 模式不提供置信度")
    return LocalDynamicsResult(
        features=features.cpu(),
        aggregation_weights=weights.cpu() if weights is not None else None,
    )


__all__ = ["LocalDynamicsResult", "build_local_dynamics"]
