"""在冻结 patch token 上构造确定性的局部时间对应。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CorrespondenceResult:
    """相邻帧对应后的速度与可选置信度。"""

    velocity: torch.Tensor
    confidence: torch.Tensor | None


def _local_neighborhood(
    grid_size: tuple[int, int], radius: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回每个网格位置的候选索引、有效掩码和归一化空间距离。"""

    height, width = grid_size
    if height < 1 or width < 1:
        raise ValueError(f"无效 patch 网格：{grid_size}")
    if radius < 1:
        raise ValueError("局部匹配半径必须至少为 1")
    positions = torch.arange(height * width, device=device)
    rows = torch.div(positions, width, rounding_mode="floor")
    columns = positions % width
    indices, validity, distances = [], [], []
    scale = float(max(radius, 1))
    for delta_row in range(-radius, radius + 1):
        for delta_column in range(-radius, radius + 1):
            candidate_row = rows + delta_row
            candidate_column = columns + delta_column
            valid = (
                (candidate_row >= 0)
                & (candidate_row < height)
                & (candidate_column >= 0)
                & (candidate_column < width)
            )
            candidate_row = candidate_row.clamp(0, height - 1)
            candidate_column = candidate_column.clamp(0, width - 1)
            indices.append(candidate_row * width + candidate_column)
            validity.append(valid)
            distances.append(
                torch.full(
                    (height * width,),
                    math.hypot(delta_row, delta_column) / scale,
                    dtype=torch.float32,
                    device=device,
                )
            )
    return (
        torch.stack(indices, dim=1),
        torch.stack(validity, dim=1),
        torch.stack(distances, dim=1),
    )


def _pair_scores(
    source: torch.Tensor,
    target: torch.Tensor,
    indices: torch.Tensor,
    valid: torch.Tensor,
    distances: torch.Tensor,
    spatial_penalty: float,
) -> torch.Tensor:
    """分候选计算余弦匹配分数，避免构造 `[B,T,P,K,D]` 大张量。"""

    source_normalized = F.normalize(source, p=2, dim=-1, eps=1e-12)
    scores = []
    for candidate_index in range(indices.shape[1]):
        candidate = target.index_select(2, indices[:, candidate_index])
        similarity = torch.sum(
            source_normalized * F.normalize(candidate, p=2, dim=-1, eps=1e-12),
            dim=-1,
        )
        score = similarity - spatial_penalty * distances[:, candidate_index]
        scores.append(score.masked_fill(~valid[:, candidate_index], float("-inf")))
    return torch.stack(scores, dim=-1)


@torch.inference_mode()
def align_local_velocity(
    patch: torch.Tensor,
    *,
    grid_size: tuple[int, int],
    mode: str,
    radius: int = 1,
    temperature: float = 0.07,
    spatial_penalty: float = 0.05,
) -> CorrespondenceResult:
    """构造相邻帧速度；soft 模式同时返回归一化熵置信度。

    输入和速度分别为 `[B,T,P,D]` 与 `[B,T-1,P,D]`。局部搜索只在
    `grid_size` 定义的邻域内进行，不读取标签或生成器信息。
    """

    if patch.ndim != 4 or patch.shape[1] < 2:
        raise ValueError(f"期望 patch [B,T,P,D] 且 T>=2，实际 {tuple(patch.shape)}")
    if patch.shape[2] != grid_size[0] * grid_size[1]:
        raise ValueError(
            f"patch 数量 {patch.shape[2]} 与网格 {grid_size} 不一致"
        )
    if mode == "same_grid":
        return CorrespondenceResult(
            velocity=patch[:, 1:] - patch[:, :-1], confidence=None
        )
    if mode not in {"hard_local", "soft_local"}:
        raise ValueError(f"不支持的局部对应模式：{mode}")
    if temperature <= 0:
        raise ValueError("soft matching temperature 必须为正数")
    if spatial_penalty < 0:
        raise ValueError("spatial_penalty 不能为负数")

    source = patch[:, :-1]
    target = patch[:, 1:]
    indices, valid, distances = _local_neighborhood(
        grid_size, radius, patch.device
    )
    scores = _pair_scores(
        source, target, indices, valid, distances, spatial_penalty
    )

    if mode == "hard_local":
        best = torch.argmax(scores, dim=-1)
        aligned = torch.zeros_like(source)
        for candidate_index in range(indices.shape[1]):
            candidate = target.index_select(2, indices[:, candidate_index])
            aligned = torch.where(
                (best == candidate_index).unsqueeze(-1), candidate, aligned
            )
        return CorrespondenceResult(velocity=aligned - source, confidence=None)

    probabilities = torch.softmax(scores / temperature, dim=-1)
    aligned = torch.zeros_like(source)
    for candidate_index in range(indices.shape[1]):
        candidate = target.index_select(2, indices[:, candidate_index])
        aligned.add_(probabilities[..., candidate_index].unsqueeze(-1) * candidate)

    entropy = -torch.sum(
        probabilities * torch.log(probabilities.clamp_min(1e-12)), dim=-1
    )
    valid_count = valid.sum(dim=1).to(dtype=entropy.dtype)
    normalizer = torch.log(valid_count).view(1, 1, -1)
    confidence = torch.where(
        normalizer > 0,
        1.0 - entropy / normalizer,
        torch.ones_like(entropy),
    ).clamp(0.0, 1.0)
    return CorrespondenceResult(
        velocity=aligned - source,
        confidence=confidence,
    )


__all__ = ["CorrespondenceResult", "align_local_velocity"]
