"""只用真实校准数据建立和评分运动状态条件 Gaussian。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from math_utils import StableGaussianParams


@dataclass(frozen=True)
class ConditionalGaussianParams:
    """按升序真实分位边界组织的多个 Gaussian 参数。"""

    boundaries: np.ndarray
    bins: tuple[StableGaussianParams, ...]
    state_name: str = "current_speed"

    def __post_init__(self) -> None:
        boundaries = np.asarray(self.boundaries, dtype=np.float64)
        if boundaries.ndim != 1 or len(self.bins) != len(boundaries) + 1:
            raise ValueError("条件 Gaussian 的边界数量与 bin 参数不一致")
        if not np.isfinite(boundaries).all() or np.any(np.diff(boundaries) < 0):
            raise ValueError("条件 Gaussian 边界必须是有限非降序列")


def assign_condition_bins(
    state: np.ndarray | torch.Tensor, boundaries: np.ndarray
) -> np.ndarray:
    """右侧边界归入更快区间，规则对 calibration/test 完全一致。"""

    values = np.asarray(torch.as_tensor(state).cpu(), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("条件状态包含非有限值")
    return np.searchsorted(
        np.asarray(boundaries, dtype=np.float64), values, side="right"
    ).astype(np.int64)


@torch.inference_mode()
def score_conditional_gaussian_mean_float64(
    features: torch.Tensor | np.ndarray,
    state: torch.Tensor | np.ndarray,
    params: ConditionalGaussianParams,
    *,
    device: str | torch.device,
) -> np.ndarray:
    """逐位置选择真实运动 bin 的 Gaussian，再对窗口内 likelihood 取均值。"""

    values = torch.as_tensor(features, dtype=torch.float32, device="cpu")
    states = torch.as_tensor(state, dtype=torch.float32, device="cpu")
    if values.ndim < 3 or tuple(states.shape) != tuple(values.shape[:-1]):
        raise ValueError(
            f"条件特征/状态 shape 不一致：{tuple(values.shape)} 与 {tuple(states.shape)}"
        )
    flat = values.reshape(-1, values.shape[-1])
    assignments = torch.from_numpy(
        assign_condition_bins(states, params.boundaries).reshape(-1)
    )
    target = torch.device(device)
    likelihood = torch.empty(len(flat), dtype=torch.float64, device=target)
    for bin_index, gaussian in enumerate(params.bins):
        selected_cpu = torch.nonzero(assignments == bin_index, as_tuple=False).flatten()
        if not len(selected_cpu):
            continue
        selected = flat.index_select(0, selected_cpu).to(target, dtype=torch.float64)
        mean = torch.as_tensor(gaussian.mean, dtype=torch.float64, device=target)
        whitening = torch.as_tensor(
            gaussian.whitening, dtype=torch.float64, device=target
        )
        white = (selected - mean) @ whitening
        constant = float(whitening.shape[1]) * np.log(2.0 * np.pi)
        scores = -0.5 * (constant + torch.sum(white * white, dim=-1))
        likelihood.index_copy_(0, selected_cpu.to(target), scores)
        # 三个高维 bin 顺序评分；结果已写回小型 likelihood 向量后立即释放
        # 选中特征和白化中间量，防止 CUDA allocator 在同一 batch 内累积。
        del selected, mean, whitening, white, scores
        if target.type == "cuda":
            torch.cuda.empty_cache()
    return (
        likelihood.reshape(len(values), -1)
        .mean(dim=1, dtype=torch.float64)
        .cpu()
        .numpy()
    )


__all__ = [
    "ConditionalGaussianParams",
    "assign_condition_bins",
    "score_conditional_gaussian_mean_float64",
]
