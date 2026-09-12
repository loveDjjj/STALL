"""Numerically stable whitening and empirical-CDF scoring primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


CDF_TIE_POLICY = "right_inclusive"


def stable_sorted(values: np.ndarray) -> np.ndarray:
    """Return a stable float64 sort used by every release CDF."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"CDF reference must be one-dimensional, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("CDF reference contains non-finite values")
    return np.sort(array, kind="mergesort")


def empirical_cdf_right_inclusive(
    scores: np.ndarray, reference_sorted: np.ndarray
) -> np.ndarray:
    """Compute P(reference <= score), so exact ties are included on the right."""
    reference = np.asarray(reference_sorted, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    if reference.ndim != 1 or len(reference) == 0:
        raise ValueError("CDF reference must be a non-empty one-dimensional array")
    if not np.isfinite(values).all() or not np.isfinite(reference).all():
        raise ValueError("CDF inputs contain non-finite values")
    return np.searchsorted(reference, values, side="right") / float(len(reference))


def empirical_cdf_with_positive_infinity(
    scores: np.ndarray, reference_sorted: np.ndarray
) -> np.ndarray:
    """计算右包含经验 CDF，并将协议定义的 ``+inf`` 映射为 1。

    Global T1 对严格为零的相邻帧差分不定义方向。原始 STALL 在最小
    似然聚合前将这些位置置为 ``+inf``，使其不影响其他有效转移；若
    一个窗口全部为零差分，则其聚合值仍为 ``+inf``，对应最高 CDF。
    ``NaN`` 与 ``-inf`` 不属于该协议，必须继续显式报错。
    """

    values = np.asarray(scores, dtype=np.float64)
    if np.isnan(values).any() or np.isneginf(values).any():
        raise ValueError("CDF values contain NaN or negative infinity")
    result = np.ones(len(values), dtype=np.float64)
    finite = np.isfinite(values)
    result[finite] = empirical_cdf_right_inclusive(
        values[finite], reference_sorted
    )
    return result


def l2_normalized_second_order(patch: torch.Tensor) -> torch.Tensor:
    """Compute same-grid D2 in the input dtype, then feature-wise L2 normalize."""
    if patch.ndim != 4 or patch.shape[1] < 3:
        raise ValueError(f"expected [N,T,P,D] with T>=3, got {tuple(patch.shape)}")
    acceleration = patch[:, 2:] - 2.0 * patch[:, 1:-1] + patch[:, :-2]
    return torch.nn.functional.normalize(
        acceleration, p=2, dim=-1, eps=1e-12
    )


def l2_normalized_first_order(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized lag-1 differences and their exact-zero mask."""
    if features.ndim != 3 or features.shape[1] < 2:
        raise ValueError(f"expected [N,T,D] with T>=2, got {tuple(features.shape)}")
    differences = features[:, 1:] - features[:, :-1]
    zero_mask = torch.linalg.vector_norm(differences, dim=-1) == 0
    normalized = torch.nn.functional.normalize(
        differences, p=2, dim=-1, eps=1e-12
    )
    return normalized, zero_mask


@dataclass(frozen=True)
class StableGaussianParams:
    mean: np.ndarray
    whitening: np.ndarray
    calibration_raw: np.ndarray
    shrinkage: float | None = None


@torch.inference_mode()
def score_gaussian_aggregate_float64(
    features: torch.Tensor | np.ndarray,
    params: StableGaussianParams,
    aggregation: str,
    device: str | torch.device = "cuda:0",
    invalid_mask: torch.Tensor | np.ndarray | None = None,
    position_weights: torch.Tensor | np.ndarray | None = None,
    compute_percentile: bool = True,
    allow_positive_infinity_percentile: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Score each sample independently with a fixed-shape float64 GEMM.

    `features` is `[N,...,D]`. All non-feature dimensions are flattened within
    a sample before applying mean, min, or max likelihood aggregation. An
    optional mask marks likelihood positions as positive infinity, matching
    STALL's treatment of exact zero temporal differences under min aggregation.
    `position_weights` 只允许用于 mean 聚合，并在每个样本内部归一化。
    ``allow_positive_infinity_percentile`` 仅供该 Global T1 协议使用。
    """
    values = torch.as_tensor(features, dtype=torch.float32, device="cpu")
    if values.ndim < 3:
        raise ValueError(f"features must be [N,...,D], got {tuple(values.shape)}")
    if aggregation not in {"mean", "min", "max"}:
        raise ValueError(f"unsupported aggregation: {aggregation}")
    mask = None
    if invalid_mask is not None:
        mask = torch.as_tensor(invalid_mask, dtype=torch.bool, device="cpu")
        if tuple(mask.shape) != tuple(values.shape[:-1]):
            raise ValueError(
                f"mask shape {tuple(mask.shape)} != feature positions {tuple(values.shape[:-1])}"
            )
    weights = None
    if position_weights is not None:
        if aggregation != "mean":
            raise ValueError("position_weights 只支持 mean 聚合")
        weights = torch.as_tensor(position_weights, dtype=torch.float64, device="cpu")
        if tuple(weights.shape) != tuple(values.shape[:-1]):
            raise ValueError(
                f"weight shape {tuple(weights.shape)} != feature positions {tuple(values.shape[:-1])}"
            )
        if not torch.isfinite(weights).all() or torch.any(weights < 0):
            raise ValueError("position_weights 必须是有限非负数")

    target = torch.device(device)
    mean = torch.as_tensor(params.mean, dtype=torch.float64, device=target)
    whitening = torch.as_tensor(
        params.whitening, dtype=torch.float64, device=target
    )
    constant = float(whitening.shape[1]) * np.log(2.0 * np.pi)
    # 批量传输和一次矩阵乘法让 GPU 看到足够大的工作单元；每个样本仍沿自身
    # 位置维度独立做相同的 mean/min/max 聚合，不改变视频或窗口的统计定义。
    shape = values.shape
    flat = values.reshape(len(values), -1, shape[-1]).to(device=target, dtype=torch.float64)
    white = torch.matmul(flat - mean, whitening)
    likelihood = -0.5 * (constant + torch.sum(white * white, dim=-1))
    if mask is not None:
        likelihood = likelihood.masked_fill(mask.reshape(len(values), -1).to(target), float("inf"))
    if aggregation == "mean":
        if weights is None:
            aggregate = likelihood.mean(dim=1, dtype=torch.float64)
        else:
            flat_weights = weights.reshape(len(values), -1).to(target)
            denominators = flat_weights.sum(dim=1)
            if torch.any(denominators <= 0):
                raise ValueError("每个样本的 position_weights 权重和必须为正")
            aggregate = torch.sum(likelihood * flat_weights, dim=1) / denominators
    elif aggregation == "min":
        aggregate = likelihood.amin(dim=1)
    else:
        aggregate = likelihood.amax(dim=1)
    raw = aggregate.cpu().numpy().astype(np.float64, copy=False)
    if compute_percentile:
        percentile = (
            empirical_cdf_with_positive_infinity(raw, params.calibration_raw)
            if allow_positive_infinity_percentile
            else empirical_cdf_right_inclusive(raw, params.calibration_raw)
        )
    else:
        percentile = np.full(len(raw), np.nan, dtype=np.float64)
    return raw, percentile


class GaussianMeanCandidateScorerFloat64:
    """Precomputed float64 scorer for several mean-aggregated Gaussians."""

    def __init__(
        self,
        candidates: list[StableGaussianParams],
        center: np.ndarray,
        device: str | torch.device = "cuda:0",
    ) -> None:
        if not candidates:
            raise ValueError("at least one Gaussian candidate is required")
        self.device = torch.device(device)
        self.dimension = int(candidates[0].mean.shape[0])
        reference = np.asarray(center, dtype=np.float64)
        if reference.shape != (self.dimension,):
            raise ValueError(f"center shape {reference.shape} != ({self.dimension},)")
        self.center = torch.as_tensor(reference, dtype=torch.float64, device=self.device)
        means = torch.stack(
            [
                torch.as_tensor(item.mean, dtype=torch.float64, device=self.device)
                for item in candidates
            ]
        )
        if tuple(means.shape) != (len(candidates), self.dimension):
            raise ValueError(f"candidate means have incompatible shape {tuple(means.shape)}")
        whitening = [
            torch.as_tensor(item.whitening, dtype=torch.float64, device=self.device)
            for item in candidates
        ]
        self.precisions = torch.stack([matrix @ matrix.T for matrix in whitening])
        ranks = torch.as_tensor(
            [matrix.shape[1] for matrix in whitening],
            dtype=torch.float64,
            device=self.device,
        )
        offsets = self.center.unsqueeze(0) - means
        self.precision_offsets = torch.einsum(
            "kde,ke->kd", self.precisions, offsets
        )
        self.offset_quadratic = torch.sum(
            offsets * self.precision_offsets, dim=1
        )
        self.constants = ranks * np.log(2.0 * np.pi)

    @torch.inference_mode()
    def score(self, features: torch.Tensor | np.ndarray) -> np.ndarray:
        values = torch.as_tensor(features, dtype=torch.float32, device="cpu")
        if values.ndim < 3 or values.shape[-1] != self.dimension:
            raise ValueError(
                f"features must be [N,...,{self.dimension}], got {tuple(values.shape)}"
            )
        output = np.empty((len(values), len(self.constants)), dtype=np.float64)
        for index, sample in enumerate(values):
            flat = sample.reshape(-1, self.dimension).to(
                device=self.device, dtype=torch.float64
            )
            centered = flat - self.center
            mean_centered = centered.mean(dim=0, dtype=torch.float64)
            second_centered = torch.mm(centered.T, centered) / float(len(centered))
            trace = torch.einsum("de,ked->k", second_centered, self.precisions)
            cross = 2.0 * torch.sum(
                mean_centered.unsqueeze(0) * self.precision_offsets, dim=1
            )
            quadratic = trace + cross + self.offset_quadratic
            output[index] = (-0.5 * (self.constants + quadratic)).cpu().numpy()
        return output


__all__ = [
    "CDF_TIE_POLICY",
    "GaussianMeanCandidateScorerFloat64",
    "StableGaussianParams",
    "WhiteningTransform",
    "apply_whitening",
    "bottomk_mean",
    "configure_strict_fp32",
    "empirical_cdf_right_inclusive",
    "empirical_cdf_with_positive_infinity",
    "l2_normalized_first_order",
    "l2_normalized_patch_first_order",
    "l2_normalized_second_order",
    "log_likelihood",
    "score_gaussian_aggregate_float64",
    "score_gaussian_mean_candidates_float64",
    "score_mean_gaussian_float64",
    "score_mean_gaussian_fp32",
    "stable_sorted",
]
