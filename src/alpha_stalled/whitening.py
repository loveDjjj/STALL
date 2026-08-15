"""Numerically stable whitening and empirical-CDF scoring primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


CDF_TIE_POLICY = "right_inclusive"


def configure_strict_fp32() -> None:
    """Disable reduced-precision CUDA matmul paths for the N1 control."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


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


def l2_normalized_second_order(patch: torch.Tensor) -> torch.Tensor:
    """Compute same-grid D2 in the input dtype, then feature-wise L2 normalize."""
    if patch.ndim != 4 or patch.shape[1] < 3:
        raise ValueError(f"expected [N,T,P,D] with T>=3, got {tuple(patch.shape)}")
    acceleration = patch[:, 2:] - 2.0 * patch[:, 1:-1] + patch[:, :-2]
    return torch.nn.functional.normalize(
        acceleration, p=2, dim=-1, eps=1e-12
    )


def l2_normalized_patch_first_order(patch: torch.Tensor) -> torch.Tensor:
    """Compute same-grid patch D1, then feature-wise L2 normalize."""
    if patch.ndim != 4 or patch.shape[1] < 2:
        raise ValueError(f"expected [N,T,P,D] with T>=2, got {tuple(patch.shape)}")
    velocity = patch[:, 1:] - patch[:, :-1]
    return torch.nn.functional.normalize(velocity, p=2, dim=-1, eps=1e-12)


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

    @classmethod
    def from_npz(
        cls,
        path: str,
        mean_key: str = "mu_patch_temp",
        whitening_key: str = "W_patch_temp",
        calibration_key: str = "calib_patch_temp_scores",
    ) -> "StableGaussianParams":
        with np.load(path, allow_pickle=True) as data:
            return cls(
                mean=np.asarray(data[mean_key], dtype=np.float64),
                whitening=np.asarray(data[whitening_key], dtype=np.float64),
                calibration_raw=stable_sorted(data[calibration_key]),
            )


@torch.inference_mode()
def score_gaussian_aggregate_float64(
    features: torch.Tensor | np.ndarray,
    params: StableGaussianParams,
    aggregation: str,
    device: str | torch.device = "cuda:0",
    invalid_mask: torch.Tensor | np.ndarray | None = None,
    compute_percentile: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Score each sample independently with a fixed-shape float64 GEMM.

    `features` is `[N,...,D]`. All non-feature dimensions are flattened within
    a sample before applying mean, min, or max likelihood aggregation. An
    optional mask marks likelihood positions as positive infinity, matching
    STALL's treatment of exact zero temporal differences under min aggregation.
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

    target = torch.device(device)
    mean = torch.as_tensor(params.mean, dtype=torch.float64, device=target)
    whitening = torch.as_tensor(
        params.whitening, dtype=torch.float64, device=target
    )
    constant = float(whitening.shape[1]) * np.log(2.0 * np.pi)
    raw = np.empty(len(values), dtype=np.float64)
    for index, sample in enumerate(values):
        flat = sample.reshape(-1, sample.shape[-1]).to(
            device=target, dtype=torch.float64
        )
        white = torch.mm(flat - mean, whitening)
        likelihood = -0.5 * (constant + torch.sum(white * white, dim=1))
        if mask is not None:
            likelihood = likelihood.masked_fill(
                mask[index].reshape(-1).to(target), float("inf")
            )
        if aggregation == "mean":
            aggregate = likelihood.mean(dtype=torch.float64)
        elif aggregation == "min":
            aggregate = likelihood.amin()
        else:
            aggregate = likelihood.amax()
        raw[index] = float(aggregate.cpu())
    percentile = (
        empirical_cdf_right_inclusive(raw, params.calibration_raw)
        if compute_percentile
        else np.full(len(raw), np.nan, dtype=np.float64)
    )
    return raw, percentile


@torch.inference_mode()
def score_mean_gaussian_float64(
    temporal_features: torch.Tensor | np.ndarray,
    params: StableGaussianParams,
    device: str | torch.device = "cuda:0",
) -> tuple[np.ndarray, np.ndarray]:
    """Score each window with an invariant two-dimensional float64 GEMM.

    The leading dimension is iterated deliberately. Consequently an outer I/O
    batch size cannot change the GEMM shape or the reduction tree for a window.
    """
    return score_gaussian_aggregate_float64(
        temporal_features,
        params,
        aggregation="mean",
        device=device,
    )


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


def score_gaussian_mean_candidates_float64(
    features: torch.Tensor | np.ndarray,
    candidates: list[StableGaussianParams],
    center: np.ndarray,
    device: str | torch.device = "cuda:0",
) -> np.ndarray:
    """Score candidates using one centered first/second moment per sample."""
    return GaussianMeanCandidateScorerFloat64(
        candidates, center, device=device
    ).score(features)


@torch.inference_mode()
def score_mean_gaussian_fp32(
    temporal_features: torch.Tensor | np.ndarray,
    params: StableGaussianParams,
    device: str | torch.device = "cuda:0",
) -> tuple[np.ndarray, np.ndarray]:
    """N0/N1 control using one batched FP32 matmul and reduction."""
    target = torch.device(device)
    features = torch.as_tensor(
        temporal_features, dtype=torch.float32, device=target
    )
    mean = torch.as_tensor(params.mean, dtype=torch.float32, device=target)
    whitening = torch.as_tensor(
        params.whitening, dtype=torch.float32, device=target
    )
    white = torch.matmul(features - mean, whitening)
    constant = float(whitening.shape[1]) * np.log(2.0 * np.pi)
    likelihood = -0.5 * (constant + torch.sum(white * white, dim=-1))
    raw = likelihood.reshape(len(features), -1).mean(dim=1).cpu().numpy()
    raw64 = np.asarray(raw, dtype=np.float64)
    percentile = empirical_cdf_right_inclusive(raw64, params.calibration_raw)
    return raw64, percentile


__all__ = [
    "CDF_TIE_POLICY",
    "GaussianMeanCandidateScorerFloat64",
    "StableGaussianParams",
    "configure_strict_fp32",
    "empirical_cdf_right_inclusive",
    "l2_normalized_first_order",
    "l2_normalized_patch_first_order",
    "l2_normalized_second_order",
    "score_gaussian_aggregate_float64",
    "score_gaussian_mean_candidates_float64",
    "score_mean_gaussian_float64",
    "score_mean_gaussian_fp32",
    "stable_sorted",
]
