"""Local D2位置级似然场与预注册Tail/CVaR聚合。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from branches.global_branch import (
    GLOBAL_SPATIAL_AGGREGATION,
    GLOBAL_TEMPORAL_AGGREGATION,
)
from branches.local_branch import local_d2_features
from math_utils import (
    StableGaussianParams,
    l2_normalized_first_order,
    score_gaussian_aggregate_float64,
)
from temporal_selection.models import CandidateWindow, SelectedWindow, WindowManifest


TAIL_RATIOS = {
    "mean": 1.0,
    "worst_05": 0.05,
    "cvar_10": 0.10,
    "cvar_20": 0.20,
}


@dataclass(frozen=True)
class TailWindowScores:
    """一个窗口batch的Global分量、Local聚合与float32似然场。"""

    global_spatial_raw: np.ndarray
    global_spatial: np.ndarray
    global_t1_raw: np.ndarray
    global_t1: np.ndarray
    local_raw: dict[str, np.ndarray]
    local_likelihood_field: np.ndarray


@dataclass(frozen=True)
class TailWindowUse:
    """一个selector/calibration协议对某个物理窗口的使用。"""

    selector: str
    calibration_mode: str
    candidate: CandidateWindow
    selection: SelectedWindow


@dataclass(frozen=True)
class TailWindowRequest:
    """跨selector与standard/crossfit去重后的物理窗口。"""

    frame_indices: tuple[int, ...]
    uses: tuple[TailWindowUse, ...]


def merge_tail_window_requests(
    manifests: Mapping[tuple[str, str], WindowManifest],
) -> list[TailWindowRequest]:
    """按完整帧索引合并多个selector/calibration协议的selected窗口。"""

    if not manifests:
        return []
    video_ids = {manifest.video_id for manifest in manifests.values()}
    if len(video_ids) != 1:
        raise ValueError("Tail请求包含不同video_id")
    grouped: dict[tuple[int, ...], list[TailWindowUse]] = {}
    for (calibration_mode, selector), manifest in manifests.items():
        manifest.validate()
        if calibration_mode not in {"standard", "crossfit5"}:
            raise ValueError("Tail calibration_mode无效")
        if manifest.selector.get("name") != selector:
            raise ValueError("Tail manifest selector名称与映射键不一致")
        candidates = {item.candidate_id: item for item in manifest.candidates}
        for selection in manifest.selected:
            candidate = candidates[selection.candidate_id]
            grouped.setdefault(candidate.frame_indices, []).append(
                TailWindowUse(
                    selector=selector,
                    calibration_mode=calibration_mode,
                    candidate=candidate,
                    selection=selection,
                )
            )
    return [
        TailWindowRequest(frame_indices=frames, uses=tuple(uses))
        for frames, uses in sorted(grouped.items(), key=lambda item: item[0][0])
    ]


@torch.inference_mode()
def local_d2_likelihood_fields(
    patch_windows: np.ndarray | torch.Tensor,
    parameters: StableGaussianParams,
    *,
    device: str | torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """计算逐D2位置似然，并在float64上完成Mean与下尾均值。"""

    features = local_d2_features(torch.as_tensor(patch_windows, dtype=torch.float32))
    if features.ndim != 4:
        raise ValueError(f"Local D2 field期望[N,T,P,D]，实际{tuple(features.shape)}")
    target = torch.device(device)
    mean = torch.as_tensor(parameters.mean, dtype=torch.float64, device=target)
    whitening = torch.as_tensor(
        parameters.whitening, dtype=torch.float64, device=target
    )
    flat = features.reshape(len(features), -1, features.shape[-1]).to(
        target, dtype=torch.float64
    )
    white = torch.matmul(flat - mean, whitening)
    constant = float(whitening.shape[1]) * np.log(2.0 * np.pi)
    likelihood = -0.5 * (constant + torch.sum(white * white, dim=-1))
    if not torch.isfinite(likelihood).all():
        raise ValueError("Local D2 likelihood field包含非有限值")
    aggregates = {}
    for name, ratio in TAIL_RATIOS.items():
        if name == "mean":
            value = likelihood.mean(dim=1, dtype=torch.float64)
        else:
            count = max(1, int(np.ceil(likelihood.shape[1] * ratio)))
            value = torch.topk(
                likelihood, k=count, dim=1, largest=False, sorted=False
            ).values.mean(dim=1, dtype=torch.float64)
        aggregates[name] = value.cpu().numpy().astype(np.float64, copy=False)
    field = likelihood.reshape(*features.shape[:-1]).cpu().numpy().astype(
        np.float32, copy=False
    )
    return field, aggregates


def score_tail_windows(
    global_windows: np.ndarray,
    patch_windows: np.ndarray,
    *,
    global_parameters: dict[str, StableGaussianParams],
    local_parameters: StableGaussianParams,
    device: str | torch.device,
) -> TailWindowScores:
    """对固定窗口计算官方Global与Local D2 Tail候选。"""

    if len(global_windows) != len(patch_windows) or not len(global_windows):
        raise ValueError("Tail评分要求非空且对齐的Global/Patch窗口batch")
    spatial_raw, spatial = score_gaussian_aggregate_float64(
        global_windows,
        global_parameters["global_spatial"],
        GLOBAL_SPATIAL_AGGREGATION,
        device=device,
    )
    temporal_features, zero_mask = l2_normalized_first_order(
        torch.from_numpy(global_windows)
    )
    temporal_raw, temporal = score_gaussian_aggregate_float64(
        temporal_features,
        global_parameters["global_t1"],
        GLOBAL_TEMPORAL_AGGREGATION,
        device=device,
        invalid_mask=zero_mask,
        allow_positive_infinity_percentile=True,
    )
    field, local_raw = local_d2_likelihood_fields(
        patch_windows, local_parameters, device=device
    )
    return TailWindowScores(
        global_spatial_raw=spatial_raw,
        global_spatial=spatial,
        global_t1_raw=temporal_raw,
        global_t1=temporal,
        local_raw=local_raw,
        local_likelihood_field=field,
    )


__all__ = [
    "TAIL_RATIOS",
    "TailWindowScores",
    "TailWindowRequest",
    "TailWindowUse",
    "local_d2_likelihood_fields",
    "merge_tail_window_requests",
    "score_tail_windows",
]
