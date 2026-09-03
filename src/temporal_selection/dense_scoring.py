"""按WindowManifest对选中窗口并集提取并计算固定Global+Local D2 raw score。"""

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
from .models import CandidateWindow, SelectedWindow, WindowManifest


@dataclass(frozen=True)
class WindowUse:
    selector: str
    candidate: CandidateWindow
    selection: SelectedWindow


@dataclass(frozen=True)
class DenseWindowRequest:
    frame_indices: tuple[int, ...]
    uses: tuple[WindowUse, ...]


def selected_window_requests(
    manifests: Mapping[str, WindowManifest],
) -> list[DenseWindowRequest]:
    """按完整frame tuple合并selector重叠窗口，拒绝video identity漂移。"""

    if not manifests:
        return []
    video_ids = {item.video_id for item in manifests.values()}
    if len(video_ids) != 1:
        raise ValueError("同一dense请求包含不同video_id")
    grouped: dict[tuple[int, ...], list[WindowUse]] = {}
    for selector, manifest in manifests.items():
        manifest.validate()
        if manifest.selector.get("name") != selector:
            raise ValueError("WindowManifest selector名称与映射键不一致")
        candidates = {item.candidate_id: item for item in manifest.candidates}
        for selection in manifest.selected:
            candidate = candidates[selection.candidate_id]
            grouped.setdefault(candidate.frame_indices, []).append(
                WindowUse(selector, candidate, selection)
            )
    return [
        DenseWindowRequest(frames, tuple(uses))
        for frames, uses in sorted(grouped.items(), key=lambda item: item[0][0])
    ]


def union_frame_indices(requests: list[DenseWindowRequest]) -> list[int]:
    """返回所有唯一原始帧索引的时间有序并集。"""

    return sorted({frame for request in requests for frame in request.frame_indices})


def request_feature_arrays(
    requests: list[DenseWindowRequest],
    *,
    extracted_frame_indices: list[int],
    global_features: np.ndarray,
    patch_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """把一次视频提取结果映射成唯一窗口batch。"""

    if len(extracted_frame_indices) != len(set(extracted_frame_indices)):
        raise ValueError("extracted frame indices重复")
    if len(global_features) != len(extracted_frame_indices) or len(patch_features) != len(extracted_frame_indices):
        raise ValueError("extracted features与frame indices长度不一致")
    positions = {frame: index for index, frame in enumerate(extracted_frame_indices)}
    global_windows, patch_windows = [], []
    for request in requests:
        try:
            selected = [positions[frame] for frame in request.frame_indices]
        except KeyError as error:
            raise ValueError("dense提取缺少selected window帧") from error
        global_windows.append(global_features[selected])
        patch_windows.append(patch_features[selected])
    if not global_windows:
        return (
            np.empty((0, 16, global_features.shape[-1]), dtype=global_features.dtype),
            np.empty((0, 16, patch_features.shape[-2], patch_features.shape[-1]), dtype=patch_features.dtype),
        )
    return np.stack(global_windows), np.stack(patch_windows)


def score_fixed_windows(
    requests: list[DenseWindowRequest],
    *,
    global_windows: np.ndarray,
    patch_windows: np.ndarray,
    global_parameters: Mapping[str, StableGaussianParams],
    local_parameters: StableGaussianParams,
    device: str,
) -> list[dict[str, object]]:
    """评分唯一窗口，并展开成各selector的标准raw记录。"""

    if len(requests) != len(global_windows) or len(requests) != len(patch_windows):
        raise ValueError("dense requests与窗口特征数量不一致")
    if not requests:
        return []
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
    local_raw, _ = score_gaussian_aggregate_float64(
        local_d2_features(torch.from_numpy(patch_windows)),
        local_parameters,
        "mean",
        device=device,
        compute_percentile=False,
    )
    records = []
    for index, request in enumerate(requests):
        for use in request.uses:
            records.append({
                "selector": use.selector,
                "candidate_id": use.candidate.candidate_id,
                "selection_rank": use.selection.rank,
                "selection_score": use.selection.score,
                "selection_reason": use.selection.reason,
                "start_seconds": use.candidate.start_seconds,
                "end_seconds": use.candidate.end_seconds,
                "center_seconds": use.candidate.center_seconds,
                "frame_indices": list(request.frame_indices),
                "global_spatial_raw": float(spatial_raw[index]),
                "global_spatial": float(spatial[index]),
                "global_t1_raw": float(temporal_raw[index]),
                "global_t1": float(temporal[index]),
                "patch_temporal_raw": float(local_raw[index]),
            })
    return records


__all__ = [
    "DenseWindowRequest",
    "WindowUse",
    "request_feature_arrays",
    "score_fixed_windows",
    "selected_window_requests",
    "union_frame_indices",
]
