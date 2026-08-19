"""我们方法的窗口校准、视频聚合与最终分数计算。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from calibration import (
    VideoReferences,
    WindowReferences,
    calibrate_video_branches,
    calibrate_window_components,
)
from math_utils import stable_sorted


def effective_k_positions(window_count: int, target_k: int) -> np.ndarray:
    """返回固定 effective-K 的均匀窗口位置。"""

    if window_count < 0:
        raise ValueError("window_count must be non-negative")
    if target_k < 1:
        raise ValueError("target_k must be positive")
    if window_count < target_k:
        return np.empty(0, dtype=int)
    if target_k == 1:
        return np.array([(window_count - 1) // 2], dtype=int)
    return np.unique(np.rint(np.linspace(0, window_count - 1, target_k)).astype(int))


def calibrate_windows(
    windows: pd.DataFrame,
    references_by_dataset: dict[str, dict[str, np.ndarray]],
    *,
    local_spatial_weight: float,
    global_spatial_weight: float,
) -> pd.DataFrame:
    """按数据集校准窗口原始分数并写入 Global/Local 窗口分支分数。"""

    required = {
        "dataset",
        "global_spatial_raw",
        "global_t1_raw",
        "patch_spatial_raw",
        "patch_temporal_raw",
    }
    missing = required.difference(windows.columns)
    if missing:
        raise ValueError(f"窗口分数缺少字段：{sorted(missing)}")
    pieces: list[pd.DataFrame] = []
    for dataset, frame in windows.groupby("dataset", sort=False):
        try:
            reference = references_by_dataset[str(dataset)]
        except KeyError as error:
            raise ValueError(f"缺少数据集 {dataset} 的真实校准参考分数") from error
        calibrated = calibrate_window_components(
            frame["global_spatial_raw"].to_numpy(),
            frame["global_t1_raw"].to_numpy(),
            frame["patch_spatial_raw"].to_numpy(),
            frame["patch_temporal_raw"].to_numpy(),
            WindowReferences(
                stable_sorted(reference["global_spatial"]),
                stable_sorted(reference["global_t1"]),
                stable_sorted(reference["patch_spatial"]),
                stable_sorted(reference["patch_temporal"]),
            ),
        )
        output = frame.copy()
        output["global_score_window"] = (
            global_spatial_weight * calibrated["global_spatial"]
            + (1.0 - global_spatial_weight) * calibrated["global_t1"]
        )
        output["local_score_window"] = (
            local_spatial_weight * calibrated["patch_spatial"]
            + (1.0 - local_spatial_weight) * calibrated["patch_temporal"]
        )
        pieces.append(output)
    return pd.concat(pieces, ignore_index=True)


def aggregate_videos(
    windows: pd.DataFrame,
    *,
    global_weight: float,
) -> pd.DataFrame:
    """按视频均值聚合窗口，并用同一数据集、同一 effective-K 的真实视频重校准。"""

    required = {"video_id", "dataset", "subset", "window_id", "global_score_window", "local_score_window"}
    missing = required.difference(windows.columns)
    if missing:
        raise ValueError(f"窗口分数缺少字段：{sorted(missing)}")
    metadata = [column for column in ("source_model", "label", "generator") if column in windows.columns]
    aggregation = {column: (column, "first") for column in metadata}
    aggregation.update(
        effective_k=("window_id", "size"),
        global_raw=("global_score_window", "mean"),
        local_raw=("local_score_window", "mean"),
    )
    videos = windows.groupby(["video_id", "dataset", "subset"], as_index=False).agg(**aggregation)
    outputs: list[pd.DataFrame] = []
    for (dataset, effective_k), target in videos.groupby(["dataset", "effective_k"], sort=False):
        reference = target[target["subset"].eq("real")]
        evaluation = target.copy()
        if len(reference) < 2:
            raise ValueError(f"{dataset} 的 effective-K={effective_k} 真实视频不足，无法建立视频级 CDF")
        calibrated = calibrate_video_branches(
            evaluation["global_raw"].to_numpy(),
            evaluation["local_raw"].to_numpy(),
            VideoReferences(
                stable_sorted(reference["global_raw"].to_numpy()),
                stable_sorted(reference["local_raw"].to_numpy()),
            ),
        )
        evaluation["global_score"] = calibrated["G"]
        evaluation["local_score"] = calibrated["L"]
        evaluation["final_score"] = (
            global_weight * evaluation["global_score"]
            + (1.0 - global_weight) * evaluation["local_score"]
        )
        outputs.append(evaluation)
    return pd.concat(outputs, ignore_index=True)


__all__ = ["aggregate_videos", "calibrate_windows", "effective_k_positions"]
