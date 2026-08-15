"""Two-level calibration and video aggregation for the locked U0 protocol."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .aggregation import selected_video_means
from .calibration import (
    U0VideoReferences,
    U0WindowReferences,
    calibrate_u0_video_branches,
    calibrate_u0_window_components,
)
from .parameters import global_references
from .u0_protocol import (
    EXPECTED_EVALUATION,
    KEY_COLUMNS,
    load_calibration_references,
    selected_calibration_means,
)
from .whitening import empirical_cdf_right_inclusive, stable_sorted


def calibration_raw_references(
    calibration_raw_dir: Path, num_shards: int, dataset: str, config: dict
) -> dict[str, np.ndarray]:
    """Build the locked four-component K1 real calibration references."""

    calibration = load_calibration_references(calibration_raw_dir, num_shards)
    calibration = calibration[calibration["dataset"] == dataset]
    if len(calibration) != 200:
        raise ValueError(f"{dataset}: expected 200 K1 calibration references")
    global_spatial, global_t1 = global_references(config)
    return {
        "global_spatial": global_spatial,
        "global_t1": global_t1,
        "patch_spatial": stable_sorted(calibration["patch_spatial_raw"].to_numpy()),
        "patch_d2": stable_sorted(calibration["patch_d2_raw"].to_numpy()),
    }


def calibrate_raw(
    raw: dict[str, np.ndarray], references: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Calibrate raw K1 components while preserving historical column names."""

    calibrated = calibrate_u0_window_components(
        raw["global_spatial_raw"],
        raw["global_t1_raw"],
        raw["patch_spatial_raw"],
        raw["patch_d2_raw"],
        U0WindowReferences(
            references["global_spatial"],
            references["global_t1"],
            references["patch_spatial"],
            references["patch_d2"],
        ),
    )
    calibrated["patch_d2"] = calibrated.pop("patch_temporal")
    calibrated.pop("S_k")
    return calibrated


def selected_k_reference(
    windows: pd.DataFrame, dataset: str, target_k: int, column: str
) -> np.ndarray:
    """Return the sorted locked calibration-video means for one effective K."""

    calibration = windows[
        (windows["dataset"] == dataset)
        & (windows["protocol_split"] == "calibration")
    ]
    try:
        return effective_k_reference(calibration, target_k, column)
    except ValueError as error:
        raise ValueError(
            f"no K={target_k} calibration reference for {dataset}/{column}"
        ) from error


def effective_k_reference(
    windows: pd.DataFrame,
    target_k: int,
    column: str,
    *,
    selected_video_ids: set[str] | None = None,
) -> np.ndarray:
    """Return sorted per-video means for one effective-K real reference bank."""

    values = selected_video_means(
        windows,
        target_k,
        {"score": column},
        selected_video_ids=selected_video_ids,
    )
    if len(values) < 2:
        raise ValueError(f"insufficient K={target_k} reference values for {column}")
    return stable_sorted(values["score"].to_numpy())


def calibrate_k3_candidate(windows: pd.DataFrame, column: str) -> pd.DataFrame:
    """Aggregate and effective-K calibrate one locked K3 candidate column."""

    per_video = (
        windows.groupby(list(KEY_COLUMNS), sort=False, observed=True)
        .agg(effective_k=("effective_k", "first"), raw=(column, "mean"))
        .reset_index()
    )
    output = []
    evaluation = per_video[per_video["protocol_split"] == "evaluation"]
    for (dataset, effective_k), target in evaluation.groupby(
        ["dataset", "effective_k"], sort=False
    ):
        reference = selected_k_reference(
            windows, str(dataset), int(effective_k), column
        )
        target = target.copy()
        target["score"] = empirical_cdf_right_inclusive(
            target["raw"].to_numpy(), reference
        )
        output.append(target[["video_id", "score"]])
    result = pd.concat(output, ignore_index=True)
    expected = sum(EXPECTED_EVALUATION.values())
    if len(result) != expected or result["video_id"].duplicated().any():
        raise ValueError(f"invalid K3 candidate aggregation: {column}")
    return result


def calibrate_windows(
    windows: pd.DataFrame,
    calibration_references: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate four raw window components and apply frozen branch fusion."""

    global_spatial_ref, global_t1_ref = global_references(config)
    output = []
    references = []
    for dataset, frame in windows.groupby("dataset", sort=False):
        target = frame.copy()
        calibration = calibration_references[
            calibration_references["dataset"] == dataset
        ]
        if len(calibration) != 200:
            raise ValueError(f"{dataset}: expected 200 local calibration windows")
        patch_spatial_ref = stable_sorted(calibration["patch_spatial_raw"].to_numpy())
        patch_d2_ref = stable_sorted(calibration["patch_d2_raw"].to_numpy())
        calibrated = calibrate_u0_window_components(
            target["global_spatial_raw"].to_numpy(),
            target["global_t1_raw"].to_numpy(),
            target["patch_spatial_raw"].to_numpy(),
            target["patch_d2_raw"].to_numpy(),
            U0WindowReferences(
                global_spatial_ref,
                global_t1_ref,
                patch_spatial_ref,
                patch_d2_ref,
            ),
        )
        for column, values in calibrated.items():
            target["patch_d2" if column == "patch_temporal" else column] = values
        output.append(target)
        for branch, values in (
            ("patch_spatial", patch_spatial_ref),
            ("patch_d2", patch_d2_ref),
        ):
            references.extend(
                {
                    "dataset": dataset,
                    "branch": branch,
                    "rank": rank,
                    "raw_score": value,
                }
                for rank, value in enumerate(values)
            )
    if not output:
        raise ValueError("no raw U0 windows to calibrate")
    return pd.concat(output, ignore_index=True), pd.DataFrame(references)


def aggregate_and_calibrate_videos(
    windows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average windows per video and apply effective-K-matched real CDFs."""

    per_video = (
        windows.groupby(list(KEY_COLUMNS), sort=False, observed=True)
        .agg(
            video_path=("video_path", "first"),
            duration_seconds=("duration_seconds", "first"),
            effective_k=("effective_k", "first"),
            window_count=("window_id", "size"),
            G_raw=("G_k", "mean"),
            L_raw=("L_k", "mean"),
        )
        .reset_index()
    )
    if not (per_video["effective_k"] == per_video["window_count"]).all():
        raise ValueError("effective_k does not match raw window count")
    frames = []
    reference_rows = []
    for dataset, dataset_videos in per_video.groupby("dataset", sort=False):
        dataset_windows = windows[windows["dataset"] == dataset]
        calibration_windows = dataset_windows[
            dataset_windows["protocol_split"] == "calibration"
        ]
        evaluation = dataset_videos[
            dataset_videos["protocol_split"] == "evaluation"
        ].copy()
        for effective_k, target in evaluation.groupby("effective_k", sort=True):
            reference = selected_calibration_means(
                calibration_windows, int(effective_k)
            )
            g_reference = stable_sorted(reference["G_raw"].to_numpy())
            l_reference = stable_sorted(reference["L_raw"].to_numpy())
            target = target.copy()
            calibrated = calibrate_u0_video_branches(
                target["G_raw"].to_numpy(),
                target["L_raw"].to_numpy(),
                U0VideoReferences(g_reference, l_reference),
            )
            for column, values in calibrated.items():
                target[column] = values
            frames.append(target)
            reference_rows.append(
                {
                    "dataset": dataset,
                    "effective_k": int(effective_k),
                    "calibration_videos": len(reference),
                    "global_min": float(g_reference.min()),
                    "global_max": float(g_reference.max()),
                    "local_min": float(l_reference.min()),
                    "local_max": float(l_reference.max()),
                }
            )
    if not frames:
        raise ValueError("no U0 evaluation videos to aggregate")
    return pd.concat(frames, ignore_index=True), pd.DataFrame(reference_rows)


__all__ = [
    "aggregate_and_calibrate_videos",
    "calibrate_windows",
    "effective_k_reference",
    "selected_k_reference",
]
