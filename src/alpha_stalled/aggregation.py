"""Deterministic effective-K selection and per-video aggregation primitives."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

import numpy as np
import pandas as pd


def effective_k_positions(window_count: int, target_k: int) -> np.ndarray:
    """Return locked midpoint/endpoint positions for an effective-K reference."""
    if window_count < 0:
        raise ValueError("window_count must be non-negative")
    if target_k < 1:
        raise ValueError("target_k must be positive")
    if window_count < target_k:
        return np.empty(0, dtype=int)
    if target_k == 1:
        return np.array([(window_count - 1) // 2], dtype=int)
    positions = np.rint(np.linspace(0, window_count - 1, target_k)).astype(int)
    return np.unique(positions)


def selected_video_means(
    windows: pd.DataFrame,
    target_k: int,
    columns: Mapping[str, str],
    *,
    selected_video_ids: Collection[str] | None = None,
    video_id_column: str = "video_id",
    window_id_column: str = "window_id",
    group_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Select locked effective-K windows and average requested columns per video."""
    source = windows
    if selected_video_ids is not None:
        source = source[source[video_id_column].isin(selected_video_ids)]
    groups = tuple(group_columns) if group_columns is not None else (video_id_column,)
    if not groups:
        raise ValueError("group_columns must be non-empty")
    grouper: str | list[str] = groups[0] if len(groups) == 1 else list(groups)
    rows: list[dict[str, object]] = []
    for key, frame in source.groupby(grouper, sort=False, observed=True):
        ordered = frame.sort_values(window_id_column)
        positions = effective_k_positions(len(ordered), target_k)
        if len(positions) != target_k:
            continue
        selected = ordered.iloc[positions]
        key_values = (key,) if len(groups) == 1 else tuple(key)
        rows.append(
            {
                **dict(zip(groups, key_values)),
                **{
                    output: float(selected[input_column].mean())
                    for output, input_column in columns.items()
                },
            }
        )
    return pd.DataFrame(rows, columns=[*groups, *columns])


__all__ = ["effective_k_positions", "selected_video_means"]
