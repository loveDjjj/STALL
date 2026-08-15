"""Shared primitives for pre-release multi-window experiment families.

These identities and lower-tail aggregates reproduce historical multiscale,
clean-universal, and Joint Typicality studies. Locked U0 uses the stricter
``u0_protocol`` and ``u0_analysis`` modules instead.
"""

from __future__ import annotations

import pandas as pd

from .aggregation import effective_k_positions, selected_video_means


KEY_COLUMNS = [
    "dataset",
    "protocol_split",
    "subset",
    "source_model",
    "filename",
]
WINDOW_KEYS = [*KEY_COLUMNS, "window_id"]


def bottom2_mean(values: pd.Series) -> float:
    """Return the mean of the two lowest values, or all values if fewer."""

    return float(values.nsmallest(min(2, len(values))).mean())


def target_k_reference(
    window_scores: pd.DataFrame,
    target_k: int,
    local_column: str,
) -> pd.DataFrame:
    """Build Global/Local means for eligible real calibration videos."""

    calibration = window_scores[
        window_scores["protocol_split"].eq("calibration")
        & window_scores["subset"].eq("real")
    ]
    reference = selected_video_means(
        calibration,
        target_k,
        {"G_mean_raw": "G_k", "L_raw": local_column},
        group_columns=KEY_COLUMNS,
    )
    if reference.empty:
        raise ValueError(
            f"no real calibration reference for effective_k={target_k}"
        )
    return reference


def calibration_references(
    window_scores: pd.DataFrame, target_k: int
) -> pd.DataFrame:
    """Build historical mean/lower-tail aggregates for each real video."""

    calibration = window_scores[
        window_scores["protocol_split"].eq("calibration")
        & window_scores["subset"].eq("real")
    ]
    rows: list[dict] = []
    for key, frame in calibration.groupby(KEY_COLUMNS, sort=False, observed=True):
        ordered = frame.sort_values("window_id")
        positions = effective_k_positions(len(ordered), target_k)
        selected = ordered.iloc[positions]
        if len(selected) != target_k:
            continue
        local = selected["L_k"]
        rows.append(
            {
                **dict(zip(KEY_COLUMNS, key)),
                "target_k": target_k,
                "G_mean_raw": float(selected["G_k"].mean()),
                "L_mean_raw": float(local.mean()),
                "L_bottom2_raw": bottom2_mean(local),
            }
        )
    reference = pd.DataFrame(rows)
    if reference.empty:
        raise ValueError(f"no calibration videos support target_k={target_k}")
    reference["L_hybrid_raw"] = 0.5 * (
        reference["L_mean_raw"] + reference["L_bottom2_raw"]
    )
    return reference


__all__ = [
    "KEY_COLUMNS",
    "WINDOW_KEYS",
    "bottom2_mean",
    "calibration_references",
    "target_k_reference",
]
