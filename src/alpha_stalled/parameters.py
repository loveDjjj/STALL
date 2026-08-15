"""Loading of frozen Gaussian parameters for locked U0 raw scoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from alpha_stalled.whitening import StableGaussianParams, stable_sorted

from .release_io import REPOSITORY_ROOT


def global_references(
    config: dict,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the frozen STALL spatial-max and temporal-min CDF references."""
    root = Path(repository_root)
    with np.load(root / config["global_branch"]["params"], allow_pickle=True) as data:
        return (
            stable_sorted(np.max(data["calib_ll_spat"].astype(np.float64), axis=1)),
            stable_sorted(np.min(data["calib_ll_temp"].astype(np.float64), axis=1)),
        )


def load_raw_params(
    config: dict,
    dataset: str,
    local_params_override: str | Path | None = None,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, StableGaussianParams]:
    """Load the four Gaussian models used before release CDF calibration."""
    root = Path(repository_root)
    global_path = root / config["global_branch"]["params"]
    with np.load(global_path, allow_pickle=True) as global_data:
        global_spatial_reference = stable_sorted(
            np.max(global_data["calib_ll_spat"].astype(np.float64), axis=1)
        )
        global_temporal_reference = stable_sorted(
            np.min(global_data["calib_ll_temp"].astype(np.float64), axis=1)
        )
        global_spatial = StableGaussianParams(
            mean=global_data["mu_spat"].astype(np.float64),
            whitening=global_data["W_spat"].astype(np.float64),
            calibration_raw=global_spatial_reference,
        )
        global_t1 = StableGaussianParams(
            mean=global_data["mu_temp"].astype(np.float64),
            whitening=global_data["W_temp"].astype(np.float64),
            calibration_raw=global_temporal_reference,
        )

    local_path = (
        Path(local_params_override)
        if local_params_override is not None
        else root / config["local_branch"]["params_by_dataset"][dataset]["path"]
    )
    with np.load(local_path, allow_pickle=True) as local_data:
        # Local CDFs are rebuilt from release K1 videos after raw traversal.
        placeholder = np.array([0.0], dtype=np.float64)
        patch_spatial = StableGaussianParams(
            mean=local_data["mu_patch_spat"].astype(np.float64),
            whitening=local_data["W_patch_spat"].astype(np.float64),
            calibration_raw=placeholder,
        )
        patch_d2 = StableGaussianParams(
            mean=local_data["mu_patch_temp"].astype(np.float64),
            whitening=local_data["W_patch_temp"].astype(np.float64),
            calibration_raw=placeholder,
        )

    return {
        "global_spatial": global_spatial,
        "global_t1": global_t1,
        "patch_spatial": patch_spatial,
        "patch_d2": patch_d2,
    }


__all__ = ["global_references", "load_raw_params"]
