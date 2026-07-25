"""Frozen synthetic local-anomaly operators for the U0 localization audit."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from u0_perturbations import insert_scene_cut


CONDITIONS = (
    "L0_original",
    "L1_local_freeze4",
    "L2_local_repeat1",
    "L3_local_flicker",
    "L4_local_affine_jitter",
    "L5_local_temporal_shift",
    "L6_full_frame_freeze4",
    "L7_scene_cut",
)


@dataclass(frozen=True)
class InjectionResult:
    frames: np.ndarray
    pixel_mask: np.ndarray
    patch_mask: np.ndarray
    temporal_mask: np.ndarray
    positive_anomaly: bool


def _center_bounds(height: int, width: int) -> tuple[slice, slice]:
    y0 = int(np.floor(height * 0.25))
    y1 = int(np.floor(height * 0.75))
    x0 = int(np.floor(width * 0.25))
    x1 = int(np.floor(width * 0.75))
    if y1 <= y0 or x1 <= x0:
        raise ValueError("frame is too small for the frozen center region")
    return slice(y0, y1), slice(x0, x1)


def _masks(length: int, height: int, width: int, full: bool = False) -> tuple[np.ndarray, np.ndarray]:
    pixel = np.zeros((length, height, width), dtype=bool)
    patch = np.zeros((length, 14, 14), dtype=bool)
    if full:
        pixel[:, :, :] = True
        patch[:, :, :] = True
    else:
        ys, xs = _center_bounds(height, width)
        pixel[:, ys, xs] = True
        patch[:, 3:10, 3:10] = True
    return pixel, patch


def inject(frames: np.ndarray, condition: str, donor: np.ndarray | None = None) -> InjectionResult:
    values = np.asarray(frames)
    if values.ndim != 4 or values.shape[-1] != 3 or len(values) != 16:
        raise ValueError(f"expected a 16-frame BGR window, got {values.shape}")
    if condition not in CONDITIONS:
        raise ValueError(f"unknown injection condition: {condition}")
    output = values.copy()
    height, width = output.shape[1:3]
    temporal = np.zeros(16, dtype=bool)
    pixel = np.zeros((16, height, width), dtype=bool)
    patch = np.zeros((16, 14, 14), dtype=bool)
    ys, xs = _center_bounds(height, width)

    if condition == "L0_original":
        return InjectionResult(output, pixel, patch, temporal, False)
    if condition == "L7_scene_cut":
        if donor is None:
            raise ValueError("scene cut requires the prelocked donor window")
        output = insert_scene_cut(output, donor)
        temporal[8] = True
        pixel[8] = True
        patch[8] = True
        return InjectionResult(output, pixel, patch, temporal, False)

    affected = [6, 7, 8, 9]
    if condition == "L1_local_freeze4":
        output[affected, ys, xs] = output[5, ys, xs]
        temporal[affected] = True
    elif condition == "L2_local_repeat1":
        output[8, ys, xs] = output[7, ys, xs]
        temporal[8] = True
    elif condition == "L3_local_flicker":
        for offset, frame_index in enumerate(affected):
            factor = 0.6 if offset % 2 == 0 else 1.4
            region = output[frame_index, ys, xs].astype(np.float32) * factor
            output[frame_index, ys, xs] = np.clip(region, 0, 255).astype(np.uint8)
        temporal[affected] = True
    elif condition == "L4_local_affine_jitter":
        region_width = xs.stop - xs.start
        shift = max(1, int(np.rint(region_width * 0.04)))
        for offset, frame_index in enumerate(affected):
            region = output[frame_index, ys, xs]
            dx = shift if offset % 2 == 0 else -shift
            matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, 0.0]], dtype=np.float32)
            output[frame_index, ys, xs] = cv2.warpAffine(
                region,
                matrix,
                (region.shape[1], region.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
        temporal[affected] = True
    elif condition == "L5_local_temporal_shift":
        source = output[[8, 9, 10, 11], ys, xs].copy()
        output[affected, ys, xs] = source
        temporal[affected] = True
    elif condition == "L6_full_frame_freeze4":
        output[affected] = output[5]
        temporal[affected] = True
        pixel_mask, patch_mask = _masks(16, height, width, full=True)
        pixel[affected] = pixel_mask[affected]
        patch[affected] = patch_mask[affected]
        return InjectionResult(output, pixel, patch, temporal, True)

    local_pixel, local_patch = _masks(16, height, width, full=False)
    pixel[temporal] = local_pixel[temporal]
    patch[temporal] = local_patch[temporal]
    return InjectionResult(output, pixel, patch, temporal, True)
