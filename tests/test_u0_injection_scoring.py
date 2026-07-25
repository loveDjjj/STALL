from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from score_u0_injections import d2_labels, localization_metrics, patch_d2_anomaly
from analyze_u0_injections import reproduction_errors
from stable_whitening import StableGaussianParams
from u0_injections import inject


class U0InjectionScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.RandomState(31)
        self.frames = rng.randint(0, 256, size=(16, 40, 60, 3), dtype=np.uint8)

    def test_d2_labels_cover_three_frame_receptive_field(self) -> None:
        result = inject(self.frames, "L1_local_freeze4")
        labels = d2_labels(result.patch_mask)
        self.assertEqual(labels.shape, (14, 196))
        self.assertEqual(int(labels.sum()), 6 * 49)
        self.assertEqual(np.flatnonzero(labels.any(axis=1)).tolist(), [4, 5, 6, 7, 8, 9])

    def test_perfect_heatmap_has_perfect_localization_metrics(self) -> None:
        result = inject(self.frames, "L1_local_freeze4")
        labels = d2_labels(result.patch_mask)
        anomaly = labels.astype(np.float64)
        metrics = localization_metrics(anomaly, result)
        self.assertAlmostEqual(metrics["heatmap_auprc"], 1.0)
        self.assertAlmostEqual(metrics["topk_patch_hit_rate"], 1.0)
        self.assertEqual(metrics["temporal_hit"], 1.0)

    def test_patch_anomaly_shape_is_time_by_patch(self) -> None:
        rng = np.random.RandomState(37)
        patch = rng.normal(size=(16, 196, 4)).astype(np.float32)
        parameter = StableGaussianParams(
            mean=np.zeros(4),
            whitening=np.eye(4),
            calibration_raw=np.array([0.0]),
        )
        anomaly = patch_d2_anomaly(patch, {"patch_d2": parameter}, "cpu")
        self.assertEqual(anomaly.shape, (14, 196))
        self.assertTrue(np.isfinite(anomaly).all())

    def test_reproduction_errors_handles_same_named_release_score(self) -> None:
        raw = pd.DataFrame(
            {
                "video_id": ["v", "v"],
                "sampling": ["k1", "k3"],
                "condition": ["L0_original", "L0_original"],
                "window_id": [0, 0],
                "global_spatial_raw": [1.0, 1.0],
                "global_t1_raw": [2.0, 2.0],
                "patch_spatial_raw": [3.0, 3.0],
                "patch_d2_raw": [4.0, 4.0],
            }
        )
        per_video = pd.DataFrame(
            {
                "video_id": ["v", "v"],
                "sampling": ["k1", "k3"],
                "condition": ["L0_original", "L0_original"],
                "S": [0.25, 0.75],
            }
        )
        locked_windows = raw[raw["sampling"].eq("k3")].copy()
        core = pd.DataFrame(
            {
                "video_id": ["v"],
                "A9": [0.25],
                "global_spatial_raw": [1.0],
                "global_t1_raw": [2.0],
                "patch_spatial_raw": [3.0],
                "patch_d2_raw": [4.0],
            }
        )
        release = pd.DataFrame({"video_id": ["v"], "S": [0.75]})

        result = reproduction_errors(raw, per_video, locked_windows, core, release)

        self.assertEqual(float(result["max_abs_error"].max()), 0.0)


if __name__ == "__main__":
    unittest.main()
