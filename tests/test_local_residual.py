from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from tools.analyze_local_residual import calibrate_r1, merge_window_scores
from tools.score_local_residual_windows import _midpoint_median, common_motion_diagnostics


class LocalResidualTests(unittest.TestCase):
    def test_midpoint_median_matches_numpy_for_even_patch_count(self) -> None:
        values = torch.tensor([[[[0.0], [2.0], [4.0], [10.0]]]])
        actual = _midpoint_median(values, dim=2)
        np.testing.assert_allclose(actual.numpy(), [[[3.0]]])

    def test_common_acceleration_has_unit_mean_energy_ratio(self) -> None:
        patch = torch.zeros((1, 4, 196, 2), dtype=torch.float32)
        patch[:, 2:] = torch.tensor([1.0, -2.0])
        diagnostics = common_motion_diagnostics(patch)
        np.testing.assert_allclose(diagnostics["rho_mean"], [1.0], atol=1e-6)

    def test_opposing_acceleration_has_zero_mean_energy_ratio(self) -> None:
        patch = torch.zeros((1, 3, 196, 2), dtype=torch.float32)
        patch[:, 2, :98, 0] = 1.0
        patch[:, 2, 98:, 0] = -1.0
        diagnostics = common_motion_diagnostics(patch)
        np.testing.assert_allclose(diagnostics["rho_mean"], [0.0], atol=1e-7)

    def test_window_merge_preserves_frozen_keys_and_builds_r1_local(self) -> None:
        common = {
            "dataset": "demo",
            "protocol_split": "evaluation",
            "subset": "real",
            "source_model": "real",
            "filename": "a.mp4",
            "window_id": 0,
            "frame_indices": "[0,1]",
        }
        baseline = pd.DataFrame(
            [{**common, "patch_d2": 0.2, "patch_spatial": 0.4, "L_k": 0.22}]
        )
        residual = pd.DataFrame(
            [
                {
                    **common,
                    "residual_raw_likelihood": -1.0,
                    "residual_d2": 0.8,
                    "rho_mean": 0.1,
                    "rho_median": 0.1,
                    "mean_motion_magnitude": 1.0,
                    "mean_common_magnitude": 0.2,
                    "median_common_magnitude": 0.2,
                }
            ]
        )
        merged = merge_window_scores(baseline, residual)
        self.assertAlmostEqual(float(merged.loc[0, "R1_L_k"]), 0.76)

    def test_r1_calibration_reuses_frozen_global_percentile(self) -> None:
        rows = []
        for split, subset, source, name, global_score, local_score in (
            ("calibration", "real", "real", "cal.mp4", 0.4, 0.3),
            ("evaluation", "real", "real", "real.mp4", 0.5, 0.4),
            ("evaluation", "annotated", "fake", "fake.mp4", 0.2, 0.1),
        ):
            rows.append(
                {
                    "dataset": "demo",
                    "protocol_split": split,
                    "subset": subset,
                    "source_model": source,
                    "filename": name,
                    "effective_k": 1,
                    "G_mean_raw": global_score,
                    "R1_L_mean_raw": local_score,
                }
            )
        per_video = pd.DataFrame(rows)
        windows = pd.DataFrame(
            [
                {
                    "dataset": row["dataset"],
                    "protocol_split": row["protocol_split"],
                    "subset": row["subset"],
                    "source_model": row["source_model"],
                    "filename": row["filename"],
                    "window_id": 0,
                    "G_k": row["G_mean_raw"],
                    "R1_L_k": row["R1_L_mean_raw"],
                }
                for row in rows
            ]
        )
        frozen = pd.DataFrame(
            [
                {
                    "dataset": "demo",
                    "protocol_split": "evaluation",
                    "subset": subset,
                    "source_model": source,
                    "filename": name,
                    "MW2": score,
                    "G_mean": global_score,
                    "L_mean": local_score,
                }
                for subset, source, name, score, global_score, local_score in (
                    ("real", "real", "real.mp4", 1.0, 1.0, 1.0),
                    ("annotated", "fake", "fake.mp4", 0.0, 0.0, 0.0),
                )
            ]
        )
        calibrated = calibrate_r1(per_video, windows, frozen)
        self.assertEqual(calibrated.sort_values("subset")["R1_G"].tolist(), [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
