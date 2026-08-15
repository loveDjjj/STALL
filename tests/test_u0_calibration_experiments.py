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

from alpha_stalled import u0_analysis, u0_calibration_experiments, u0_scoring
import analyze_independent_real_complement
import analyze_u0_calibration_sensitivity
import analyze_u0_cross_dataset_calibration
import analyze_u0_injections
import analyze_u0_oas_candidate
import analyze_u0_robustness
import build_u0_calibration_reserve
import fit_u0_calibration_sensitivity
import score_u0_external_genvidbench_k1
import score_u0_locked_k1_cache


def calibration_fixture(name: str) -> pd.DataFrame:
    rows = []
    for video_index, video_id in enumerate(("c0", "c1")):
        rows.append(
            {
                "video_id": video_id,
                "sampling": "k1",
                "window_id": 0,
                "global_spatial_raw": 0.1 + video_index,
                "global_t1_raw": 0.2 + video_index,
                f"patch_spatial__{name}": 0.3 + video_index,
                f"patch_d2__{name}": 0.4 + video_index,
            }
        )
        for window_id in range(3):
            rows.append(
                {
                    "video_id": video_id,
                    "sampling": "k3",
                    "window_id": window_id,
                    "global_spatial_raw": 0.1 + video_index + window_id / 10,
                    "global_t1_raw": 0.2 + video_index + window_id / 10,
                    f"patch_spatial__{name}": 0.3 + video_index + window_id / 10,
                    f"patch_d2__{name}": 0.4 + video_index + window_id / 10,
                }
            )
    return pd.DataFrame(rows)


def evaluation_fixture(name: str) -> pd.DataFrame:
    rows = []
    for video_id, effective_k, offset in (("e1", 1, 0.35), ("e3", 3, 0.75)):
        for window_id in range(effective_k):
            rows.append(
                {
                    "video_id": video_id,
                    "sampling": "k3",
                    "window_id": window_id,
                    "global_spatial_raw": offset + window_id / 20,
                    "global_t1_raw": offset + 0.1 + window_id / 20,
                    f"patch_spatial__{name}": offset + 0.2 + window_id / 20,
                    f"patch_d2__{name}": offset + 0.3 + window_id / 20,
                }
            )
    return pd.DataFrame(rows)


class U0CalibrationExperimentTests(unittest.TestCase):
    def test_candidate_and_cross_paths_share_exact_scores(self) -> None:
        name = "fixture"
        calibration = calibration_fixture(name)
        evaluation = evaluation_fixture(name)
        selected = {"c0", "c1"}
        global_spatial = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
        global_t1 = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)

        candidate = u0_calibration_experiments.calibrate_candidate(
            evaluation,
            calibration,
            selected,
            name,
            global_spatial,
            global_t1,
        ).sort_values("video_id")
        cross = u0_calibration_experiments.calibrate_cross(
            evaluation,
            calibration,
            selected,
            name,
            global_spatial,
            global_t1,
        ).sort_values("video_id")
        self.assertEqual(candidate["effective_k"].tolist(), [1, 3])
        np.testing.assert_array_equal(
            candidate[["G", "L", "S"]].to_numpy(),
            cross[["G", "L", "S"]].to_numpy(),
        )
        np.testing.assert_allclose(
            candidate["S"], 0.6 * candidate["G"] + 0.4 * candidate["L"]
        )

    def test_calibration_cli_exports_are_shared_objects(self) -> None:
        shared = u0_calibration_experiments
        self.assertIs(analyze_u0_calibration_sensitivity.candidate, shared.candidate)
        self.assertIs(
            analyze_u0_calibration_sensitivity.calibrate_candidate,
            shared.calibrate_candidate,
        )
        self.assertIs(
            analyze_independent_real_complement.calibrate_candidate,
            shared.calibrate_candidate,
        )
        self.assertIs(
            analyze_u0_cross_dataset_calibration.calibrate_cross,
            shared.calibrate_cross,
        )
        self.assertIs(analyze_u0_oas_candidate.calibrate_cross, shared.calibrate_cross)
        self.assertIs(build_u0_calibration_reserve.DATASETS, shared.CALIBRATION_RESERVE_SPECS)
        self.assertIs(fit_u0_calibration_sensitivity.DATASETS, shared.CALIBRATION_RESERVE_SPECS)

    def test_effective_k_and_raw_scorer_exports_are_shared_objects(self) -> None:
        self.assertIs(
            analyze_u0_robustness.selected_k_reference,
            u0_analysis.effective_k_reference,
        )
        self.assertIs(
            analyze_u0_injections.selected_k_reference,
            u0_analysis.effective_k_reference,
        )
        self.assertIs(score_u0_locked_k1_cache.score_raw_batch, u0_scoring.score_raw_batch)
        self.assertIs(
            score_u0_external_genvidbench_k1.score_raw_batch,
            u0_scoring.score_raw_batch,
        )


if __name__ == "__main__":
    unittest.main()
