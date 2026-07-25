from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release/u0"


class U0RobustnessLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.robustness = json.loads(
            (RELEASE / "robustness_subset_manifest.json").read_text()
        )
        cls.calibration = json.loads(
            (RELEASE / "calibration_manifest.json").read_text()
        )
        cls.injection = json.loads(
            (RELEASE / "injection_subset_manifest.json").read_text()
        )
        cls.robustness_plan = json.loads(
            (RELEASE / "robustness_perturbation_plan.json").read_text()
        )
        cls.injection_plan = json.loads(
            (RELEASE / "injection_plan.json").read_text()
        )

    def test_balanced_subset_and_no_calibration_overlap(self) -> None:
        self.assertEqual(self.robustness["video_count"], 1600)
        self.assertEqual(self.robustness["real_per_dataset"], 200)
        self.assertTrue(all(row["count"] == 50 for row in self.robustness["generator_counts"]))
        robust_ids = {item["video_id"] for item in self.robustness["videos"]}
        calibration_ids = {item["video_id"] for item in self.calibration["videos"]}
        self.assertFalse(robust_ids & calibration_ids)

    def test_scene_cut_maps_are_complete_and_never_self(self) -> None:
        evaluation = self.robustness_plan["evaluation_scene_cut_donor"]
        calibration = self.robustness_plan["calibration_scene_cut_donor"]
        self.assertEqual(len(evaluation), 1600)
        self.assertEqual(len(calibration), 600)
        self.assertTrue(all(video_id != donor for video_id, donor in evaluation.items()))
        self.assertTrue(all(video_id != donor for video_id, donor in calibration.items()))

    def test_injection_geometry_and_membership_are_frozen(self) -> None:
        injection_ids = {item["video_id"] for item in self.injection["videos"]}
        self.assertEqual(len(injection_ids), 300)
        self.assertEqual(set(self.injection_plan["video_ids"]), injection_ids)
        self.assertEqual(self.injection_plan["spatial_region"]["area_fraction"], 0.25)
        self.assertEqual(
            self.injection_plan["temporal_region"]["affected_zero_based_frames"],
            [6, 7, 8, 9],
        )
        self.assertEqual(len(self.injection_plan["conditions"]), 8)
        self.assertEqual(len(self.injection_plan["target_k3_window"]), 300)
        self.assertEqual(len(self.injection_plan["affected_native_frame_indices"]), 300)
        self.assertEqual(len(self.injection_plan["k1_overlap_count"]), 300)
        self.assertEqual(
            sum(value == 0 for value in self.injection_plan["k1_overlap_count"].values()),
            165,
        )


if __name__ == "__main__":
    unittest.main()
