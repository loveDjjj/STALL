from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import yaml
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release/u0"
for directory in (ROOT / "src", ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from alpha_stalled.calibration import cdf_with_positive_infinity
from alpha_stalled.u0_protocol import selected_calibration_means
from verify_u0_locked_release import sha256_file


class U0LockedReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = yaml.safe_load(
            (ROOT / "configs/alpha_stalled_u0_locked.yaml").read_text()
        )
        cls.calibration = json.loads(
            (RELEASE / "calibration_manifest.json").read_text()
        )
        cls.evaluation = json.loads(
            (RELEASE / "evaluation_manifest.json").read_text()
        )
        cls.indices = json.loads((RELEASE / "frame_indices.json").read_text())
        cls.hashes = json.loads(
            (RELEASE / "config_and_checkpoint_hashes.json").read_text()
        )
        cls.metadata = json.loads((RELEASE / "reproduction_metadata.json").read_text())
        cls.validation = json.loads((RELEASE / "validation.json").read_text())

    def test_locked_local_protocol_is_fully_uniform(self) -> None:
        local = self.config["local_branch"]
        self.assertEqual(local["region_size"], 1)
        self.assertEqual(local["patch_spatial_aggregation"], "mean")
        self.assertEqual(local["temporal_aggregation"], "mean")
        self.assertEqual(local["beta"], 0.1)
        self.assertEqual(self.config["video_aggregation"]["alpha"], 0.6)
        self.assertEqual(self.config["numerics"]["implementation"], "N2")
        self.assertEqual(self.config["numerics"]["whitening_matrix_dtype"], "float64")

    def test_manifest_counts_and_overlap(self) -> None:
        self.assertEqual(self.calibration["video_count"], 600)
        self.assertEqual(self.evaluation["video_count"], 21421)
        self.assertEqual(
            self.calibration["dataset_counts"],
            {"comgenvid": 200, "genvideo": 200, "videofeedback": 200},
        )
        self.assertEqual(
            self.evaluation["dataset_counts"],
            {"comgenvid": 4298, "genvideo": 13623, "videofeedback": 3500},
        )
        calibration_ids = {row["video_id"] for row in self.calibration["videos"]}
        evaluation_ids = {row["video_id"] for row in self.evaluation["videos"]}
        self.assertEqual(len(calibration_ids), 600)
        self.assertEqual(len(evaluation_ids), 21421)
        self.assertFalse(calibration_ids & evaluation_ids)
        self.assertEqual({row["subset"] for row in self.calibration["videos"]}, {"real"})

    def test_every_locked_window_has_unique_16_frame_indices(self) -> None:
        windows_by_video = self.indices["videos"]
        self.assertEqual(self.indices["video_count"], 22021)
        effective_k = {
            row["video_id"]: row["effective_k"]
            for row in self.calibration["videos"] + self.evaluation["videos"]
        }
        self.assertEqual(set(windows_by_video), set(effective_k))
        for video_id, windows in windows_by_video.items():
            self.assertEqual(len(windows), effective_k[video_id])
            self.assertEqual(len(windows), len({tuple(window) for window in windows}))
            for window in windows:
                self.assertEqual(len(window), 16)
                self.assertEqual(len(set(window)), 16)
                self.assertGreaterEqual(min(window), 0)
        calibration_reference = self.indices["calibration_reference_windows"]
        self.assertEqual(self.indices["calibration_reference_count"], 600)
        self.assertEqual(
            set(calibration_reference),
            {row["video_id"] for row in self.calibration["videos"]},
        )
        for window in calibration_reference.values():
            self.assertEqual(len(window), 16)
            self.assertEqual(len(set(window)), 16)

    def test_hash_registry_matches_locked_config(self) -> None:
        inputs = self.hashes["input_files"]
        self.assertEqual(
            inputs["dino_checkpoint"]["sha256"],
            self.config["backbone"]["checkpoint_sha256"],
        )
        self.assertEqual(
            inputs["global_params"]["sha256"],
            self.config["global_branch"]["params_sha256"],
        )
        for dataset, item in self.config["local_branch"]["params_by_dataset"].items():
            self.assertEqual(inputs[f"local_params_{dataset}"]["sha256"], item["sha256"])

    def test_local_parameters_are_self_contained_in_release(self) -> None:
        expected_root = RELEASE / "params"
        for dataset, item in self.config["local_branch"]["params_by_dataset"].items():
            path = ROOT / item["path"]
            self.assertEqual(path.parent, expected_root, dataset)
            self.assertTrue(path.is_file(), dataset)
            self.assertEqual(sha256_file(path), item["sha256"], dataset)

    def test_authoritative_metric_is_consistent_across_release_files(self) -> None:
        release = self.config["release"]
        self.assertEqual(release["stable_macro_auc"], self.metadata["macro_auc"])
        self.assertEqual(
            release["stable_macro_real_positive_ap"],
            self.metadata["macro_real_positive_ap"],
        )
        self.assertTrue(self.validation["passed"])
        self.assertEqual(self.validation["macro_auc"], self.metadata["macro_auc"])
        self.assertEqual(
            self.validation["macro_real_positive_ap"],
            self.metadata["macro_real_positive_ap"],
        )

    def test_positive_infinity_global_score_maps_to_one(self) -> None:
        result = cdf_with_positive_infinity(
            np.array([-2.0, 0.0, np.inf]), np.array([-3.0, -1.0, 1.0])
        )
        np.testing.assert_array_equal(result, [1.0 / 3.0, 2.0 / 3.0, 1.0])
        with self.assertRaises(ValueError):
            cdf_with_positive_infinity(
                np.array([np.nan]), np.array([-3.0, -1.0, 1.0])
            )

    def test_effective_k_calibration_selection_is_deterministic(self) -> None:
        frame = pd.DataFrame(
            {
                "video_id": ["a"] * 3 + ["b"] * 3,
                "window_id": [0, 1, 2] * 2,
                "G_k": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "L_k": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            }
        )
        k1 = selected_calibration_means(frame, 1).set_index("video_id")
        k2 = selected_calibration_means(frame, 2).set_index("video_id")
        k3 = selected_calibration_means(frame, 3).set_index("video_id")
        for result in (k1, k2, k3):
            self.assertAlmostEqual(float(result.loc["a", "G_raw"]), 0.2, places=15)
            self.assertAlmostEqual(float(result.loc["a", "L_raw"]), 0.5, places=15)


if __name__ == "__main__":
    unittest.main()
