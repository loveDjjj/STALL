from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release/u0_external_genvidbench"


class ExternalGenVidBenchLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.calibration = json.loads(
            (RELEASE / "calibration_manifest.json").read_text(encoding="utf-8")
        )
        cls.evaluation = json.loads(
            (RELEASE / "evaluation_manifest.json").read_text(encoding="utf-8")
        )
        cls.frames = json.loads(
            (RELEASE / "frame_indices.json").read_text(encoding="utf-8")
        )
        cls.audit = json.loads(
            (RELEASE / "data_audit.json").read_text(encoding="utf-8")
        )
        cls.params = json.loads(
            (RELEASE / "local_params_metadata.json").read_text(encoding="utf-8")
        )

    def test_counts_and_real_only_calibration(self) -> None:
        self.assertEqual(self.calibration["video_count"], 199)
        self.assertEqual(self.calibration["subset_counts"], {"real": 199})
        self.assertEqual(self.evaluation["video_count"], 900)
        self.assertEqual(
            self.evaluation["source_counts"], {"ms": 300, "pika": 300, "vript": 300}
        )

    def test_ids_are_unique_and_disjoint(self) -> None:
        calibration_ids = {item["video_id"] for item in self.calibration["videos"]}
        evaluation_ids = {item["video_id"] for item in self.evaluation["videos"]}
        self.assertEqual(len(calibration_ids), 199)
        self.assertEqual(len(evaluation_ids), 900)
        self.assertFalse(calibration_ids & evaluation_ids)
        self.assertEqual(self.audit["calibration_evaluation_overlap_count"], 0)

    def test_every_locked_window_is_strict(self) -> None:
        all_ids = {
            item["video_id"]
            for payload in (self.calibration, self.evaluation)
            for item in payload["videos"]
        }
        self.assertEqual(set(self.frames["videos"]), all_ids)
        for windows in self.frames["videos"].values():
            self.assertTrue(windows)
            self.assertEqual(len({tuple(window) for window in windows}), len(windows))
            for window in windows:
                self.assertEqual(len(window), 16)
                self.assertEqual(len(set(window)), 16)
        self.assertEqual(
            len(self.frames["calibration_reference_windows"]), 199
        )

    def test_protocol_boundaries_are_explicit(self) -> None:
        self.assertTrue(self.audit["locked_before_u0_metrics"])
        self.assertEqual(
            self.audit["unindexed_calibration_files"],
            ["-uZZf5BE-JI-Scene-043.mp4"],
        )
        self.assertEqual(self.audit["t2vz"]["file_count"], 300)
        self.assertEqual(self.audit["t2vz"]["eligible_fake_count"], 0)
        self.assertEqual(self.audit["t2vz"]["native_fps_median"], 4.0)

    def test_locked_local_parameters_are_self_contained(self) -> None:
        path = ROOT / self.params["params"]
        self.assertEqual(path.parent, RELEASE / "params")
        self.assertTrue(path.is_file())
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            self.params["params_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
