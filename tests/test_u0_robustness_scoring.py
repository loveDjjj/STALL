from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from score_u0_robustness import (
    donor_window_id,
    load_protocol,
    score_condition,
    temporal_embeddings,
)
from stable_whitening import StableGaussianParams


class U0RobustnessScoringTests(unittest.TestCase):
    def test_normalized_donor_window_matching(self) -> None:
        self.assertEqual(donor_window_id(0, 1, 3), 1)
        self.assertEqual([donor_window_id(i, 3, 5) for i in range(3)], [0, 2, 4])
        self.assertEqual([donor_window_id(i, 3, 2) for i in range(3)], [0, 0, 1])

    def test_locked_protocol_has_k1_only_for_calibration(self) -> None:
        calibration, _, windows, _, donors = load_protocol(
            "calibration",
            "comgenvid",
            ROOT / "release/u0",
            ROOT / "release/u0/robustness_subset_manifest.json",
            ROOT / "release/u0/robustness_perturbation_plan.json",
        )
        self.assertEqual(len(calibration), 200)
        self.assertEqual(len(donors), 200)
        self.assertTrue(all(set(groups) == {"k1", "k3"} for groups in windows.values()))
        evaluation, _, windows, _, donors = load_protocol(
            "evaluation",
            "comgenvid",
            ROOT / "release/u0",
            ROOT / "release/u0/robustness_subset_manifest.json",
            ROOT / "release/u0/robustness_perturbation_plan.json",
        )
        self.assertEqual(len(evaluation), 300)
        self.assertEqual(len(donors), 300)
        self.assertTrue(all(set(groups) == {"k3"} for groups in windows.values()))

    def test_temporal_embedding_operators_and_raw_scoring(self) -> None:
        rng = np.random.RandomState(11)
        source = {
            "global": rng.normal(size=(16, 4)).astype(np.float32),
            "patch": rng.normal(size=(16, 5, 4)).astype(np.float32),
        }
        dropped = temporal_embeddings(source, "R5_drop25", "key")
        repeated = temporal_embeddings(source, "R7_repeat25", "key")
        self.assertEqual(dropped["global"].shape, (12, 4))
        self.assertEqual(repeated["patch"].shape, (16, 5, 4))
        parameter = StableGaussianParams(
            mean=np.zeros(4),
            whitening=np.eye(4),
            calibration_raw=np.array([-10.0, 0.0]),
        )
        rows = score_condition([dropped, dropped], {name: parameter for name in (
            "global_spatial", "global_t1", "patch_spatial", "patch_d2"
        )}, "cpu")
        self.assertEqual(len(rows), 2)
        self.assertTrue(np.isfinite(list(rows[0].values())).all())


if __name__ == "__main__":
    unittest.main()
