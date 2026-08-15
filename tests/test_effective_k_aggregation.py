from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_stalled.aggregation import effective_k_positions, selected_video_means
from alpha_stalled.calibration import (
    U0VideoReferences,
    calibrate_u0_video_branches,
)


class EffectiveKAggregationTests(unittest.TestCase):
    def test_locked_positions_cover_midpoint_and_endpoints(self) -> None:
        np.testing.assert_array_equal(effective_k_positions(3, 1), [1])
        np.testing.assert_array_equal(effective_k_positions(2, 1), [0])
        np.testing.assert_array_equal(effective_k_positions(3, 2), [0, 2])
        np.testing.assert_array_equal(effective_k_positions(5, 3), [0, 2, 4])
        np.testing.assert_array_equal(effective_k_positions(2, 3), [])
        with self.assertRaisesRegex(ValueError, "target_k must be positive"):
            effective_k_positions(3, 0)
        with self.assertRaisesRegex(ValueError, "window_count must be non-negative"):
            effective_k_positions(-1, 1)

    def test_selected_means_are_one_value_per_video(self) -> None:
        frame = pd.DataFrame(
            {
                "video_id": ["a"] * 3 + ["b"] * 2 + ["c"],
                "window_id": [2, 0, 1, 1, 0, 0],
                "G_k": [0.9, 0.1, 0.5, 0.8, 0.2, 1.0],
                "L_k": [0.7, 0.3, 0.5, 0.6, 0.4, 1.0],
            }
        )
        result = selected_video_means(
            frame, 2, {"G_raw": "G_k", "L_raw": "L_k"}
        ).set_index("video_id")
        self.assertEqual(list(result.index), ["a", "b"])
        np.testing.assert_allclose(result.loc["a"], [0.5, 0.5])
        np.testing.assert_allclose(result.loc["b"], [0.5, 0.5])

        filtered = selected_video_means(
            frame,
            1,
            {"score": "G_k"},
            selected_video_ids={"b", "c"},
        )
        self.assertEqual(filtered["video_id"].tolist(), ["b", "c"])
        np.testing.assert_allclose(filtered["score"], [0.2, 1.0])

    def test_composite_video_identity_is_explicit_and_preserved(self) -> None:
        frame = pd.DataFrame(
            {
                "dataset": ["d", "d", "d", "d"],
                "subset": ["real", "real", "fake", "fake"],
                "filename": ["a", "a", "a", "a"],
                "window_id": [1, 0, 1, 0],
                "score": [0.8, 0.2, 0.6, 0.4],
            }
        )
        result = selected_video_means(
            frame,
            2,
            {"mean": "score"},
            group_columns=("dataset", "subset", "filename"),
        )
        self.assertEqual(
            result.columns.tolist(), ["dataset", "subset", "filename", "mean"]
        )
        self.assertEqual(result["subset"].tolist(), ["real", "fake"])
        np.testing.assert_allclose(result["mean"], [0.5, 0.5])
        with self.assertRaisesRegex(ValueError, "group_columns must be non-empty"):
            selected_video_means(frame, 1, {"mean": "score"}, group_columns=())

    def test_video_branch_calibration_uses_effective_k_references(self) -> None:
        actual = calibrate_u0_video_branches(
            np.array([0.0, 2.0]),
            np.array([3.0, 1.0]),
            U0VideoReferences(
                np.array([0.0, 1.0, 2.0]),
                np.array([1.0, 2.0, 3.0]),
            ),
        )
        np.testing.assert_array_equal(actual["G"], [1.0 / 3.0, 1.0])
        np.testing.assert_array_equal(actual["L"], [1.0, 1.0 / 3.0])
        np.testing.assert_allclose(actual["S"], [0.6, 11.0 / 15.0], atol=1e-15)


if __name__ == "__main__":
    unittest.main()
