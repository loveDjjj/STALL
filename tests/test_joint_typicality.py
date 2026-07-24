from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from analyze_joint_typicality import (
    add_conflict_labels,
    clipped_percentile,
    crossfit_conditioned_margins,
    crossfit_margins,
    gaussianize,
    lower_tail_quadratic,
    upper_tail_realness,
)


class JointTypicalityTests(unittest.TestCase):
    def test_crossfit_shapes_and_finite_percentiles(self) -> None:
        calibration = np.column_stack([np.arange(20), np.arange(20)[::-1]]).astype(float)
        evaluation = np.array([[5.0, 5.0], [15.0, 15.0]])
        result = crossfit_margins(calibration, evaluation, seed=42, folds=5)
        self.assertEqual(result.calibration_percentiles.shape, (20, 2))
        self.assertEqual(result.evaluation_fold_percentiles.shape, (5, 2, 2))
        self.assertTrue(np.isfinite(result.calibration_percentiles).all())

    def test_one_sided_distance_does_not_penalize_high_scores(self) -> None:
        z = gaussianize(np.array([[0.9, 0.9], [0.1, 0.1]]), calibration_size=20)
        distance = lower_tail_quadratic(z, np.eye(2))
        self.assertEqual(distance[0], 0.0)
        self.assertGreater(distance[1], 0.0)

    def test_upper_tail_realness_is_high_for_small_distance(self) -> None:
        scores = upper_tail_realness(np.array([0.0, 3.0]), np.array([0.0, 1.0, 2.0, 3.0]))
        self.assertGreater(scores[0], scores[1])

    def test_percentile_clip_uses_actual_reference_size(self) -> None:
        scores = clipped_percentile(np.array([-1.0, 10.0]), np.arange(4, dtype=float))
        self.assertTrue(np.allclose(scores, [0.2, 0.8]))

    def test_conditioned_crossfit_supports_mixed_effective_k(self) -> None:
        calibration_rows = []
        reference_rows = []
        for index in range(200):
            base = {
                "dataset": "d",
                "subset": "real",
                "source_model": "m",
                "filename": f"{index}.mp4",
            }
            calibration_rows.append(
                {
                    **base,
                    "effective_k": 1 if index < 25 else 3,
                    "G_mean_raw": index / 200,
                    "L_hybrid_raw": (200 - index) / 200,
                }
            )
            for target_k in (1, 3):
                if target_k == 1 or index >= 25:
                    reference_rows.append(
                        {
                            **base,
                            "target_k": target_k,
                            "G_mean_raw": index / 200,
                            "L_hybrid_raw": (200 - index) / 200,
                        }
                    )
        evaluation = pd.DataFrame(
            [
                {"dataset": "d", "subset": "real", "source_model": "m", "filename": "e1.mp4", "effective_k": 1, "G_mean_raw": 0.5, "L_hybrid_raw": 0.5},
                {"dataset": "d", "subset": "annotated", "source_model": "f", "filename": "e3.mp4", "effective_k": 3, "G_mean_raw": 0.2, "L_hybrid_raw": 0.2},
            ]
        )
        result = crossfit_conditioned_margins(
            pd.DataFrame(calibration_rows),
            evaluation,
            pd.DataFrame(reference_rows),
            "L_hybrid_raw",
            seed=42,
        )
        self.assertEqual(result.calibration_percentiles.shape, (200, 2))
        self.assertEqual(result.evaluation_fold_percentiles.shape, (5, 2, 2))
        self.assertTrue(np.isfinite(result.evaluation_fold_percentiles).all())

    def test_conflict_q95_is_computed_from_ensembled_oof_margins(self) -> None:
        calibration_rows = []
        for seed, offset in ((1, 0.0), (2, 0.2)):
            for index in range(20):
                calibration_rows.append(
                    {
                        "dataset": "d",
                        "subset": "real",
                        "source_model": "real",
                        "filename": f"{index}.mp4",
                        "seed": seed,
                        "G_crossfit": index / 20 + offset,
                        "L_crossfit": 0.5,
                        "J0": index / 20,
                        "J0X": index / 20,
                        "J1": index / 20,
                        "J2": index / 20,
                        "J2G": index / 20,
                        "J3": index / 20,
                    }
                )
        evaluation = pd.DataFrame(
            [
                {
                    "dataset": "d",
                    "subset": "annotated",
                    "source_model": "fake",
                    "filename": "fake.mp4",
                    "G_crossfit": 1.0,
                    "L_crossfit": 0.0,
                    "conflict_threshold": -1.0,
                    "J0": 0.0,
                    "J0X": 0.0,
                    "J1": 0.0,
                    "J2": 0.0,
                    "J2G": 0.0,
                    "J3": 0.0,
                }
            ]
        )
        labeled, _ = add_conflict_labels(
            evaluation, pd.DataFrame(calibration_rows)
        )
        ensembled_difference = np.abs(np.arange(20) / 20 + 0.1 - 0.5)
        self.assertAlmostEqual(
            labeled.conflict_threshold.iloc[0],
            np.quantile(ensembled_difference, 0.95),
        )
        self.assertTrue(labeled.conflict.iloc[0])


if __name__ == "__main__":
    unittest.main()
