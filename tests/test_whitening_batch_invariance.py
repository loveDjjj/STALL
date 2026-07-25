from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stable_whitening import (
    StableGaussianParams,
    score_gaussian_aggregate_float64,
    score_gaussian_mean_candidates_float64,
    score_mean_gaussian_float64,
)


class WhiteningBatchInvarianceTests(unittest.TestCase):
    def test_outer_batch_size_cannot_change_float64_window_score(self) -> None:
        rng = np.random.RandomState(7)
        features = rng.normal(size=(17, 3, 5, 8)).astype(np.float32)
        params = StableGaussianParams(
            mean=rng.normal(size=8),
            whitening=rng.normal(size=(8, 7)),
            calibration_raw=np.sort(rng.normal(size=31)),
        )
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        references = None
        for batch_size in (1, 4, 8, 16):
            raw_parts = []
            percentile_parts = []
            for start in range(0, len(features), batch_size):
                raw, percentile = score_mean_gaussian_float64(
                    features[start : start + batch_size], params, device=device
                )
                raw_parts.append(raw)
                percentile_parts.append(percentile)
            result = (
                np.concatenate(raw_parts),
                np.concatenate(percentile_parts),
            )
            if references is None:
                references = result
            else:
                np.testing.assert_array_equal(result[0], references[0])
                np.testing.assert_array_equal(result[1], references[1])

    def test_min_aggregation_and_invalid_mask(self) -> None:
        features = np.array(
            [
                [[1.0, 0.0], [3.0, 0.0]],
                [[2.0, 0.0], [4.0, 0.0]],
            ],
            dtype=np.float32,
        )
        params = StableGaussianParams(
            mean=np.zeros(2),
            whitening=np.eye(2),
            calibration_raw=np.array([-20.0, -10.0, -5.0]),
        )
        raw, percentile = score_gaussian_aggregate_float64(
            features,
            params,
            aggregation="min",
            device="cpu",
            invalid_mask=np.array([[False, True], [False, False]]),
        )
        constant = 2.0 * np.log(2.0 * np.pi)
        expected = np.array(
            [
                -0.5 * (constant + 1.0),
                -0.5 * (constant + 16.0),
            ]
        )
        np.testing.assert_allclose(raw, expected, atol=0.0, rtol=0.0)
        np.testing.assert_array_equal(percentile, [1.0, 2.0 / 3.0])

    def test_candidate_sufficient_statistics_match_direct_mean(self) -> None:
        rng = np.random.RandomState(19)
        features = rng.normal(size=(5, 4, 3, 8)).astype(np.float32)
        candidates = [
            StableGaussianParams(
                mean=rng.normal(size=8),
                whitening=rng.normal(size=(8, rank)),
                calibration_raw=np.sort(rng.normal(size=13)),
            )
            for rank in (8, 6, 4)
        ]
        center = rng.normal(size=8)
        accelerated = score_gaussian_mean_candidates_float64(
            features, candidates, center, device="cpu"
        )
        direct = np.column_stack(
            [
                score_mean_gaussian_float64(
                    features, candidate, device="cpu"
                )[0]
                for candidate in candidates
            ]
        )
        np.testing.assert_allclose(accelerated, direct, atol=2e-13, rtol=2e-15)


if __name__ == "__main__":
    unittest.main()
