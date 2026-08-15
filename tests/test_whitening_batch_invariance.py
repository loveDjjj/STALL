from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stable_whitening as legacy_whitening
from alpha_stalled import whitening
from alpha_stalled.whitening import (
    StableGaussianParams,
    l2_normalized_patch_first_order,
    l2_normalized_second_order,
    score_gaussian_aggregate_float64,
    score_gaussian_mean_candidates_float64,
    score_mean_gaussian_float64,
)


class WhiteningBatchInvarianceTests(unittest.TestCase):
    def test_legacy_module_reexports_shared_numerical_core(self) -> None:
        self.assertEqual(legacy_whitening.__all__, whitening.__all__)
        for name in whitening.__all__:
            self.assertIs(getattr(legacy_whitening, name), getattr(whitening, name))

    def test_gaussian_params_npz_loader_normalizes_dtype_and_cdf_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "params.npz"
            np.savez(
                path,
                mu_patch_temp=np.array([1.0, 2.0], dtype=np.float32),
                W_patch_temp=np.eye(2, dtype=np.float32),
                calib_patch_temp_scores=np.array([3.0, 1.0, 2.0], dtype=np.float32),
            )
            params = StableGaussianParams.from_npz(str(path))

        self.assertEqual(params.mean.dtype, np.float64)
        self.assertEqual(params.whitening.dtype, np.float64)
        self.assertEqual(params.calibration_raw.dtype, np.float64)
        np.testing.assert_array_equal(params.calibration_raw, [1.0, 2.0, 3.0])

    def test_patch_d1_and_d2_match_explicit_same_grid_formulas(self) -> None:
        patch = torch.arange(2 * 5 * 3 * 4, dtype=torch.float32).reshape(2, 5, 3, 4)
        d1 = patch[:, 1:] - patch[:, :-1]
        d2 = patch[:, 2:] - 2.0 * patch[:, 1:-1] + patch[:, :-2]
        expected_d1 = torch.nn.functional.normalize(d1, p=2, dim=-1, eps=1e-12)
        expected_d2 = torch.nn.functional.normalize(d2, p=2, dim=-1, eps=1e-12)
        torch.testing.assert_close(
            l2_normalized_patch_first_order(patch), expected_d1, rtol=0, atol=0
        )
        torch.testing.assert_close(
            l2_normalized_second_order(patch), expected_d2, rtol=0, atol=0
        )

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
