from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from alpha_stalled import global_branch
from alpha_stalled.calibration import (
    GLOBAL_SPATIAL_WEIGHT,
    fuse_global_components,
)
from alpha_stalled.whitening import (
    StableGaussianParams,
    l2_normalized_first_order,
    score_gaussian_aggregate_float64,
)
import score_duration_aware_original_k1
import score_u0_locked_k1_cache
import score_u0_locked_windows
import score_u0_robustness


class GlobalBranchContractTests(unittest.TestCase):
    def params(self, seed: int) -> StableGaussianParams:
        rng = np.random.RandomState(seed)
        return StableGaussianParams(
            mean=rng.normal(size=4),
            whitening=rng.normal(size=(4, 3)),
            calibration_raw=np.sort(rng.normal(size=13)),
        )

    def test_locked_constants_and_public_surface_are_narrow(self) -> None:
        self.assertEqual(global_branch.GLOBAL_SPATIAL_AGGREGATION, "max")
        self.assertEqual(global_branch.GLOBAL_TEMPORAL_AGGREGATION, "min")
        self.assertEqual(global_branch.GLOBAL_TEMPORAL_ORDER, 1)
        self.assertEqual(global_branch.GLOBAL_SPATIAL_WEIGHT, 0.5)
        self.assertEqual(GLOBAL_SPATIAL_WEIGHT, 0.5)
        excluded = {"d3", "volatility", "third_order", "fourth_order"}
        self.assertTrue(excluded.isdisjoint(global_branch.__all__))

    def test_t1_features_delegate_exact_primitive_and_keep_zero_mask(self) -> None:
        features = torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0], [2.0, 0.0]]],
            dtype=torch.float32,
        )
        actual, actual_zero = global_branch.global_t1_features(features)
        expected, expected_zero = l2_normalized_first_order(features)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_zero, expected_zero, rtol=0, atol=0)
        self.assertEqual(actual_zero.tolist(), [[True, False]])

    def test_raw_scorer_matches_direct_float64_max_min_likelihood(self) -> None:
        rng = np.random.RandomState(41)
        features = rng.normal(size=(3, 5, 4)).astype(np.float32)
        features[0, 1] = features[0, 0]
        spatial_params = self.params(43)
        temporal_params = self.params(47)

        actual = global_branch.score_global_raw(
            features,
            spatial_params,
            temporal_params,
            device="cpu",
        )
        expected_spatial, _ = score_gaussian_aggregate_float64(
            features,
            spatial_params,
            aggregation="max",
            device="cpu",
            compute_percentile=False,
        )
        temporal_features, zero = l2_normalized_first_order(torch.from_numpy(features))
        expected_temporal, _ = score_gaussian_aggregate_float64(
            temporal_features,
            temporal_params,
            aggregation="min",
            device="cpu",
            invalid_mask=zero,
            compute_percentile=False,
        )
        np.testing.assert_array_equal(actual.spatial, expected_spatial)
        np.testing.assert_array_equal(actual.temporal_t1, expected_temporal)

    def test_all_zero_temporal_differences_produce_positive_infinity(self) -> None:
        features = np.ones((1, 4, 4), dtype=np.float32)
        actual = global_branch.score_global_raw(
            features,
            self.params(53),
            self.params(59),
            device="cpu",
        )
        self.assertTrue(np.isposinf(actual.temporal_t1[0]))

    def test_global_fusion_is_single_shared_implementation(self) -> None:
        self.assertIs(fuse_global_components, global_branch.fuse_global_components)
        np.testing.assert_allclose(
            fuse_global_components(np.array([0.2, 0.8]), np.array([0.6, 0.4])),
            [0.4, 0.6],
        )
        with self.assertRaisesRegex(ValueError, "weight must be"):
            fuse_global_components(np.array([0.2]), np.array([0.4]), spatial_weight=-0.1)

    def test_formal_scorers_import_canonical_global_raw_function(self) -> None:
        for module in (
            score_u0_locked_windows,
            score_u0_locked_k1_cache,
            score_duration_aware_original_k1,
            score_u0_robustness,
        ):
            self.assertIs(module.score_global_raw, global_branch.score_global_raw)


if __name__ == "__main__":
    unittest.main()
