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

from alpha_stalled import local_branch
from alpha_stalled.calibration import (
    LOCAL_SPATIAL_WEIGHT,
    fuse_local_components,
)
from alpha_stalled.whitening import (
    StableGaussianParams,
    l2_normalized_patch_first_order,
    l2_normalized_second_order,
    score_gaussian_aggregate_float64,
)
from patch_math import empirical_percentile
import score_duration_aware_original_k1
import score_u0_locked_k1_cache
import score_u0_locked_windows


class LocalBranchContractTests(unittest.TestCase):
    def params(self, seed: int) -> StableGaussianParams:
        rng = np.random.RandomState(seed)
        return StableGaussianParams(
            mean=rng.normal(size=4),
            whitening=rng.normal(size=(4, 3)),
            calibration_raw=np.sort(rng.normal(size=11)),
        )

    def test_locked_constants_and_public_surface_are_narrow(self) -> None:
        self.assertEqual(local_branch.LOCAL_AGGREGATION, "mean")
        self.assertEqual(local_branch.LOCAL_TEMPORAL_ORDER, 2)
        self.assertEqual(local_branch.LOCAL_SPATIAL_WEIGHT, 0.1)
        self.assertEqual(LOCAL_SPATIAL_WEIGHT, 0.1)
        excluded = {
            "motion_hard",
            "motion_soft",
            "pool_patch_regions",
            "second_order_residual",
            "same_grid_third_order",
            "same_grid_fourth_order",
        }
        self.assertTrue(excluded.isdisjoint(local_branch.__all__))

    def test_local_d1_and_d2_delegate_exact_numerical_primitives(self) -> None:
        patch = torch.arange(2 * 5 * 3 * 4, dtype=torch.float32).reshape(2, 5, 3, 4)
        torch.testing.assert_close(
            local_branch.local_d1_features(patch),
            l2_normalized_patch_first_order(patch),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            local_branch.local_d2_features(patch),
            l2_normalized_second_order(patch),
            rtol=0,
            atol=0,
        )

    def test_raw_scorer_matches_direct_float64_mean_likelihood(self) -> None:
        rng = np.random.RandomState(23)
        patch = rng.normal(size=(3, 5, 2, 4)).astype(np.float32)
        spatial_params = self.params(29)
        temporal_params = self.params(31)

        actual = local_branch.score_local_raw(
            patch,
            spatial_params,
            temporal_params,
            device="cpu",
        )
        expected_spatial, _ = score_gaussian_aggregate_float64(
            patch,
            spatial_params,
            aggregation="mean",
            device="cpu",
            compute_percentile=False,
        )
        expected_temporal, _ = score_gaussian_aggregate_float64(
            l2_normalized_second_order(torch.from_numpy(patch)),
            temporal_params,
            aggregation="mean",
            device="cpu",
            compute_percentile=False,
        )
        np.testing.assert_array_equal(actual.patch_spatial, expected_spatial)
        np.testing.assert_array_equal(actual.patch_temporal, expected_temporal)

        d1 = local_branch.score_local_raw(
            patch,
            spatial_params,
            temporal_params,
            device="cpu",
            temporal_order=1,
        )
        self.assertEqual(d1.patch_temporal.shape, (3,))
        with self.assertRaisesRegex(ValueError, "must be 1 or 2"):
            local_branch.score_local_raw(
                patch,
                spatial_params,
                temporal_params,
                device="cpu",
                temporal_order=3,
            )

    def test_local_fusion_is_single_shared_implementation(self) -> None:
        self.assertIs(fuse_local_components, local_branch.fuse_local_components)
        np.testing.assert_allclose(
            fuse_local_components(np.array([0.2, 0.8]), np.array([0.6, 0.4])),
            [0.56, 0.44],
        )
        with self.assertRaisesRegex(ValueError, "weight must be"):
            fuse_local_components(np.array([0.2]), np.array([0.4]), spatial_weight=1.1)

    def test_formal_scorers_import_the_canonical_local_raw_function(self) -> None:
        self.assertIs(score_u0_locked_windows.score_local_raw, local_branch.score_local_raw)
        self.assertIs(score_u0_locked_k1_cache.score_local_raw, local_branch.score_local_raw)
        self.assertIs(
            score_duration_aware_original_k1.score_local_raw,
            local_branch.score_local_raw,
        )

    def test_patch_math_percentile_uses_canonical_tie_policy(self) -> None:
        reference = np.array([1.0, 2.0, 2.0, 3.0], dtype=np.float64)
        np.testing.assert_array_equal(
            empirical_percentile(np.array([2.0, 2.5]), reference),
            [0.75, 0.75],
        )
        with self.assertRaises(ValueError):
            empirical_percentile(np.array([np.nan]), reference)


if __name__ == "__main__":
    unittest.main()
