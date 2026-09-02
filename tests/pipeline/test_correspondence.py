"""Stage 1 局部对应与置信度聚合回归测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import apply_overrides, load_config
from correspondence.local import align_local_velocity
from dynamics.local import build_local_dynamics
from math_utils import (
    StableGaussianParams,
    l2_normalized_second_order,
    score_gaussian_aggregate_float64,
)


class CorrespondenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(ROOT / "configs/benchmark.yaml")

    def test_c0_same_grid_exactly_matches_historical_d2(self) -> None:
        generator = torch.Generator().manual_seed(13)
        patch = torch.randn(2, 5, 4, 8, generator=generator)
        result = build_local_dynamics(
            patch,
            grid_size=(2, 2),
            local_config=self.config["method"]["local"],
            device="cpu",
        )
        self.assertIsNone(result.aggregation_weights)
        torch.testing.assert_close(
            result.features,
            l2_normalized_second_order(patch),
            rtol=0.0,
            atol=0.0,
        )

    def test_hard_local_matching_follows_a_one_patch_shift(self) -> None:
        # 唯一 object token 每帧向右移动一格；半径 1 应找到该局部对应。
        patch = torch.zeros(1, 3, 3, 3)
        background = torch.tensor([0.0, 1.0, 0.0])
        object_token = torch.tensor([1.0, 0.0, 0.0])
        patch[:] = background
        patch[0, 0, 0] = object_token
        patch[0, 1, 1] = object_token
        patch[0, 2, 2] = object_token
        result = align_local_velocity(
            patch,
            grid_size=(1, 3),
            mode="hard_local",
            radius=1,
            spatial_penalty=0.0,
        )
        self.assertEqual(tuple(result.velocity.shape), (1, 2, 3, 3))
        torch.testing.assert_close(
            result.velocity[0, 0, 0], torch.zeros(3), rtol=0.0, atol=0.0
        )

    def test_soft_matching_returns_bounded_confidence(self) -> None:
        generator = torch.Generator().manual_seed(29)
        patch = torch.randn(2, 4, 9, 6, generator=generator)
        result = align_local_velocity(
            patch,
            grid_size=(3, 3),
            mode="soft_local",
            radius=1,
            temperature=0.07,
            spatial_penalty=0.05,
        )
        self.assertEqual(tuple(result.velocity.shape), (2, 3, 9, 6))
        self.assertEqual(tuple(result.confidence.shape), (2, 3, 9))
        self.assertTrue(torch.isfinite(result.confidence).all())
        self.assertGreaterEqual(float(result.confidence.min()), 0.0)
        self.assertLessEqual(float(result.confidence.max()), 1.0)

    def test_c3_confidence_shape_matches_d2_positions(self) -> None:
        local = apply_overrides(
            self.config,
            [
                "method.local.correspondence.type=soft_local",
                "method.local.correspondence.confidence=aggregation",
            ],
        )["method"]["local"]
        patch = torch.randn(2, 5, 9, 7, generator=torch.Generator().manual_seed(31))
        result = build_local_dynamics(
            patch, grid_size=(3, 3), local_config=local, device="cpu"
        )
        self.assertEqual(tuple(result.features.shape), (2, 3, 9, 7))
        self.assertEqual(tuple(result.aggregation_weights.shape), (2, 3, 9))

    def test_weighted_gaussian_mean_matches_manual_result(self) -> None:
        features = np.array([[[0.0], [1.0], [2.0]]], dtype=np.float32)
        weights = np.array([[1.0, 2.0, 1.0]], dtype=np.float64)
        params = StableGaussianParams(
            mean=np.array([0.0]),
            whitening=np.array([[1.0]]),
            calibration_raw=np.array([-10.0, 0.0]),
        )
        raw, _ = score_gaussian_aggregate_float64(
            features,
            params,
            "mean",
            device="cpu",
            position_weights=weights,
            compute_percentile=False,
        )
        likelihood = -0.5 * (np.log(2.0 * np.pi) + features[0, :, 0] ** 2)
        expected = np.sum(likelihood * weights[0]) / weights.sum()
        self.assertAlmostEqual(float(raw[0]), float(expected), places=12)

    def test_invalid_confidence_combination_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "只允许与 soft_local"):
            apply_overrides(
                self.config,
                [
                    "method.local.correspondence.type=hard_local",
                    "method.local.correspondence.confidence=aggregation",
                ],
            )

    def test_trajectory_geometry_has_expected_values_and_shapes(self) -> None:
        # 单个 patch 依次沿 x、x、y 移动：中间位置曲率分别为 0 和 1。
        points = torch.tensor(
            [[[[0.0, 0.0]], [[1.0, 0.0]], [[2.0, 0.0]], [[2.0, 1.0]]]]
        )
        expected_shapes = {
            "curvature": (1, 2, 1, 1),
            "speed_ratio": (1, 2, 1, 1),
            "path_chord": (1, 2, 1, 1),
            "d2_curvature": (1, 2, 1, 3),
            "geometry": (1, 2, 1, 4),
        }
        outputs = {}
        for dynamics, shape in expected_shapes.items():
            local = dict(self.config["method"]["local"])
            local["dynamics"] = dynamics
            result = build_local_dynamics(
                points, grid_size=(1, 1), local_config=local, device="cpu"
            )
            self.assertEqual(tuple(result.features.shape), shape)
            self.assertTrue(torch.isfinite(result.features).all())
            outputs[dynamics] = result.features
        torch.testing.assert_close(
            outputs["curvature"].flatten(), torch.tensor([0.0, 1.0])
        )
        torch.testing.assert_close(
            outputs["speed_ratio"], torch.zeros_like(outputs["speed_ratio"])
        )
        self.assertAlmostEqual(float(outputs["path_chord"][0, 0, 0, 0]), 0.0, places=5)
        self.assertAlmostEqual(
            float(outputs["path_chord"][0, 1, 0, 0]), 2.0**0.5 - 1.0, places=5
        )

    def test_zero_velocity_geometry_remains_finite(self) -> None:
        points = torch.zeros(2, 4, 4, 3)
        local = dict(self.config["method"]["local"])
        local["dynamics"] = "geometry"
        result = build_local_dynamics(
            points, grid_size=(2, 2), local_config=local, device="cpu"
        )
        self.assertTrue(torch.isfinite(result.features).all())
        # 零速度 turning angle 没有方向，协议将 curvature 定义为 0。
        self.assertTrue(torch.equal(result.features[..., 1], torch.zeros(2, 2, 4)))


if __name__ == "__main__":
    unittest.main()
