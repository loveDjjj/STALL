"""锁定官方 STALL VATEX Global 参数的加载与窗口评分协议。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from branches.global_branch import load_official_stall_parameters
from math_utils import l2_normalized_first_order, score_gaussian_aggregate_float64


class OfficialGlobalTests(unittest.TestCase):
    """Global 必须使用官方的 max/min 视频统计与右包含 CDF。"""

    def test_official_parameter_loader_uses_video_max_and_min_references(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "official.npz"
            np.savez(
                path,
                mu_spat=np.zeros(2, dtype=np.float32),
                W_spat=np.eye(2, dtype=np.float32),
                calib_ll_spat=np.array([[-5.0, -2.0], [-4.0, -3.0]]),
                mu_temp=np.zeros(2, dtype=np.float32),
                W_temp=np.eye(2, dtype=np.float32),
                calib_ll_temp=np.array([[-1.0, -7.0], [-3.0, -4.0]]),
            )
            parameters = load_official_stall_parameters(path)

        np.testing.assert_array_equal(parameters["global_spatial"].calibration_raw, [-3.0, -2.0])
        np.testing.assert_array_equal(parameters["global_t1"].calibration_raw, [-7.0, -4.0])

    def test_window_score_matches_official_aggregation_and_zero_difference_rule(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "official.npz"
            np.savez(
                path,
                mu_spat=np.zeros(2, dtype=np.float32),
                W_spat=np.eye(2, dtype=np.float32),
                calib_ll_spat=np.array([[-5.0, -3.0], [-4.0, -2.0]]),
                mu_temp=np.zeros(2, dtype=np.float32),
                W_temp=np.eye(2, dtype=np.float32),
                calib_ll_temp=np.array([[-6.0, -5.0], [-4.0, -3.0]]),
            )
            parameters = load_official_stall_parameters(path)

        # 第一条窗口含一个零差分。官方 STALL 在 min 聚合前将它设为 +inf，
        # 因而只由其余有效转移决定时间分数。
        features = torch.tensor(
            [[[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]], dtype=torch.float32
        )
        spatial_raw, spatial_cdf = score_gaussian_aggregate_float64(
            features, parameters["global_spatial"], "max", device="cpu"
        )
        temporal, zero_mask = l2_normalized_first_order(features)
        temporal_raw, temporal_cdf = score_gaussian_aggregate_float64(
            temporal, parameters["global_t1"], "min", device="cpu", invalid_mask=zero_mask
        )
        expected_spatial = -np.log(2.0 * np.pi)
        # 二维单位向量经 I 白化后平方范数为 1。
        expected_temporal = -0.5 * (2.0 * np.log(2.0 * np.pi) + 1.0)
        np.testing.assert_allclose(spatial_raw, [expected_spatial], rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(temporal_raw, [expected_temporal], rtol=0.0, atol=1e-12)
        self.assertEqual(float(spatial_cdf[0]), 1.0)
        self.assertEqual(float(temporal_cdf[0]), 1.0)


if __name__ == "__main__":
    unittest.main()
