"""跨真实域阈值与矩阵统计的回归测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from config import load_config
from cross_domain import evaluate_cross_domain_cell, real_only_threshold
from math_utils import (
    GaussianMeanCandidateScorerFloat64,
    StableGaussianParams,
    score_gaussian_aggregate_float64,
)


class CrossDomainThresholdTests(unittest.TestCase):
    def test_threshold_never_exceeds_requested_empirical_fpr(self) -> None:
        scores = np.arange(200, dtype=np.float64)
        for target in (0.001, 0.01, 0.05):
            threshold = real_only_threshold(scores, target)
            self.assertLessEqual(float((scores < threshold).mean()), target)

    def test_threshold_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            real_only_threshold(np.array([]), 0.01)
        with self.assertRaises(ValueError):
            real_only_threshold(np.array([0.0, np.nan]), 0.01)
        with self.assertRaises(ValueError):
            real_only_threshold(np.array([0.0, 1.0]), 1.0)

    def test_multi_gaussian_moment_scorer_matches_direct_whitening(self) -> None:
        rng = np.random.default_rng(17)
        features = rng.normal(size=(3, 5, 4)).astype(np.float32)
        candidates = [
            StableGaussianParams(
                mean=rng.normal(size=4),
                whitening=rng.normal(size=(4, rank)),
                calibration_raw=np.array([0.0, 1.0]),
            )
            for rank in (2, 3)
        ]
        scorer = GaussianMeanCandidateScorerFloat64(
            candidates, center=np.mean([item.mean for item in candidates], axis=0),
            device="cpu",
        )
        actual = scorer.score(features)
        for index, params in enumerate(candidates):
            expected, _ = score_gaussian_aggregate_float64(
                features, params, "mean", device="cpu", compute_percentile=False
            )
            np.testing.assert_allclose(actual[:, index], expected, atol=1e-10)

    def test_cross_domain_cell_reports_real_only_operating_points(self) -> None:
        config = load_config(ROOT / "configs/benchmark.yaml")
        config["method"]["fusion"] = {"global_weight": 0.5, "local_weight": 0.5}

        def windows(prefix: str, subset: str, values: list[float]) -> pd.DataFrame:
            rows = []
            for index, value in enumerate(values):
                rows.append({
                    "video_id": f"{prefix}:{index}",
                    "dataset": prefix,
                    "subset": subset,
                    "source_model": "real" if subset == "real" else "fake-model",
                    "video_path": f"{prefix}/{index}.mp4",
                    "window_id": 0,
                    "global_spatial": value,
                    "global_t1": value,
                    "patch_temporal_raw": value,
                })
            return pd.DataFrame(rows)

        calibration = windows("source", "real", [0.1, 0.2, 0.8, 0.9])
        evaluation = pd.concat([
            windows("target-real", "real", [0.7, 0.9]),
            windows("target-fake", "annotated", [-0.2, 0.0]),
        ], ignore_index=True)
        summary, points, generators = evaluate_cross_domain_cell(
            calibration,
            evaluation,
            config,
            calibration_bank="source",
            evaluation_domain="target",
        )
        self.assertEqual(len(summary), 1)
        self.assertEqual(len(points), 3)
        self.assertEqual(len(generators), 1)
        self.assertTrue((points["calibration_empirical_fpr"] <= points["target_real_fpr"]).all())


if __name__ == "__main__":
    unittest.main()
