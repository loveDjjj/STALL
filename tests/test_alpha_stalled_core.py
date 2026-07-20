from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval_alpha_stalled import compute_metrics, fuse_scores
from eval_score_csv import evaluate_score_csv
from patch_math import bottomk_mean, empirical_percentile
from patch_matching import patch_temporal_delta, pool_patch_regions, same_grid_finite_difference


def _write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


class AlphaStalledFusionTest(unittest.TestCase):
    def test_fuse_scores_uses_global_patch_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            global_csv = root / "global.csv"
            patch_csv = root / "patch.csv"
            rows = [
                {"subset": "real", "source_model": "RealSet", "filename": "r.mp4"},
                {"subset": "annotated", "source_model": "GenA", "filename": "f.mp4"},
            ]
            _write_csv(
                global_csv,
                [
                    {**rows[0], "final_score": 0.8},
                    {**rows[1], "final_score": 0.2},
                ],
            )
            _write_csv(
                patch_csv,
                [
                    {**rows[0], "patch_final_score": 0.6},
                    {**rows[1], "patch_final_score": 0.4},
                ],
            )

            fused = fuse_scores(
                global_csv,
                patch_csv,
                alpha=0.75,
                global_score_col="final_score",
                patch_score_col="patch_final_score",
            )

            self.assertEqual(len(fused), 2)
            self.assertAlmostEqual(float(fused.loc[0, "final_score"]), 0.75 * 0.8 + 0.25 * 0.6)
            self.assertAlmostEqual(float(fused.loc[1, "final_score"]), 0.75 * 0.2 + 0.25 * 0.4)

    def test_fuse_scores_rejects_non_one_to_one_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            global_csv = root / "global.csv"
            patch_csv = root / "patch.csv"
            key = {"subset": "real", "source_model": "RealSet", "filename": "r.mp4"}
            _write_csv(global_csv, [{**key, "final_score": 0.8}])
            _write_csv(
                patch_csv,
                [
                    {**key, "patch_final_score": 0.6},
                    {**key, "patch_final_score": 0.7},
                ],
            )

            with self.assertRaises(ValueError):
                fuse_scores(
                    global_csv,
                    patch_csv,
                    alpha=0.5,
                    global_score_col="final_score",
                    patch_score_col="patch_final_score",
                )

    def test_compute_metrics_returns_pairwise_average(self) -> None:
        df = pd.DataFrame(
            [
                {"subset": "real", "source_model": "RealSet", "filename": "r1.mp4", "final_score": 0.9},
                {"subset": "real", "source_model": "RealSet", "filename": "r2.mp4", "final_score": 0.8},
                {"subset": "annotated", "source_model": "GenA", "filename": "f1.mp4", "final_score": 0.2},
                {"subset": "annotated", "source_model": "GenA", "filename": "f2.mp4", "final_score": 0.1},
                {"subset": "annotated", "source_model": "GenB", "filename": "f3.mp4", "final_score": 0.3},
                {"subset": "annotated", "source_model": "GenB", "filename": "f4.mp4", "final_score": 0.4},
            ]
        )
        metrics = compute_metrics(df, seed=42)
        self.assertIn("Average", set(metrics["Generative Model"]))
        avg = metrics[metrics["Generative Model"] == "Average"].iloc[0]
        self.assertAlmostEqual(float(avg["final_score AUC"]), 1.0)
        self.assertAlmostEqual(float(avg["final_score AP"]), 1.0)

    def test_eval_score_csv_uses_requested_score_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.csv"
            _write_csv(
                path,
                [
                    {"subset": "real", "source_model": "RealSet", "filename": "r1.mp4", "patch_final_score": 0.9},
                    {"subset": "real", "source_model": "RealSet", "filename": "r2.mp4", "patch_final_score": 0.8},
                    {"subset": "annotated", "source_model": "GenA", "filename": "f1.mp4", "patch_final_score": 0.2},
                    {"subset": "annotated", "source_model": "GenA", "filename": "f2.mp4", "patch_final_score": 0.1},
                    {"subset": "annotated", "source_model": "GenB", "filename": "f3.mp4", "patch_final_score": 0.3},
                    {"subset": "annotated", "source_model": "GenB", "filename": "f4.mp4", "patch_final_score": 0.4},
                ],
            )

            metrics = evaluate_score_csv(path, "patch_final_score", seed=42)

            avg = metrics[metrics["Generative Model"] == "Average"].iloc[0]
            self.assertAlmostEqual(float(avg["patch_final_score AUC"]), 1.0)
            self.assertAlmostEqual(float(avg["patch_final_score AP"]), 1.0)

    def test_eval_score_csv_respects_higher_is_fake_direction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.csv"
            _write_csv(
                path,
                [
                    {"subset": "real", "source_model": "RealSet", "filename": "r1.mp4", "fake_score": 0.1},
                    {"subset": "real", "source_model": "RealSet", "filename": "r2.mp4", "fake_score": 0.2},
                    {"subset": "annotated", "source_model": "GenA", "filename": "f1.mp4", "fake_score": 0.8},
                    {"subset": "annotated", "source_model": "GenA", "filename": "f2.mp4", "fake_score": 0.9},
                    {"subset": "annotated", "source_model": "GenB", "filename": "f3.mp4", "fake_score": 0.7},
                    {"subset": "annotated", "source_model": "GenB", "filename": "f4.mp4", "fake_score": 0.6},
                ],
            )

            metrics = evaluate_score_csv(path, "fake_score", seed=42, higher_is="fake")

            avg = metrics[metrics["Generative Model"] == "Average"].iloc[0]
            self.assertAlmostEqual(float(avg["fake_score AUC"]), 1.0)
            self.assertAlmostEqual(float(avg["fake_score AP"]), 1.0)

    def test_same_grid_second_order_temporal_features(self) -> None:
        patch = np.array(
            [
                [[0.0, 0.0]],
                [[1.0, 0.0]],
                [[3.0, 0.0]],
                [[6.0, 0.0]],
            ],
            dtype=np.float32,
        )

        features = patch_temporal_delta(
            patch,
            grid_size=(1, 1),
            mode="same_grid_second_order",
            region_size=1,
        )

        self.assertEqual(tuple(features.shape), (2, 1, 2))
        expected = np.array([[[1.0, 0.0]], [[1.0, 0.0]]], dtype=np.float32)
        np.testing.assert_allclose(features, expected, atol=1e-6)

    def test_same_grid_higher_order_temporal_features(self) -> None:
        t = np.arange(6, dtype=np.float32)
        # Cubic polynomial: third finite difference is constant and non-zero.
        cubic = (t ** 3).reshape(-1, 1, 1)
        third = patch_temporal_delta(
            np.concatenate([cubic, np.zeros_like(cubic)], axis=2),
            grid_size=(1, 1),
            mode="same_grid_third_order",
            region_size=1,
        )
        self.assertEqual(tuple(third.shape), (3, 1, 2))
        np.testing.assert_allclose(third, np.tile([[[1.0, 0.0]]], (3, 1, 1)), atol=1e-6)

        # Quartic polynomial: fourth finite difference is constant and non-zero.
        quartic = (t ** 4).reshape(-1, 1, 1)
        fourth = same_grid_finite_difference(
            np.concatenate([quartic, np.zeros_like(quartic)], axis=2),
            order=4,
        )
        self.assertEqual(tuple(fourth.shape), (2, 1, 2))
        np.testing.assert_allclose(fourth, np.tile([[[1.0, 0.0]]], (2, 1, 1)), atol=1e-6)

    def test_pool_patch_regions_averages_non_overlapping_blocks(self) -> None:
        patch = np.arange(16, dtype=np.float32).reshape(1, 4, 4)

        pooled, grid = pool_patch_regions(
            patch,
            grid_size=(2, 2),
            region_size=2,
        )

        self.assertEqual(grid, (1, 1))
        self.assertEqual(tuple(pooled.shape), (1, 1, 4))
        np.testing.assert_allclose(pooled[0, 0], np.array([6.0, 7.0, 8.0, 9.0], dtype=np.float32))

    def test_bottomk_mean_uses_lowest_likelihood_values(self) -> None:
        ll = np.array(
            [
                [[5.0, 1.0], [3.0, 2.0]],
                [[10.0, -2.0], [4.0, 8.0]],
            ],
            dtype=np.float32,
        )

        out = bottomk_mean(ll, ratio=0.5)

        np.testing.assert_allclose(out, np.array([1.5, 1.0], dtype=np.float32), atol=1e-6)

    def test_percentile_uses_right_side_empirical_cdf(self) -> None:
        calib = np.array([1.0, 2.0, 2.0, 4.0], dtype=np.float32)
        scores = np.array([0.5, 2.0, 3.0, 4.0], dtype=np.float32)

        out = empirical_percentile(scores, calib)

        np.testing.assert_allclose(out, np.array([0.0, 0.75, 0.75, 1.0]))


if __name__ == "__main__":
    unittest.main()
