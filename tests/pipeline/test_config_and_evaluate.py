"""验证配置覆盖、方法约束和统一指标输出。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import apply_overrides, load_config, validate_config
from evaluation.tables import build_metric_tables, build_pairwise_metric_table, normalize_scores
from evaluation.metrics import paired_bootstrap
from math_utils import WhiteningTransform


class ConfigAndEvaluateTests(unittest.TestCase):
    """新主干不依赖历史实验文件名的最小回归测试。"""

    def setUp(self) -> None:
        self.config = load_config(ROOT / "configs/benchmark.yaml")

    def test_global_only_ablation_disables_local_branch(self) -> None:
        config = apply_overrides(
            self.config,
            ["method.local.enabled=false"],
        )
        validate_config(config)
        self.assertFalse(config["method"]["local"]["enabled"])

    def test_default_method_is_refit_d2_only_k3(self) -> None:
        local = self.config["method"]["local"]
        self.assertEqual(local["parameter_source"], "fit_real_only")
        self.assertFalse(local["spatial_enabled"])
        self.assertTrue(local["temporal_enabled"])
        self.assertEqual(local["temporal_order"], 2)
        self.assertEqual(local["spatial_weight"], 0.0)
        self.assertEqual(local["temporal_weight"], 1.0)
        self.assertEqual(self.config["sampling"]["num_windows"], 3)

    def test_method_requires_at_least_one_evidence_branch(self) -> None:
        with self.assertRaisesRegex(ValueError, "至少必须启用"):
            apply_overrides(
                self.config,
                ["method.global.enabled=false", "method.local.enabled=false"],
            )

    def test_d1_and_k1_are_explicit_config_changes(self) -> None:
        config = apply_overrides(
            self.config,
            ["method.local.temporal_order=1", "sampling.num_windows=1"],
        )
        self.assertEqual(config["method"]["local"]["temporal_order"], 1)
        self.assertEqual(config["sampling"]["num_windows"], 1)

    def test_oas_whitening_regularizes_ill_conditioned_covariance(self) -> None:
        rng = np.random.default_rng(23)
        shared = rng.normal(size=(128, 1))
        # 后三维几乎重复，经验协方差会产生非常小的特征值。
        values = np.concatenate(
            [shared, shared + rng.normal(scale=1e-6, size=(128, 3))], axis=1
        ).astype(np.float32)
        empirical = WhiteningTransform(values, device="cpu")
        oas = WhiteningTransform(values, device="cpu", covariance_estimator="oas")
        self.assertGreater(oas.shrinkage_, 0.0)
        self.assertGreater(float(oas.eigenvalues_.min()), float(empirical.eigenvalues_.min()))

    def test_empirical_whitening_supports_scalar_descriptor(self) -> None:
        values = np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(-1, 1)
        transform = WhiteningTransform(values, device="cpu", covariance_estimator="empirical")
        self.assertEqual(tuple(transform.whitening_matrix_.shape), (1, 1))
        whitened = transform.transform(values)
        self.assertEqual(tuple(whitened.shape), (32, 1))
        self.assertTrue(bool(torch.isfinite(whitened).all()))

    def test_standard_score_csv_produces_dataset_and_generator_tables(self) -> None:
        scores = normalize_scores(
            pd.DataFrame(
                [
                    {"video_id": "r1", "dataset": "demo", "subset": "real", "source_model": "real", "final_score": 0.9},
                    {"video_id": "r2", "dataset": "demo", "subset": "real", "source_model": "real", "final_score": 0.8},
                    {"video_id": "f1", "dataset": "demo", "subset": "annotated", "source_model": "generator_a", "final_score": 0.2},
                    {"video_id": "f2", "dataset": "demo", "subset": "annotated", "source_model": "generator_a", "final_score": 0.1},
                ]
            )
        )
        dataset_metrics, generator_metrics = build_metric_tables(scores, "smoke")
        self.assertEqual(dataset_metrics.loc[0, "dataset"], "demo")
        self.assertEqual(float(dataset_metrics.loc[0, "auc"]), 1.0)
        self.assertEqual(generator_metrics.loc[0, "generator"], "generator_a")

    def test_pairwise_table_balances_each_generator_and_adds_macro3(self) -> None:
        scores = normalize_scores(
            pd.DataFrame(
                [
                    {"video_id": "r1", "dataset": "demo", "subset": "real", "source_model": "real_a", "final_score": 0.9},
                    {"video_id": "r2", "dataset": "demo", "subset": "real", "source_model": "real_b", "final_score": 0.8},
                    {"video_id": "f1", "dataset": "demo", "subset": "annotated", "source_model": "generator_a", "final_score": 0.2},
                    {"video_id": "f2", "dataset": "demo", "subset": "annotated", "source_model": "generator_a", "final_score": 0.1},
                ]
            )
        )
        table = build_pairwise_metric_table(scores, "smoke", 42)
        self.assertEqual(table["dataset"].tolist(), ["demo", "Macro-3"])
        self.assertEqual(table.loc[0, "n_pairwise_real"], 2)
        self.assertEqual(table.loc[0, "n_pairwise_fake"], 2)
        self.assertEqual(float(table.loc[1, "auc"]), 1.0)

    def test_bootstrap_accepts_deterministic_hashed_seed(self) -> None:
        scores = pd.DataFrame(
            [
                {"dataset": "demo", "subset": "real", "source_model": "real", "final_score": 0.9, "global_score": 0.8},
                {"dataset": "demo", "subset": "real", "source_model": "real", "final_score": 0.8, "global_score": 0.7},
                {"dataset": "demo", "subset": "annotated", "source_model": "generator_a", "final_score": 0.2, "global_score": 0.3},
                {"dataset": "demo", "subset": "annotated", "source_model": "generator_a", "final_score": 0.1, "global_score": 0.2},
            ]
        )
        result = paired_bootstrap(
            scores, seed=17, iterations=3,
            comparisons=(("final_score", "global_score", "final_vs_global"),),
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "comparison"], "final_vs_global")


if __name__ == "__main__":
    unittest.main()
