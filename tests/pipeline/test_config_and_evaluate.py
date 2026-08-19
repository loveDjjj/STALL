"""验证配置覆盖、方法约束和统一指标输出。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import apply_overrides, load_config, validate_config
from evaluation.tables import build_metric_tables, normalize_scores


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


if __name__ == "__main__":
    unittest.main()
