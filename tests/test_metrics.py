from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import metrics as legacy_metrics
from alpha_stalled import metrics

from tools import analyze_multi_window_scores, audit_u0_metric_protocol
from tools import analyze_u0_core_ablation, score_u0_locked_k1_cache
from alpha_stalled import u0_analysis


class MetricsContractTests(unittest.TestCase):
    def test_legacy_module_reexports_shared_implementation(self) -> None:
        self.assertIs(legacy_metrics.Score, metrics.Score)
        self.assertIs(legacy_metrics.ScoreDirection, metrics.ScoreDirection)
        self.assertIs(legacy_metrics.build_results_table, metrics.build_results_table)
        self.assertIs(
            legacy_metrics._sample_balanced_real,
            metrics.sample_balanced_real,
        )

    def test_historical_cli_helpers_reexport_shared_implementations(self) -> None:
        self.assertIs(
            analyze_multi_window_scores.macro_cluster_bootstrap,
            metrics.macro_cluster_bootstrap,
        )
        self.assertIs(audit_u0_metric_protocol.binary_metrics, metrics.binary_metrics)
        self.assertIs(audit_u0_metric_protocol.repeat_by_count, metrics.repeat_by_count)
        self.assertIs(audit_u0_metric_protocol.stable_seed, metrics.stable_seed)
        self.assertIs(score_u0_locked_k1_cache.calibrate_raw, u0_analysis.calibrate_raw)
        self.assertIs(
            score_u0_locked_k1_cache.calibration_raw_references,
            u0_analysis.calibration_raw_references,
        )
        self.assertIs(
            analyze_u0_core_ablation.calibrate_k3_candidate,
            u0_analysis.calibrate_k3_candidate,
        )

    def test_balanced_real_sampling_uses_equal_source_quotas(self) -> None:
        real = pd.DataFrame(
            [
                {"source_model": source, "row_id": f"{source}-{index}"}
                for source in ("RealA", "RealB")
                for index in range(5)
            ]
        )

        first = metrics.sample_balanced_real(real, n=4, seed=17)
        second = metrics.sample_balanced_real(real, n=4, seed=17)

        self.assertEqual(first["source_model"].value_counts().to_dict(), {"RealA": 2, "RealB": 2})
        self.assertEqual(first["row_id"].tolist(), second["row_id"].tolist())

    def test_pairwise_rows_balance_each_generator_against_real(self) -> None:
        rows = [
            {"subset": "real", "source_model": "RealA", "score": 0.9},
            {"subset": "real", "source_model": "RealA", "score": 0.8},
            {"subset": "real", "source_model": "RealB", "score": 0.7},
        ]
        rows.extend(
            {"subset": "annotated", "source_model": "GenA", "score": value}
            for value in (0.1, 0.2, 0.3, 0.4, 0.5)
        )
        rows.extend(
            {"subset": "annotated", "source_model": "GenB", "score": value}
            for value in (0.1, 0.2)
        )

        table = metrics.build_results_table(
            pd.DataFrame(rows),
            {"score": metrics.ScoreDirection.HIGHER_IS_REAL},
            skip_global_compare=True,
            verbose=False,
        ).set_index("Generative Model")

        self.assertEqual(int(table.loc["GenA", "n_real"]), 3)
        self.assertEqual(int(table.loc["GenA", "n_annotated"]), 3)
        self.assertEqual(int(table.loc["GenB", "n_real"]), 2)
        self.assertEqual(int(table.loc["GenB", "n_annotated"]), 2)
        self.assertAlmostEqual(float(table.loc["Average", "score AUC"]), 1.0)
        self.assertAlmostEqual(float(table.loc["Average", "score AP"]), 1.0)

    def test_score_direction_controls_positive_class_for_auc_and_ap(self) -> None:
        frame = pd.DataFrame(
            [
                {"subset": "real", "source_model": "Real", "realness": 0.9, "anomaly": 0.1},
                {"subset": "real", "source_model": "Real", "realness": 0.8, "anomaly": 0.2},
                {"subset": "annotated", "source_model": "Gen", "realness": 0.2, "anomaly": 0.8},
                {"subset": "annotated", "source_model": "Gen", "realness": 0.1, "anomaly": 0.9},
            ]
        )
        table = metrics.build_results_table(
            frame,
            {
                "realness": metrics.ScoreDirection.HIGHER_IS_REAL,
                "anomaly": metrics.ScoreDirection.HIGHER_IS_FAKE,
            },
            skip_global_compare=True,
            verbose=False,
        ).set_index("Generative Model")

        self.assertAlmostEqual(float(table.loc["Gen", "realness AUC"]), 1.0)
        self.assertAlmostEqual(float(table.loc["Gen", "realness AP"]), 1.0)
        self.assertAlmostEqual(float(table.loc["Gen", "anomaly AUC"]), 1.0)
        self.assertAlmostEqual(float(table.loc["Gen", "anomaly AP"]), 1.0)

    def test_predictor_metrics_accept_numpy_list_and_tensor_semantics(self) -> None:
        result = metrics.predictor_scalar2metrics(
            [0.1, 0.9, 0.2, 0.8],
            np.array([0, 1, 0, 1], dtype=np.uint8),
            threshold=0.5,
        )
        self.assertEqual(set(result), {"AUC", "AP", "F1_score", "Accuracy"})
        self.assertTrue(all(float(value) == 1.0 for value in result.values()))

    def test_results_require_real_and_generated_rows(self) -> None:
        frame = pd.DataFrame(
            [{"subset": "real", "source_model": "Real", "score": 0.9}]
        )
        with self.assertRaisesRegex(ValueError, "real.*annotated"):
            metrics.build_results_table(
                frame,
                {"score": metrics.ScoreDirection.HIGHER_IS_REAL},
                verbose=False,
            )

    def test_shared_metric_tables_use_generator_then_dataset_macro(self) -> None:
        rows = []
        for dataset in ("d1", "d2"):
            rows.extend(
                {"dataset": dataset, "subset": "real", "source_model": "Real", "score": value}
                for value in (0.8, 0.9)
            )
            rows.extend(
                {"dataset": dataset, "subset": "annotated", "source_model": generator, "score": value}
                for generator in ("G1", "G2")
                for value in (0.1, 0.2)
            )
        dataset_table, generator_table = metrics.metric_tables(
            pd.DataFrame(rows),
            seed=42,
            score_columns=("score",),
            config_names={"score": "Shared score"},
        )
        self.assertEqual(len(generator_table), 4)
        self.assertEqual(set(dataset_table["dataset"]), {"d1", "d2", "Macro-3"})
        self.assertTrue((dataset_table[["auc", "ap"]] == 1.0).all().all())
        macro = dataset_table[dataset_table["dataset"] == "Macro-3"].iloc[0]
        self.assertEqual(int(macro["n_generators"]), 4)

    def test_shared_paired_bootstrap_is_deterministic(self) -> None:
        rows = []
        for subset, source, base, new in (
            ("real", "Real", 0.7, 0.9),
            ("annotated", "Gen", 0.3, 0.1),
        ):
            rows.extend(
                {
                    "dataset": "demo",
                    "subset": subset,
                    "source_model": source,
                    "base": base + 0.01 * index,
                    "new": new + 0.01 * index,
                }
                for index in range(6)
            )
        frame = pd.DataFrame(rows)
        kwargs = {
            "seed": 17,
            "iterations": 20,
            "comparisons": (("new", "base", "new_vs_base"),),
        }
        first = metrics.paired_bootstrap(frame, **kwargs)
        second = metrics.paired_bootstrap(frame, **kwargs)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(set(first["metric"]), {"auc", "ap"})
        self.assertTrue((first["delta"] >= 0).all())


if __name__ == "__main__":
    unittest.main()
