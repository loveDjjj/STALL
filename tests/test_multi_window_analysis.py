from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from analyze_multi_window_scores import (
    add_frozen_baseline,
    add_recalibrated_scores,
    aggregate_window_scores,
    calibration_references,
    macro_cluster_bootstrap,
    stratified_metric_tables,
)


class MultiWindowAnalysisTests(unittest.TestCase):
    def test_branch_aggregations_are_video_weighted(self) -> None:
        rows = []
        for window_id, local in enumerate((0.1, 0.5, 0.9)):
            rows.append(
                {
                    "dataset": "d",
                    "protocol_split": "calibration",
                    "subset": "real",
                    "source_model": "m",
                    "filename": "v.mp4",
                    "duration_seconds": 6.0,
                    "effective_k": 3,
                    "unique_frame_count": 48,
                    "window_id": window_id,
                    "G_k": 0.6,
                    "L_k": local,
                    "S_k": 0.6 * 0.6 + 0.4 * local,
                }
            )
        result = aggregate_window_scores(pd.DataFrame(rows)).iloc[0]
        self.assertAlmostEqual(result.G_mean_raw, 0.6)
        self.assertAlmostEqual(result.L_mean_raw, 0.5)
        self.assertAlmostEqual(result.L_bottom2_raw, 0.3)
        self.assertAlmostEqual(result.L_hybrid_raw, 0.4)
        self.assertEqual(result.windows, 3)

    def test_target_k_reference_keeps_one_row_per_video(self) -> None:
        rows = []
        for filename, values in (("a.mp4", (0.1, 0.5, 0.9)), ("b.mp4", (0.2, 0.4, 0.8))):
            for window_id, value in enumerate(values):
                rows.append(
                    {
                        "dataset": "d",
                        "protocol_split": "calibration",
                        "subset": "real",
                        "source_model": "m",
                        "filename": filename,
                        "window_id": window_id,
                        "G_k": value,
                        "L_k": value,
                    }
                )
        reference = calibration_references(pd.DataFrame(rows), target_k=2)
        self.assertEqual(len(reference), 2)
        self.assertAlmostEqual(reference.loc[reference.filename.eq("a.mp4"), "L_mean_raw"].iloc[0], 0.5)
        self.assertAlmostEqual(reference.loc[reference.filename.eq("a.mp4"), "L_bottom2_raw"].iloc[0], 0.5)
        single = calibration_references(pd.DataFrame(rows), target_k=1)
        self.assertAlmostEqual(single.loc[single.filename.eq("a.mp4"), "L_mean_raw"].iloc[0], 0.5)

    def test_baseline_join_restores_frozen_protocol_order(self) -> None:
        evaluation = pd.DataFrame(
            [
                {"dataset": "d", "protocol_split": "evaluation", "subset": "real", "source_model": "m", "filename": "b.mp4"},
                {"dataset": "d", "protocol_split": "evaluation", "subset": "real", "source_model": "m", "filename": "a.mp4"},
            ]
        )
        baseline = pd.DataFrame(
            [
                {"dataset": "d", "subset": "real", "source_model": "m", "filename": "a.mp4", "B2": 0.1, "P0": 0.2, "final_selected": 0.3},
                {"dataset": "d", "subset": "real", "source_model": "m", "filename": "b.mp4", "B2": 0.4, "P0": 0.5, "final_selected": 0.6},
            ]
        )
        path = Path(self.id().replace(".", "_") + ".csv")
        try:
            baseline.to_csv(path, index=False)
            merged = add_frozen_baseline(evaluation, path)
            self.assertEqual(merged.filename.tolist(), ["a.mp4", "b.mp4"])
        finally:
            path.unlink(missing_ok=True)

    def test_macro_bootstrap_weights_datasets_not_generators(self) -> None:
        rows = []
        for dataset, generators in (("a", ("g1",)), ("b", ("g1", "g2", "g3"))):
            for index in range(4):
                rows.append(
                    {
                        "dataset": dataset,
                        "subset": "real",
                        "source_model": "real",
                        "filename": f"real-{index}",
                        "MW0": 0.0 if dataset == "a" else 1.0,
                        "MW1": 1.0,
                    }
                )
            for generator in generators:
                for index in range(4):
                    rows.append(
                        {
                            "dataset": dataset,
                            "subset": "annotated",
                            "source_model": generator,
                            "filename": f"{generator}-{index}",
                            "MW0": 1.0 if dataset == "a" else 0.0,
                            "MW1": 0.0,
                        }
                    )
        result = macro_cluster_bootstrap(
            pd.DataFrame(rows), ["MW1"], seed=42, iterations=10
        )
        auc = result[result.metric.eq("auc")].iloc[0]
        self.assertAlmostEqual(auc.delta, 0.5)

    def test_unconditional_calibration_accepts_unseen_effective_k(self) -> None:
        rows = []
        for index in range(200):
            for window_id in range(2):
                rows.append(
                    {
                        "dataset": "d",
                        "protocol_split": "calibration",
                        "subset": "real",
                        "source_model": "real",
                        "filename": f"real-{index}.mp4",
                        "duration_seconds": 4.0,
                        "effective_k": 2,
                        "unique_frame_count": 32,
                        "window_id": window_id,
                        "G_k": index / 200,
                        "L_k": index / 200,
                        "S_k": index / 200,
                    }
                )
        for window_id in range(125):
            rows.append(
                {
                    "dataset": "d",
                    "protocol_split": "evaluation",
                    "subset": "annotated",
                    "source_model": "fake",
                    "filename": "long.mp4",
                    "duration_seconds": 250.0,
                    "effective_k": 125,
                    "unique_frame_count": 2000,
                    "window_id": window_id,
                    "G_k": 0.25,
                    "L_k": 0.25,
                    "S_k": 0.25,
                }
            )
        windows = pd.DataFrame(rows)
        result = add_recalibrated_scores(
            aggregate_window_scores(windows),
            windows,
            calibration_mode="unconditional",
        )
        self.assertEqual(result.calibration_reference_n.iloc[0], 200)
        self.assertEqual(result.calibration_mode.iloc[0], "unconditional")
        self.assertTrue(0.0 <= result.MW4.iloc[0] <= 1.0)

    def test_stratified_tables_include_dataset_and_generator_metrics(self) -> None:
        rows = []
        for subset, score in (("real", 1.0), ("annotated", 0.0)):
            for index in range(4):
                rows.append(
                    {
                        "dataset": "d",
                        "subset": subset,
                        "source_model": "real" if subset == "real" else "fake",
                        "filename": f"{subset}-{index}.mp4",
                        "duration_bin": "2-4",
                        "MW0": score,
                    }
                )
        datasets, generators = stratified_metric_tables(
            pd.DataFrame(rows),
            "duration_bin",
            seed=42,
            config_names={"MW0": "baseline"},
        )
        self.assertEqual(set(datasets.dataset), {"d", "Macro-3"})
        self.assertEqual(generators.generator.tolist(), ["fake"])


if __name__ == "__main__":
    unittest.main()
