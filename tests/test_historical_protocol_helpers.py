from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from alpha_stalled import historical_window_analysis, legacy_local_d2_protocol
import analyze_clean_universal
import analyze_intermediate_layers
import analyze_joint_typicality
import analyze_multi_window_feasibility
import analyze_multi_window_scores
import analyze_unified_multiscale
import audit_videofeedback_whitening
import run_local_d2_residuals
import score_k1_calibration_from_cache


class HistoricalProtocolHelperTests(unittest.TestCase):
    def test_window_analysis_clis_reexport_shared_contract(self) -> None:
        shared = historical_window_analysis
        for module in (
            analyze_clean_universal,
            analyze_intermediate_layers,
            analyze_unified_multiscale,
            audit_videofeedback_whitening,
        ):
            self.assertIs(module.KEY_COLUMNS, shared.KEY_COLUMNS)
            self.assertIs(module.WINDOW_KEYS, shared.WINDOW_KEYS)
            self.assertIs(module.target_k_reference, shared.target_k_reference)
        self.assertIs(
            analyze_multi_window_scores.calibration_references,
            shared.calibration_references,
        )
        self.assertIs(
            analyze_joint_typicality.calibration_references,
            shared.calibration_references,
        )

    def test_target_k_reference_uses_real_calibration_only(self) -> None:
        rows = []
        for filename, split, subset, values in (
            ("a.mp4", "calibration", "real", (0.1, 0.5, 0.9)),
            ("b.mp4", "calibration", "real", (0.2, 0.4, 0.8)),
            ("fake.mp4", "calibration", "annotated", (9.0, 9.0, 9.0)),
            ("test.mp4", "evaluation", "real", (8.0, 8.0, 8.0)),
        ):
            for window_id, value in enumerate(values):
                rows.append(
                    {
                        "dataset": "d",
                        "protocol_split": split,
                        "subset": subset,
                        "source_model": "source",
                        "filename": filename,
                        "window_id": window_id,
                        "G_k": value,
                        "local": value + 0.1,
                    }
                )
        reference = historical_window_analysis.target_k_reference(
            pd.DataFrame(rows), 2, "local"
        )
        self.assertEqual(set(reference["filename"]), {"a.mp4", "b.mp4"})
        self.assertEqual(reference["G_mean_raw"].tolist(), [0.5, 0.5])
        np.testing.assert_allclose(reference["L_raw"], [0.6, 0.6])

    def test_local_d2_clis_reexport_shared_dataset_contract(self) -> None:
        shared = legacy_local_d2_protocol
        self.assertIs(run_local_d2_residuals.dataset_specs, shared.dataset_specs)
        self.assertIs(
            run_local_d2_residuals.build_strict_eval_index,
            shared.build_strict_eval_index,
        )
        self.assertIs(analyze_multi_window_feasibility.dataset_specs, shared.dataset_specs)
        self.assertIs(analyze_multi_window_feasibility.KEY_COLUMNS, shared.KEY_COLUMNS)
        self.assertIs(score_k1_calibration_from_cache.dataset_specs, shared.dataset_specs)

    def test_strict_index_recovers_stage1_identity_order_independently(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "eval.csv"
            pd.DataFrame(
                [
                    {
                        "subset": "real",
                        "source_model": "r",
                        "video_path": "videos/b.mp4",
                        "value": 2,
                    },
                    {
                        "subset": "annotated",
                        "source_model": "g",
                        "video_path": "videos/a.mp4",
                        "value": 1,
                    },
                ]
            ).to_csv(index, index=False)
            spec = legacy_local_d2_protocol.LocalDatasetSpec(
                "demo", root / "calib.csv", index, root / "cache", "mean", 0.5, 1
            )
            stage1 = pd.DataFrame(
                [
                    {
                        "dataset": "demo",
                        "subset": "annotated",
                        "source_model": "g",
                        "filename": "a.mp4",
                    },
                    {
                        "dataset": "demo",
                        "subset": "real",
                        "source_model": "r",
                        "filename": "b.mp4",
                    },
                ]
            )
            recovered = legacy_local_d2_protocol.build_strict_eval_index(spec, stage1)
            self.assertEqual(set(recovered["value"]), {1, 2})
            self.assertNotIn("filename", recovered)


if __name__ == "__main__":
    unittest.main()
