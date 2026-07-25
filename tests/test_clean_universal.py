from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


class CleanUniversalTests(unittest.TestCase):
    def test_u0_is_predeclared_region1_mean(self) -> None:
        from analyze_clean_universal import UNIVERSAL_CONFIGS

        self.assertEqual(UNIVERSAL_CONFIGS["U0"], (1, "mean"))

    def test_c1_is_recalibrated_minimum(self) -> None:
        from analyze_clean_universal import add_cross_layer_temporal_scores

        frame = pd.DataFrame(
            {
                "dataset": ["d"] * 4,
                "protocol_split": ["calibration"] * 2 + ["evaluation"] * 2,
                "subset": ["real", "real", "real", "annotated"],
                "region1_mean_temporal": [0.2, 0.8, 0.5, 0.1],
                "layer17_temporal": [0.4, 0.6, 0.9, 0.3],
                "mean_motion_magnitude": [1.0, 3.0, 2.0, 4.0],
                "patch_spatial": [0.5] * 4,
            }
        )
        out = add_cross_layer_temporal_scores(frame)
        np.testing.assert_allclose(out["C1_raw"], [0.2, 0.6, 0.5, 0.1])
        np.testing.assert_allclose(out["C1_temporal"], [0.5, 1.0, 0.5, 0.0])
        self.assertEqual(float(out.loc[2, "motion_real_cdf"]), 0.5)
        self.assertEqual(float(out.loc[2, "C2_temporal"]), 0.5)
        self.assertEqual(float(out.loc[3, "C2_temporal"]), 0.1)

    def test_raw_scores_are_recalibrated_from_final_params(self) -> None:
        from tempfile import TemporaryDirectory

        from analyze_clean_universal import canonicalize_temporal_percentiles

        with TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {
                "calib_patch_temp_scores": np.array([0.0, 1.0], dtype=np.float32)
            }
            for region in (1, 2, 3):
                for aggregation in ("mean", "bottom20"):
                    np.savez(root / f"d_region{region}_{aggregation}.npz", **payload)
            np.savez(root / "d_layer17_region1_mean.npz", **payload)
            row = {"dataset": "d", "layer17_raw": 0.5, "layer17_temporal": -1.0}
            for region in (1, 2, 3):
                for aggregation in ("mean", "bottom20"):
                    row[f"region{region}_{aggregation}_raw"] = 1.0
                    row[f"region{region}_{aggregation}_temporal"] = -1.0
            out = canonicalize_temporal_percentiles(pd.DataFrame([row]), root)
            self.assertEqual(float(out.loc[0, "layer17_temporal"]), 0.5)
            self.assertEqual(float(out.loc[0, "region3_bottom20_temporal"]), 1.0)

    def test_motion_thresholds_use_calibration_real_only(self) -> None:
        from analyze_clean_universal import real_motion_thresholds

        frame = pd.DataFrame(
            {
                "dataset": ["d"] * 3,
                "protocol_split": ["calibration", "calibration", "evaluation"],
                "subset": ["real", "real", "real"],
                "source_model": ["r", "r", "r"],
                "filename": ["a", "b", "c"],
                "mean_motion_magnitude": [1.0, 3.0, 1000.0],
            }
        )
        result = real_motion_thresholds(frame).iloc[0]
        self.assertEqual(result.real_calibration_videos, 2)
        self.assertEqual(result.window_motion_median_for_C2, 2.0)

    def test_release_metrics_preserve_identity_controls(self) -> None:
        metrics = pd.read_csv(
            ROOT / "results/clean_universal/universal_dataset_metrics.csv"
        )
        dataset_rows = metrics[metrics["dataset"] != "Macro-3"].set_index(
            ["dataset", "config"]
        )
        for dataset, identity_config in {
            "comgenvid": "U5",
            "videofeedback": "U0",
            "genvideo": "U1",
        }.items():
            historical = dataset_rows.loc[
                (dataset, "HistoricalTuned"), ["auc", "ap"]
            ]
            identity = dataset_rows.loc[(dataset, identity_config), ["auc", "ap"]]
            np.testing.assert_allclose(historical, identity, atol=0.0, rtol=0.0)

        cross_layer = pd.read_csv(
            ROOT / "results/clean_universal/cross_layer_dataset_metrics.csv"
        ).set_index(["dataset", "config"])
        for dataset in ("comgenvid", "videofeedback", "genvideo", "Macro-3"):
            u0 = metrics[
                (metrics["dataset"] == dataset) & (metrics["config"] == "U0")
            ].iloc[0]
            c0 = cross_layer.loc[(dataset, "C0"), ["auc", "ap"]]
            np.testing.assert_allclose(
                u0[["auc", "ap"]].astype(float), c0.astype(float), atol=0.0, rtol=0.0
            )

    def test_release_calibration_and_admission_audits(self) -> None:
        calibration = pd.read_csv(
            ROOT / "results/clean_universal/calibration_audit.csv"
        )
        self.assertTrue((calibration["calibration_real_videos"] == 200).all())
        self.assertTrue((calibration["calibration_fake_videos"] == 0).all())
        self.assertTrue(
            (calibration["calibration_evaluation_video_overlap"] == 0).all()
        )

        admission = pd.read_csv(
            ROOT / "results/clean_universal/cross_layer_admission.csv"
        )
        self.assertEqual(set(admission["config"]), {"C1", "C2"})
        self.assertFalse(admission["admitted"].astype(bool).any())

    def test_float64_audit_matches_official_batch4_percentiles(self) -> None:
        audit = pd.read_csv(
            ROOT / "results/clean_universal/numerical_audit/float64_recheck.csv"
        )
        self.assertGreater(len(audit), 0)
        np.testing.assert_allclose(
            audit["float64_percentile"],
            audit["batch4_percentile"],
            atol=3e-8,
            rtol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
