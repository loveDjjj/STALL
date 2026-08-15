import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from analyze_second_order_independent_calibration import build_variants


class D1D2VariantDefinitionTest(unittest.TestCase):
    def test_temporal_only_and_full_variants_are_distinct(self):
        values = {
            "patch_d1": 0.20,
            "patch_d2": 0.30,
            "L_d1_k": 0.40,
            "L_k": 0.50,
        }

        def calibrated(_windows, column):
            return pd.DataFrame({"video_id": ["v"], "score": [values[column]]})

        locked = pd.DataFrame(
            {
                "video_id": ["v"],
                "dataset": ["comgenvid"],
                "protocol_split": ["evaluation"],
                "subset": ["real"],
                "source_model": ["real"],
                "filename": ["v.mp4"],
                "G": [0.60],
                "L": [0.50],
                "S": [0.56],
            }
        )
        with patch(
            "analyze_second_order_independent_calibration.calibrate_k3_candidate",
            side_effect=calibrated,
        ):
            result, regression = build_variants(pd.DataFrame(), locked)

        row = result.iloc[0]
        self.assertEqual(row.local_d1, 0.20)
        self.assertEqual(row.local_d2, 0.30)
        self.assertEqual(row.local_branch_d1, 0.40)
        self.assertEqual(row.local_branch_d2, 0.50)
        self.assertAlmostEqual(row.full_d1, 0.6 * 0.60 + 0.4 * 0.40)
        self.assertAlmostEqual(row.lstl, 0.56)
        self.assertEqual(regression["local_branch_d2_max_abs_error"], 0.0)
        self.assertEqual(regression["lstl_max_abs_error"], 0.0)


class IndependentCalibrationArtifactsTest(unittest.TestCase):
    def test_remaining_real_manifest_is_strict_and_disjoint(self):
        path = ROOT / "release/u0/independent_remaining_real_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["video_count"], 3080)
        self.assertEqual(payload["effective_k_counts"], {"1": 1037, "3": 2043})
        supplemental = {item["video_id"] for item in payload["videos"]}
        self.assertEqual(len(supplemental), 3080)
        known = set()
        for name in (
            "calibration_manifest.json",
            "evaluation_manifest.json",
            "calibration_reserve_manifest.json",
        ):
            source = json.loads((ROOT / "release/u0" / name).read_text(encoding="utf-8"))
            known.update(item["video_id"] for item in source["videos"])
        self.assertFalse(supplemental & known)
        for item in payload["videos"]:
            self.assertEqual(item["dataset"], "videofeedback")
            self.assertEqual(item["subset"], "real")
            self.assertEqual(item["effective_k"], len(item["k3_windows"]))
            for window in item["k3_windows"]:
                self.assertEqual(len(window), 16)
                self.assertEqual(len(set(window)), 16)

    def test_all_remaining_real_splits_are_disjoint_and_complete(self):
        expected_test_real = {
            "comgenvid": 1498,
            "videofeedback": 3880,
            "genvideo": 9784,
        }
        expected_fixed_test_real = {
            "comgenvid": 898,
            "videofeedback": 500,
            "genvideo": 7984,
        }
        expected_reserve_test_real = {
            "comgenvid": 400,
            "videofeedback": 100,
            "genvideo": 1600,
        }
        expected_historical_calibration_test_real = {
            "comgenvid": 200,
            "videofeedback": 200,
            "genvideo": 200,
        }
        expected_supplemental_test_real = {
            "comgenvid": 0,
            "videofeedback": 3080,
            "genvideo": 0,
        }
        expected_generated = {
            "comgenvid": 3400,
            "videofeedback": 3000,
            "genvideo": 5639,
        }
        expected_raw_real = {
            "comgenvid": 1700,
            "videofeedback": 4080,
            "genvideo": 9984,
        }
        for seed in (17, 29, 43):
            for dataset in expected_test_real:
                path = (
                    ROOT
                    / "results/second_order_independent_calibration/splits_complement"
                    / f"seed_{seed}/{dataset}.json"
                )
                payload = json.loads(path.read_text(encoding="utf-8"))
                calibration = {item["video_id"] for item in payload["calibration_real"]}
                test = {item["video_id"] for item in payload["test_real"]}
                self.assertEqual(len(calibration), 200)
                self.assertEqual(len(test), expected_test_real[dataset])
                self.assertEqual(
                    payload["fixed_evaluation_test_real_count"],
                    expected_fixed_test_real[dataset],
                )
                self.assertEqual(
                    payload["reserve_complement_test_real_count"],
                    expected_reserve_test_real[dataset],
                )
                self.assertEqual(
                    payload["historical_calibration_test_real_count"],
                    expected_historical_calibration_test_real[dataset],
                )
                self.assertEqual(
                    payload["supplemental_test_real_count"],
                    expected_supplemental_test_real[dataset],
                )
                self.assertFalse(calibration & test)
                self.assertEqual(payload["overlap_count"], 0)
                self.assertEqual(payload["generated_calibration_count"], 0)
                self.assertEqual(payload["generated_test_count"], expected_generated[dataset])
                self.assertEqual(payload["raw_index_real_count"], expected_raw_real[dataset])
                self.assertEqual(
                    payload["strict_2s_eligible_real_count"],
                    200 + expected_test_real[dataset],
                )

    def test_both_metric_protocols_cover_three_seeds_and_macro(self):
        path = (
            ROOT
            / "results/second_order_independent_calibration"
            / "independent_complement_seed_metrics.csv"
        )
        frame = pd.read_csv(path)
        self.assertEqual(
            set(frame["metric_protocol"]),
            {"paper_pairwise_balanced", "pooled_all_generated"},
        )
        for protocol, group in frame.groupby("metric_protocol"):
            self.assertEqual(set(group["seed"]), {17, 29, 43}, protocol)
            self.assertEqual(
                set(group["dataset"]),
                {"comgenvid", "videofeedback", "genvideo", "Macro-3"},
                protocol,
            )
            self.assertEqual(len(group), 12, protocol)


if __name__ == "__main__":
    unittest.main()
