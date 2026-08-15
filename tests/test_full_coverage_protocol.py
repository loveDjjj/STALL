import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import analyze_full_coverage_protocol as coverage_cli  # noqa: E402
from alpha_stalled.duration_aware_metrics import (  # noqa: E402
    balanced_pair_frame,
    duration_matched_real,
    metric,
    proportional_allocation,
)


class FullCoverageProtocolTest(unittest.TestCase):
    def test_cli_reexports_shared_metric_objects(self) -> None:
        self.assertIs(coverage_cli.metric, metric)
        self.assertIs(coverage_cli.duration_matched_real, duration_matched_real)
        self.assertIs(coverage_cli.balanced_pair_frame, balanced_pair_frame)

    def test_standardized_ap_uses_fixed_half_prevalence(self) -> None:
        real = np.array([0.9, 0.8])
        fake = np.array([0.95, 0.7, 0.2, 0.1])
        duplicated_fake = np.repeat(fake, 3)
        first = metric(real, fake)
        second = metric(real, duplicated_fake)
        self.assertAlmostEqual(first["auc"], second["auc"])
        self.assertAlmostEqual(first["ap_std50"], second["ap_std50"])
        self.assertAlmostEqual(first["fake_ap_std50"], second["fake_ap_std50"])
        self.assertNotAlmostEqual(first["ap_raw"], second["ap_raw"])

    def test_proportional_allocation_is_exact(self) -> None:
        allocation = proportional_allocation(pd.Series({1: 1, 2: 3}), 10)
        self.assertEqual(allocation, {1: 3, 2: 7})
        self.assertEqual(sum(allocation.values()), 10)

    def test_duration_matching_uses_each_real_once(self) -> None:
        rows = []
        for video_id in ("r0", "r1", "r2", "r3"):
            for duration in (1, 2):
                rows.append(
                    {
                        "video_id": video_id,
                        "source_model": "real_source",
                        "protocol_duration_sec": duration,
                        "G": 0.8,
                        "S": 0.9,
                    }
                )
        real = pd.DataFrame(rows)
        fake = pd.DataFrame(
            {
                "protocol_duration_sec": [1, 2, 2],
                "video_id": ["f0", "f1", "f2"],
            }
        )
        selected = duration_matched_real(real, fake, "test")
        self.assertEqual(len(selected), 4)
        self.assertEqual(selected["video_id"].nunique(), 4)
        self.assertEqual(selected.groupby("protocol_duration_sec").size().to_dict(), {1: 1, 2: 3})

    def test_balanced_pair_selection_respects_real_sources(self) -> None:
        rows = []
        for source in ("a", "b"):
            for index in range(3):
                for duration in (1, 2):
                    rows.append(
                        {
                            "video_id": f"{source}{index}",
                            "source_model": source,
                            "protocol_duration_sec": duration,
                            "G": 0.8,
                            "S": 0.9,
                        }
                    )
        fake = pd.DataFrame(
            {
                "video_id": [f"f{i}" for i in range(8)],
                "protocol_duration_sec": [1, 2] * 4,
                "G": 0.2,
                "S": 0.1,
            }
        )
        selected_real, selected_fake = balanced_pair_frame(pd.DataFrame(rows), fake, "test")
        self.assertEqual(len(selected_real), 6)
        self.assertEqual(len(selected_fake), 6)
        self.assertEqual(selected_real.groupby("source_model").size().to_dict(), {"a": 3, "b": 3})


if __name__ == "__main__":
    unittest.main()
