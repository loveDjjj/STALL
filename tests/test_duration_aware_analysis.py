import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from analyze_duration_aware_23source import pair_protocol


class DurationAwareAnalysisTest(unittest.TestCase):
    def test_pairwise_real_duration_counts_match_fake(self) -> None:
        rows = []
        for candidate in ("n200", "nmax", "ncustom"):
            for index in range(4):
                duration = 1 if index < 3 else 2
                rows.append(
                    {
                        "candidate": candidate,
                        "dataset": "comgenvid",
                        "subset": "annotated",
                        "source_model": "fake-model",
                        "video_id": f"fake-{index}",
                        "protocol_duration_sec": duration,
                        "G": 0.1,
                        "L": 0.2,
                        "S": 0.14,
                    }
                )
            for index in range(4):
                for duration in (1, 2):
                    rows.append(
                        {
                            "candidate": candidate,
                            "dataset": "comgenvid",
                            "subset": "real",
                            "source_model": "real-model",
                            "video_id": f"real-{index}",
                            "protocol_duration_sec": duration,
                            "G": 0.9,
                            "L": 0.8,
                            "S": 0.86,
                        }
                    )
        pairs = pair_protocol(pd.DataFrame(rows), "full23", seed=42)
        self.assertEqual(len(pairs), 4)
        self.assertEqual(pairs["protocol_duration_sec"].value_counts().to_dict(), {1: 3, 2: 1})
        self.assertEqual(pairs["real_video_id"].nunique(), 4)


if __name__ == "__main__":
    unittest.main()
