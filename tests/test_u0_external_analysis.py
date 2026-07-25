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

from analyze_u0_external_genvidbench import paired_bootstrap, selected_reference


class U0ExternalAnalysisTests(unittest.TestCase):
    def test_effective_k_reference_is_one_value_per_real_video(self) -> None:
        frame = pd.DataFrame(
            {
                "video_id": ["a"] * 3 + ["b"] * 3,
                "window_id": [0, 1, 2] * 2,
                "G_k": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        np.testing.assert_allclose(selected_reference(frame, 1, "G_k"), [2.0, 5.0])
        np.testing.assert_allclose(selected_reference(frame, 2, "G_k"), [2.0, 5.0])
        np.testing.assert_allclose(selected_reference(frame, 3, "G_k"), [2.0, 5.0])

    def test_external_bootstrap_resamples_video_rows(self) -> None:
        rows = []
        for index in range(8):
            real = index < 4
            rows.append(
                {
                    "video_id": f"r{index}" if real else f"m{index}",
                    "subset": "real" if real else "annotated",
                    "source_model": "vript" if real else "ms",
                    "original_stall": 0.7 if real else 0.3,
                    "clean_k1": 0.8 if real else 0.2,
                    "locked_u0": 0.9 if real else 0.1,
                }
            )
        for index in range(4):
            rows.append(
                {
                    "video_id": f"p{index}",
                    "subset": "annotated",
                    "source_model": "pika",
                    "original_stall": 0.3,
                    "clean_k1": 0.2,
                    "locked_u0": 0.1,
                }
            )
        samples, summary = paired_bootstrap(pd.DataFrame(rows), iterations=5, seed=7)
        self.assertEqual(len(samples), 20)
        self.assertEqual(len(summary), 4)
        self.assertTrue((summary["mean"] == 0.0).all())


if __name__ == "__main__":
    unittest.main()
