import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import pandas as pd

from analyze_duration_aware_k1_factorial import (
    CONFIGS,
    CONTRASTS,
    bootstrap,
    original_stall_scores,
)


class DurationAwareK1FactorialTest(unittest.TestCase):
    def test_original_stall_is_direct_vatex_component_fusion(self) -> None:
        raw = pd.DataFrame(
            {
                "global_spatial_raw": [1.5, 3.0],
                "global_t1_raw": [15.0, 5.0],
            }
        )
        scored = original_stall_scores(
            raw,
            np.array([1.0, 2.0, 3.0]),
            np.array([10.0, 20.0, 30.0]),
        )
        expected = np.array([0.5 * (1 / 3 + 1 / 3), 0.5 * (1.0 + 0.0)])
        self.assertTrue(np.allclose(scored["K1_STALL"], expected))

    def test_bootstrap_reports_both_ap_orientations(self) -> None:
        groups = []
        for dataset in ("comgenvid", "videofeedback", "genvideo"):
            values = {
                config: np.array([0.9, 0.8, 0.7], dtype=np.float64)
                for config in CONFIGS
            }
            groups.append(
                {
                    "dataset": dataset,
                    "source_model": "generator",
                    "real_ids": np.array(["r0", "r1", "r2"]),
                    "fake_ids": np.array(["f0", "f1", "f2"]),
                    "real": values,
                    "fake": values,
                }
            )
        result = bootstrap(groups, iterations=10, seed=42)
        self.assertEqual(set(result["metric"]), {"ap_real", "ap_fake"})
        self.assertEqual(len(result), len(CONTRASTS) * 2)
        self.assertTrue(np.allclose(result["mean_delta_ap"], 0.0))
        self.assertTrue(np.allclose(result["ci95_low"], 0.0))
        self.assertTrue(np.allclose(result["ci95_high"], 0.0))


if __name__ == "__main__":
    unittest.main()
