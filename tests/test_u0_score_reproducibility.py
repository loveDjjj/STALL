from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_stalled.whitening import empirical_cdf_right_inclusive, stable_sorted


class U0ScoreReproducibilityTests(unittest.TestCase):
    def test_pre_release_temporal_unified_metrics_are_preserved(self) -> None:
        metrics = pd.read_csv(
            ROOT / "results/clean_universal/universal_dataset_metrics.csv"
        )
        u0 = metrics[metrics["config"] == "U0"].set_index("dataset")
        expected = {
            "comgenvid": (0.8986376555671847, 0.9092231422935637),
            "videofeedback": (0.8622572222222222, 0.8684111477215204),
            "genvideo": (0.8566029629970787, 0.8390945185528046),
            "Macro-3": (0.8724992802621618, 0.8722429361892963),
        }
        self.assertEqual(set(u0.index), set(expected))
        for dataset, values in expected.items():
            np.testing.assert_allclose(
                u0.loc[dataset, ["auc", "ap"]].astype(float),
                values,
                atol=1e-15,
                rtol=0.0,
            )

    def test_cdf_is_independent_of_reference_input_order(self) -> None:
        reference = np.array([0.2, 0.1, 0.2, 0.3], dtype=np.float64)
        scores = np.array([0.2, 0.25], dtype=np.float64)
        forward = empirical_cdf_right_inclusive(scores, stable_sorted(reference))
        reverse = empirical_cdf_right_inclusive(
            scores, stable_sorted(reference[::-1])
        )
        np.testing.assert_array_equal(forward, reverse)


if __name__ == "__main__":
    unittest.main()
