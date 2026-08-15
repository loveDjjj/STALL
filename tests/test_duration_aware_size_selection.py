import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from select_duration_aware_calibration_size import quantile_mse


class DurationAwareSizeSelectionTest(unittest.TestCase):
    def test_uniform_midpoints_have_zero_quantile_error(self) -> None:
        values = (np.arange(10) + 0.5) / 10
        self.assertAlmostEqual(quantile_mse(values), 0.0)

    def test_concentrated_percentiles_have_larger_error(self) -> None:
        uniform = (np.arange(20) + 0.5) / 20
        concentrated = np.repeat(0.5, 20)
        self.assertGreater(quantile_mse(concentrated), quantile_mse(uniform))


if __name__ == "__main__":
    unittest.main()
