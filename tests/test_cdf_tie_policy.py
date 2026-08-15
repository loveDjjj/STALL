from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_stalled.whitening import empirical_cdf_right_inclusive, stable_sorted


class CdfTiePolicyTests(unittest.TestCase):
    def test_exact_ties_are_included_on_the_right(self) -> None:
        reference = stable_sorted(np.array([2.0, 1.0, 2.0, 3.0]))
        result = empirical_cdf_right_inclusive(
            np.array([0.0, 1.0, 2.0, 2.5, 3.0]), reference
        )
        np.testing.assert_array_equal(result, [0.0, 0.25, 0.75, 0.75, 1.0])

    def test_sort_is_float64_and_deterministic(self) -> None:
        values = np.array([1.0, -0.0, 1.0, 0.0], dtype=np.float32)
        first = stable_sorted(values)
        second = stable_sorted(values[::-1])
        self.assertEqual(first.dtype, np.float64)
        np.testing.assert_array_equal(first, second)

    def test_non_finite_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            stable_sorted(np.array([0.0, np.nan]))
        with self.assertRaises(ValueError):
            empirical_cdf_right_inclusive(
                np.array([np.inf]), np.array([0.0, 1.0])
            )


if __name__ == "__main__":
    unittest.main()
