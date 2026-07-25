from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.covariance import oas


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from fit_u0_cross_and_oas_params import oas_whitening


class OASWhiteningTests(unittest.TestCase):
    def test_matches_sklearn_float64_oas(self) -> None:
        rng = np.random.RandomState(23)
        samples = rng.normal(size=(211, 7)).astype(np.float64)
        mean, whitening, metadata = oas_whitening(samples, "cpu")
        covariance, shrinkage = oas(samples)
        np.testing.assert_allclose(mean, samples.mean(axis=0), atol=2e-15, rtol=0.0)
        self.assertAlmostEqual(metadata["oas_shrinkage"], shrinkage, places=14)
        precision = whitening @ whitening.T
        expected = np.linalg.inv(covariance + 1e-5 * np.eye(covariance.shape[0]))
        np.testing.assert_allclose(precision, expected, atol=2e-10, rtol=2e-12)


if __name__ == "__main__":
    unittest.main()
