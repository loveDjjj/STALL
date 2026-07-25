import unittest

import numpy as np
import pandas as pd

from tools.audit_u0_metric_protocol import binary_metrics, tpr_at_fpr


class U0MetricProtocolTests(unittest.TestCase):
    def test_auc_orientation_is_equivalent(self):
        frame = pd.DataFrame(
            {
                "subset": ["real", "real", "annotated", "annotated"],
                "S": [0.9, 0.7, 0.2, 0.4],
            }
        )
        metrics = binary_metrics(frame)
        self.assertEqual(metrics["auc"], 1.0)
        self.assertEqual(metrics["fake_positive_ap"], 1.0)
        self.assertEqual(metrics["real_positive_ap"], 1.0)
        self.assertEqual(metrics["balanced_accuracy_at_0p5"], 1.0)

    def test_real_and_fake_ap_are_not_assumed_equal(self):
        frame = pd.DataFrame(
            {
                "subset": ["real", "real", "real", "annotated", "annotated"],
                "S": [0.9, 0.8, 0.1, 0.7, 0.2],
            }
        )
        metrics = binary_metrics(frame)
        self.assertNotEqual(metrics["fake_positive_ap"], metrics["real_positive_ap"])

    def test_tpr_at_fpr_uses_only_admissible_operating_points(self):
        label = np.array([0, 0, 1, 1], dtype=np.uint8)
        score = np.array([0.1, 0.2, 0.9, 0.8], dtype=np.float64)
        self.assertEqual(tpr_at_fpr(label, score, 0.01), 1.0)


if __name__ == "__main__":
    unittest.main()
