from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tools.bootstrap_u0_robustness import BASELINE, CONDITIONS, SCENARIO, wide_scores


class U0RobustnessBootstrapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.scores = self.root / "scores.csv"
        self.release = self.root / "release.csv"
        pd.DataFrame(
            [
                {
                    "video_id": video_id,
                    "dataset": "toy",
                    "subset": subset,
                    "source_model": source,
                }
                for video_id, subset, source in (
                    ("v2", "annotated", "fake"),
                    ("v1", "real", "real-source"),
                )
            ]
        ).to_csv(self.release, index=False)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_scores(self, conditions: tuple[str, ...] = CONDITIONS) -> None:
        rows = []
        for video_id, subset, source in (
            ("v1", "real", "real-source"),
            ("v2", "annotated", "fake"),
        ):
            for index, condition in enumerate(conditions):
                rows.append(
                    {
                        "video_id": video_id,
                        "dataset": "toy",
                        "subset": subset,
                        "source_model": source,
                        "scenario": SCENARIO,
                        "condition": condition,
                        "S": index / 10 + (0.01 if video_id == "v2" else 0.0),
                    }
                )
        pd.DataFrame(rows).to_csv(self.scores, index=False)

    def test_wide_scores_preserves_locked_release_order(self) -> None:
        self._write_scores()
        wide = wide_scores(self.scores, self.release)
        self.assertEqual(wide["video_id"].tolist(), ["v2", "v1"])
        self.assertAlmostEqual(wide.loc[0, BASELINE], 0.01)
        self.assertAlmostEqual(wide.loc[1, CONDITIONS[-1]], 0.9)

    def test_wide_scores_rejects_missing_condition(self) -> None:
        self._write_scores(CONDITIONS[:-1])
        with self.assertRaisesRegex(ValueError, "missing robustness conditions"):
            wide_scores(self.scores, self.release)


if __name__ == "__main__":
    unittest.main()
