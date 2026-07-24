from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from analyze_multi_window_feasibility import (
    apply_window_exclusions,
    nonoverlap_windows,
    uniform_windows,
    window_sets,
)


class MultiWindowSamplingTests(unittest.TestCase):
    def test_declared_undecodable_window_is_removed_without_dropping_video(self) -> None:
        row = pd.Series(
            {
                "dataset": "d",
                "protocol_split": "evaluation",
                "subset": "annotated",
                "source_model": "m",
                "video_path": "/tmp/v.mp4",
            }
        )
        sets = {"K3_uniform": [[0] * 16, [1] * 16, [2] * 16]}
        exclusions = {
            ("d", "evaluation", "annotated", "m", "v.mp4", "K3_uniform"): {
                tuple([2] * 16)
            }
        }
        filtered = apply_window_exclusions(row, sets, exclusions)
        self.assertEqual(filtered["K3_uniform"], [[0] * 16, [1] * 16])

    def test_exact_two_seconds_deduplicates_uniform_windows(self) -> None:
        indices = list(range(16))
        self.assertEqual(uniform_windows(indices, 3), [indices])
        self.assertEqual(uniform_windows(indices, 5), [indices])

    def test_uniform_windows_cover_beginning_middle_and_end(self) -> None:
        indices = list(range(40))
        windows = uniform_windows(indices, 3)
        self.assertEqual([window[0] for window in windows], [0, 12, 24])
        self.assertTrue(all(len(window) == 16 for window in windows))

    def test_short_start_range_can_produce_two_unique_windows(self) -> None:
        windows = uniform_windows(list(range(17)), 3)
        self.assertEqual([window[0] for window in windows], [0, 1])

    def test_nonoverlap_excludes_incomplete_tail(self) -> None:
        windows = nonoverlap_windows(list(range(40)))
        self.assertEqual([window[0] for window in windows], [0, 16])
        self.assertTrue(all(len(window) == 16 for window in windows))

    def test_video_shorter_than_two_seconds_is_excluded(self) -> None:
        self.assertEqual(uniform_windows(list(range(15)), 3), [])
        self.assertEqual(nonoverlap_windows(list(range(15))), [])

    def test_window_sets_preserves_numeric_column_name(self) -> None:
        row = pd.Series(
            {
                "downsample_idxs": str(list(range(24))),
                "2_sec_idxs": str(list(range(4, 20))),
            }
        )
        sets = window_sets(row)
        self.assertEqual(sets["K1_current"][0], list(range(4, 20)))
        self.assertEqual(len(sets["K3_uniform"]), 3)


if __name__ == "__main__":
    unittest.main()
