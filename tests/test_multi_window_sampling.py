from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
SRC = ROOT / "src"
for directory in (SRC, TOOLS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from alpha_stalled.sampling import (
    current_window as canonical_current_window,
    nonoverlap_windows as canonical_nonoverlap_windows,
    parse_indices as canonical_parse_indices,
    uniform_windows as canonical_uniform_windows,
)
from analyze_multi_window_feasibility import (
    apply_window_exclusions,
    current_window,
    nonoverlap_windows,
    parse_indices,
    uniform_windows,
    window_sets,
)


class MultiWindowSamplingTests(unittest.TestCase):
    def test_legacy_analysis_entry_reexports_canonical_sampling(self) -> None:
        self.assertIs(parse_indices, canonical_parse_indices)
        self.assertIs(current_window, canonical_current_window)
        self.assertIs(uniform_windows, canonical_uniform_windows)
        self.assertIs(nonoverlap_windows, canonical_nonoverlap_windows)

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

    def test_invalid_sampling_arguments_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            uniform_windows(list(range(16)), 0)
        with self.assertRaises(ValueError):
            uniform_windows(list(range(16)), 1, window_frames=0)
        with self.assertRaises(ValueError):
            nonoverlap_windows(list(range(16)), window_frames=0)

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
