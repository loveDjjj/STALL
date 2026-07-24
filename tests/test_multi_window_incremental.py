from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from score_multi_window_incremental import prepare_row


class MultiWindowIncrementalTests(unittest.TestCase):
    def test_reuse_requires_exact_video_and_frame_indices(self) -> None:
        windows = [list(range(16)), list(range(8, 24)), list(range(16, 32))]
        row = pd.Series(
            {
                "dataset": "d",
                "protocol_split": "evaluation",
                "subset": "real",
                "source_model": "m",
                "filename": "v.mp4",
                "indices_K3_uniform": str(windows),
            }
        )
        key = (("d", "evaluation", "real", "m", "v.mp4"), tuple(windows[0]))
        full, reused, missing = prepare_row(row, "K3_uniform", {key: {"G_k": 0.5}})
        self.assertEqual(full, windows)
        self.assertEqual(list(reused), [0])
        self.assertEqual(missing, windows[1:])


if __name__ == "__main__":
    unittest.main()
