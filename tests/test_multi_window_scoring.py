from __future__ import annotations

import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import pandas as pd


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from score_multi_window import (
    _decode_spans,
    decode_manifest_row_with_retries,
    stable_shard,
    video_key,
)


class MultiWindowScoringTests(unittest.TestCase):
    def test_decode_spans_splits_large_gaps(self) -> None:
        self.assertEqual(_decode_spans([0, 3, 6, 100, 103], 10), [(0, 6), (100, 103)])

    def test_sharding_is_stable_and_complete(self) -> None:
        row = pd.Series(
            {
                "dataset": "d",
                "protocol_split": "evaluation",
                "subset": "real",
                "source_model": "m",
                "filename": "v.mp4",
            }
        )
        shard = stable_shard(row, 3)
        self.assertIn(shard, range(3))
        self.assertEqual(shard, stable_shard(row.copy(), 3))
        self.assertEqual(video_key(row), ("d", "evaluation", "real", "m", "v.mp4"))

    def test_decode_retry_recovers_from_transient_failure(self) -> None:
        expected = {"frames": "ok"}
        with patch(
            "score_multi_window.decode_manifest_row",
            side_effect=[ValueError("transient"), expected],
        ) as mocked:
            result = decode_manifest_row_with_retries(
                pd.Series(dtype=object),
                "K3_uniform",
                seek_gap=64,
                attempts=3,
                retry_delay=0.0,
            )
        self.assertEqual(result, expected)
        self.assertEqual(mocked.call_count, 2)


if __name__ == "__main__":
    unittest.main()
