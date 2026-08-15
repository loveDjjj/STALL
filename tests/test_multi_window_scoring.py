from __future__ import annotations

import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import pandas as pd
import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from score_multi_window import (
    KEY_COLUMNS,
    _decode_spans,
    decode_manifest_row_with_retries,
    stable_shard,
    video_key,
)
import score_multi_window
from alpha_stalled import legacy_window_scoring


class MultiWindowScoringTests(unittest.TestCase):
    def test_cli_reexports_shared_historical_protocol(self) -> None:
        self.assertIs(score_multi_window.KEY_COLUMNS, legacy_window_scoring.KEY_COLUMNS)
        self.assertIs(score_multi_window.LOCAL_PARAMS, legacy_window_scoring.LOCAL_PARAMS)
        self.assertIs(score_multi_window.SAMPLINGS, legacy_window_scoring.SAMPLINGS)
        self.assertIs(score_multi_window.stable_shard, legacy_window_scoring.stable_shard)
        self.assertIs(score_multi_window.video_key, legacy_window_scoring.video_key)
        self.assertIs(score_multi_window.load_windows, legacy_window_scoring.load_windows)
        self.assertIs(score_multi_window.load_completed, legacy_window_scoring.load_completed)
        self.assertIs(score_multi_window.score_batch, legacy_window_scoring.score_batch)

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

    def test_window_parser_rejects_wrong_length_and_duplicates(self) -> None:
        row = pd.Series(
            {
                **{column: "value" for column in KEY_COLUMNS},
                "indices_K3_uniform": "[[0,1,2]]",
            }
        )
        with self.assertRaisesRegex(ValueError, "invalid K3_uniform windows"):
            legacy_window_scoring.load_windows(row, "K3_uniform")
        window = list(range(16))
        row["indices_K3_uniform"] = str([window, window]).replace(" ", "")
        with self.assertRaisesRegex(ValueError, "duplicate K3_uniform windows"):
            legacy_window_scoring.load_windows(row, "K3_uniform")

    def test_shared_retry_accepts_explicit_decoder(self) -> None:
        calls = []

        def decoder(row, sampling, seek_gap):
            calls.append((sampling, seek_gap))
            if len(calls) == 1:
                raise ValueError("transient")
            return {"frames": "ok"}

        result = legacy_window_scoring.decode_manifest_row_with_retries(
            pd.Series(dtype=object),
            "K3_uniform",
            seek_gap=64,
            attempts=2,
            retry_delay=0.0,
            decoder=decoder,
        )
        self.assertEqual(result, {"frames": "ok"})
        self.assertEqual(calls, [("K3_uniform", 64), ("K3_uniform", 64)])

    def test_historical_score_batch_preserves_columns_and_fusion(self) -> None:
        source = pd.Series(
            {
                "dataset": "d",
                "protocol_split": "evaluation",
                "subset": "real",
                "source_model": "m",
                "filename": "v.mp4",
                "video_path": "videos/v.mp4",
                "duration_seconds": 2.0,
                "active_sampling": "K1_current",
            }
        )
        decoded = [
            {
                "row": source,
                "frames": [object()] * 16,
                "windows": [list(range(16))],
                "window_positions": [list(range(16))],
                "unique_indices": list(range(16)),
            }
        ]

        class Extractor:
            def frames_to_global_patch_embeddings(self, frames, batch_size):
                self.batch_size = batch_size
                return [
                    {
                        "global": np.zeros((16, 2), dtype=np.float32),
                        "patch": np.zeros((16, 3, 2), dtype=np.float32),
                    }
                ]

        class GlobalScorer:
            def _scores_from_embs(self, values):
                self.shape = values.shape
                return {
                    "spat_percentile": np.array([0.2]),
                    "temp_percentile": np.array([0.4]),
                    "final_score": np.array([0.3]),
                }

        class LocalScorer:
            aggregation_config = {"mode": "mean"}
            params_bottomk_ratio = 0.2
            params_temporal_run_length = 1
            params_patch_region_size = 1

            def score_batch(self, values, **kwargs):
                self.shape = values.shape
                self.kwargs = kwargs
                return {
                    "patch_spat_percentile": np.array([0.5]),
                    "patch_temp_percentile": np.array([0.7]),
                    "patch_final_score": np.array([0.68]),
                }

        extractor = Extractor()
        global_scorer = GlobalScorer()
        local_scorer = LocalScorer()
        rows = legacy_window_scoring.score_batch(
            decoded, extractor, global_scorer, local_scorer, 8
        )
        self.assertEqual(extractor.batch_size, 8)
        self.assertEqual(global_scorer.shape, (1, 16, 2))
        self.assertEqual(local_scorer.shape, (1, 16, 3, 2))
        self.assertEqual(local_scorer.kwargs["patch_temp_mode"], "same_grid_second_order")
        self.assertEqual(rows[0]["frame_indices"], "[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]")
        self.assertAlmostEqual(rows[0]["S_k"], 0.6 * 0.3 + 0.4 * 0.68)


if __name__ == "__main__":
    unittest.main()
