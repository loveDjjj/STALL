import json
import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import score_duration_aware_23source as scoring_cli  # noqa: E402
from alpha_stalled.duration_aware_scoring import (  # noqa: E402
    KEY_COLUMNS,
    WINDOW_KEYS,
    decode_video,
)


def task_frame(effective_k: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "task_id": "task-0",
                "video_id": "video-0",
                "dataset": "fixture",
                "protocol_split": "evaluation",
                "subset": "real",
                "source_model": "real-source",
                "filename": "video.mp4",
                "video_path": "video.mp4",
                "protocol_duration_sec": 1,
                "sampling": "k3_uniform",
                "effective_k": effective_k,
                "frame_indices": json.dumps([[1, 2], [2, 3]]),
            }
        ]
    )


class DurationAwareScoringTest(unittest.TestCase):
    def test_cli_reexports_shared_contract(self) -> None:
        self.assertIs(scoring_cli.KEY_COLUMNS, KEY_COLUMNS)
        self.assertIs(scoring_cli.WINDOW_KEYS, WINDOW_KEYS)
        self.assertIs(scoring_cli.decode_video, decode_video)

    def test_decode_unions_frames_and_preserves_window_positions(self) -> None:
        calls = []

        def decoder(path, indices, seek_gap):
            calls.append((path, indices, seek_gap))
            return [f"frame-{index}" for index in indices]

        result = decode_video(
            task_frame(),
            seek_gap=4,
            attempts=1,
            decoder=decoder,
            resolver=lambda value: f"resolved:{value}",
        )
        self.assertEqual(calls, [("resolved:video.mp4", [1, 2, 3], 4)])
        self.assertEqual(result["unique_indices"], [1, 2, 3])
        self.assertEqual(result["positions"], [[0, 1], [1, 2]])
        self.assertEqual(len(result["tasks"]), 2)

    def test_decode_retries_without_changing_task_identity(self) -> None:
        attempts = []

        def decoder(path, indices, seek_gap):
            attempts.append((path, tuple(indices), seek_gap))
            if len(attempts) == 1:
                raise RuntimeError("transient")
            return list(indices)

        result = decode_video(
            task_frame(),
            seek_gap=1,
            attempts=2,
            retry_delay=0.0,
            decoder=decoder,
            resolver=lambda value: value,
        )
        self.assertEqual(len(attempts), 2)
        self.assertEqual(result["tasks"][0][0]["task_id"], "task-0")

    def test_effective_k_mismatch_fails_before_decode(self) -> None:
        with self.assertRaisesRegex(ValueError, "effective_k mismatch"):
            decode_video(
                task_frame(effective_k=1),
                seek_gap=1,
                attempts=1,
                decoder=lambda *args: [],
                resolver=lambda value: value,
            )


if __name__ == "__main__":
    unittest.main()
