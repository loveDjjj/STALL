from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_stalled import u0_analysis, u0_scoring


def _row(video_id: str, split: str, effective_k: int = 3) -> dict[str, object]:
    return {
        "video_id": video_id,
        "dataset": "demo",
        "protocol_split": split,
        "subset": "real" if split == "calibration" else "annotated",
        "source_model": "Real" if split == "calibration" else "Gen",
        "filename": f"{video_id}.mp4",
        "video_path": f"datasets/demo/{video_id}.mp4",
        "duration_seconds": 4.0,
        "effective_k": effective_k,
    }


class _Extractor:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def frames_to_global_patch_embeddings(self, batches, batch_size: int):
        self.calls.append((len(batches[0]), batch_size))
        count = len(batches[0])
        return [
            {
                "global": np.zeros((count, 4), dtype=np.float32),
                "patch": np.zeros((count, 2, 4), dtype=np.float32),
                "grid_size": (14, 14),
            }
        ]


class U0PipelineTests(unittest.TestCase):
    def test_decode_row_deduplicates_frames_and_preserves_window_positions(self) -> None:
        row = pd.Series(_row("v", "evaluation", effective_k=2))
        windows = [list(range(16)), list(range(8, 24))]
        frames = np.zeros((24, 2, 2, 3), dtype=np.uint8)
        with (
            mock.patch(
                "alpha_stalled.u0_scoring.resolve_required_video",
                return_value=Path("video.mp4"),
            ) as resolve,
            mock.patch(
                "alpha_stalled.u0_scoring.decode_selected_frames",
                return_value=frames,
            ) as decode,
        ):
            actual = u0_scoring.decode_row(row, windows, seek_gap=64, attempts=1)
        resolve.assert_called_once_with(str(row["video_path"]))
        decode.assert_called_once_with(Path("video.mp4"), list(range(24)), 64)
        self.assertEqual(actual["positions"][0], list(range(16)))
        self.assertEqual(actual["positions"][1], list(range(8, 24)))
        self.assertEqual(actual["unique_indices"], list(range(24)))
        with self.assertRaisesRegex(ValueError, "attempts must be positive"):
            u0_scoring.decode_row(row, windows, seek_gap=64, attempts=0)

    def test_score_batch_keeps_each_video_in_a_separate_extraction_call(self) -> None:
        decoded = []
        for identity in ("a", "b"):
            decoded.append(
                {
                    "row": pd.Series(_row(identity, "evaluation", effective_k=1)),
                    "windows": [list(range(16))],
                    "positions": [list(range(16))],
                    "frames": np.zeros((16, 2, 2, 3), dtype=np.uint8),
                    "unique_indices": list(range(16)),
                }
            )
        extractor = _Extractor()
        global_result = SimpleNamespace(
            spatial=np.array([-1.0, -2.0]),
            temporal_t1=np.array([-3.0, -4.0]),
        )
        local_result = SimpleNamespace(
            patch_spatial=np.array([-5.0, -6.0]),
            patch_temporal=np.array([-7.0, -8.0]),
        )
        with (
            mock.patch(
                "alpha_stalled.u0_scoring.score_global_raw",
                return_value=global_result,
            ),
            mock.patch(
                "alpha_stalled.u0_scoring.score_local_raw",
                return_value=local_result,
            ),
        ):
            rows = u0_scoring.score_batch(
                decoded,
                extractor,
                params={"global_spatial": None, "global_t1": None, "patch_spatial": None, "patch_d2": None},
                score_device="cpu",
                frame_batch_size=8,
            )
        self.assertEqual(extractor.calls, [(16, 8), (16, 8)])
        self.assertEqual([row["video_id"] for row in rows], ["a", "b"])
        self.assertEqual(rows[0]["frame_indices"], json.dumps(list(range(16)), separators=(",", ":")))
        self.assertEqual(rows[1]["patch_d2_raw"], -8.0)
        self.assertEqual(u0_scoring.score_batch([], extractor, {}, "cpu", 8), [])

    def test_two_level_calibration_uses_effective_k_references(self) -> None:
        raw_rows = []
        for identity, split, offset in (
            ("c1", "calibration", 0.0),
            ("c2", "calibration", 0.2),
            ("e1", "evaluation", 0.1),
        ):
            for window_id in range(3):
                raw_rows.append(
                    {
                        **_row(identity, split),
                        "window_id": window_id,
                        "global_spatial_raw": offset + window_id,
                        "global_t1_raw": offset + window_id,
                        "patch_spatial_raw": offset + window_id,
                        "patch_d2_raw": offset + window_id,
                    }
                )
        raw = pd.DataFrame(raw_rows)
        calibration = pd.DataFrame(
            {
                "dataset": ["demo"] * 200,
                "patch_spatial_raw": np.linspace(-1.0, 4.0, 200),
                "patch_d2_raw": np.linspace(-1.0, 4.0, 200),
            }
        )
        with mock.patch(
            "alpha_stalled.u0_analysis.global_references",
            return_value=(np.linspace(-1.0, 4.0, 200), np.linspace(-1.0, 4.0, 200)),
        ):
            windows, local_reference = u0_analysis.calibrate_windows(
                raw, calibration, config={}
            )
        evaluation, video_reference = u0_analysis.aggregate_and_calibrate_videos(windows)
        self.assertEqual(len(local_reference), 400)
        self.assertEqual(len(evaluation), 1)
        self.assertEqual(int(evaluation.iloc[0]["effective_k"]), 3)
        self.assertEqual(int(video_reference.iloc[0]["effective_k"]), 3)
        self.assertEqual(int(video_reference.iloc[0]["calibration_videos"]), 2)
        self.assertAlmostEqual(
            float(evaluation.iloc[0]["S"]),
            0.6 * float(evaluation.iloc[0]["G"]) + 0.4 * float(evaluation.iloc[0]["L"]),
            places=15,
        )

        broken = windows.copy()
        broken.loc[broken["video_id"] == "e1", "effective_k"] = 2
        with self.assertRaisesRegex(ValueError, "effective_k does not match"):
            u0_analysis.aggregate_and_calibrate_videos(broken)


if __name__ == "__main__":
    unittest.main()
