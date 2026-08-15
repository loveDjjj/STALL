from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from alpha_stalled.video_io import (
    decode_all_frames,
    decode_indexed_frames,
    decode_selected_frames,
    decode_spans,
    load_video_frames,
)
import score_multi_window
import stall


class VideoIOTests(unittest.TestCase):
    def test_legacy_exports_are_canonical_objects(self) -> None:
        self.assertIs(score_multi_window.decode_selected_frames, decode_selected_frames)
        self.assertIs(score_multi_window._decode_spans, decode_spans)
        self.assertIs(stall.load_video_frames, load_video_frames)

    def test_decode_spans_preserves_existing_gap_boundary(self) -> None:
        self.assertEqual(decode_spans([], 10), [])
        self.assertEqual(decode_spans([0, 3, 6, 16, 27], 10), [(0, 16), (27, 27)])

    def test_empty_indices_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "^frame_indices is empty$"):
            decode_selected_frames("unused.mp4", [])

    def test_unopenable_video_is_rejected(self) -> None:
        missing = ROOT / "tests" / "does_not_exist.mp4"
        with self.assertRaisesRegex(ValueError, "^cannot open video:"):
            decode_selected_frames(missing, [0])
        with self.assertRaisesRegex(ValueError, "^cannot open video:"):
            decode_all_frames(missing, require_open=True)
        with self.assertRaisesRegex(ValueError, "^cannot open video:"):
            decode_indexed_frames(missing, [0], require_all=True)
        self.assertEqual(load_video_frames(missing).size, 0)
        self.assertEqual(load_video_frames(missing, [0]).size, 0)

    def test_selected_frames_match_sequential_decode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frames.avi"
            size = (32, 24)
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"FFV1"), 8.0, size
            )
            if not writer.isOpened():
                writer.release()
                writer = cv2.VideoWriter(
                    str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8.0, size
                )
            self.assertTrue(writer.isOpened(), "no test video codec is available")
            try:
                for index in range(10):
                    frame = np.empty((size[1], size[0], 3), dtype=np.uint8)
                    frame[..., 0] = index * 17
                    frame[..., 1] = np.arange(size[0], dtype=np.uint8)
                    frame[..., 2] = np.arange(size[1], dtype=np.uint8)[:, None]
                    writer.write(frame)
            finally:
                writer.release()

            capture = cv2.VideoCapture(str(path))
            self.assertTrue(capture.isOpened())
            sequential: list[np.ndarray] = []
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    sequential.append(frame)
            finally:
                capture.release()
            self.assertEqual(len(sequential), 10)

            np.testing.assert_array_equal(
                decode_all_frames(path, require_open=True), np.asarray(sequential)
            )

            indexed = decode_indexed_frames(path, [7, 1, 7, 4], require_all=True)
            expected_indexed = np.stack(
                [sequential[index] for index in (7, 1, 7, 4)]
            )
            np.testing.assert_array_equal(indexed, expected_indexed)

            legacy_partial = decode_indexed_frames(
                path, [1, 20], require_all=False
            )
            np.testing.assert_array_equal(legacy_partial, np.stack([sequential[1]]))
            with self.assertRaisesRegex(
                ValueError, r"^missing 1 requested frames, first=\[20\]$"
            ):
                decode_indexed_frames(path, [1, 20], require_all=True)

            self.assertEqual(decode_indexed_frames(path, []).size, 0)
            with self.assertRaisesRegex(ValueError, "frame indices must be non-negative"):
                decode_indexed_frames(path, [-1], require_all=True)

            actual = decode_selected_frames(path, [7, 1, 7, 4], seek_gap=2)
            expected = np.stack([sequential[index] for index in (1, 4, 7)])
            np.testing.assert_array_equal(actual, expected)

            with self.assertRaisesRegex(
                ValueError, r"^missing 1 decoded frames, first=\[20\]$"
            ):
                decode_selected_frames(path, [1, 20], seek_gap=2)


if __name__ == "__main__":
    unittest.main()
