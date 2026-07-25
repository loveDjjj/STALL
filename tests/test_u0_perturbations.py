from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from u0_perturbations import (
    CONDITIONS,
    h264_roundtrip,
    insert_scene_cut,
    resize_half_restore,
    temporal_perturbation,
)


class U0PerturbationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frames = np.arange(16 * 12 * 14 * 3, dtype=np.uint32).reshape(16, 12, 14, 3)
        self.frames = (self.frames % 251).astype(np.uint8)

    def test_declared_condition_set_is_complete(self) -> None:
        self.assertEqual(len(CONDITIONS), 10)
        self.assertEqual(len(set(CONDITIONS)), 10)

    def test_temporal_shapes_and_determinism(self) -> None:
        expected = {
            "R0_original": 16,
            "R4_drop10": 14,
            "R5_drop25": 12,
            "R6_repeat10": 16,
            "R7_repeat25": 16,
            "R9_4fps": 8,
        }
        for condition, length in expected.items():
            first = temporal_perturbation(self.frames, condition, "video-1/window-0")
            second = temporal_perturbation(self.frames, condition, "video-1/window-0")
            self.assertEqual(len(first), length)
            np.testing.assert_array_equal(first, second)

    def test_resize_restores_shape_but_changes_pixels(self) -> None:
        result = resize_half_restore(self.frames)
        self.assertEqual(result.shape, self.frames.shape)
        self.assertFalse(np.array_equal(result, self.frames))

    def test_scene_cut_uses_target_then_donor(self) -> None:
        donor = np.full_like(self.frames, 17)
        result = insert_scene_cut(self.frames, donor)
        np.testing.assert_array_equal(result[:8], self.frames[:8])
        np.testing.assert_array_equal(result[8:], donor[8:])

    def test_h264_roundtrip_preserves_shape_and_count(self) -> None:
        result = h264_roundtrip(self.frames, crf=35)
        self.assertEqual(result.shape, self.frames.shape)
        self.assertFalse(np.array_equal(result, self.frames))
        odd = self.frames[:, :11, :13]
        self.assertEqual(h264_roundtrip(odd, crf=23).shape, odd.shape)


if __name__ == "__main__":
    unittest.main()
