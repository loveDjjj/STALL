from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from u0_injections import CONDITIONS, inject


class U0InjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.RandomState(7)
        self.frames = rng.randint(0, 256, size=(16, 40, 60, 3), dtype=np.uint8)

    def test_all_conditions_preserve_shape(self) -> None:
        donor = np.full_like(self.frames, 19)
        for condition in CONDITIONS:
            result = inject(
                self.frames, condition, donor=donor if condition == "L7_scene_cut" else None
            )
            self.assertEqual(result.frames.shape, self.frames.shape)
            self.assertEqual(result.pixel_mask.shape, (16, 40, 60))
            self.assertEqual(result.patch_mask.shape, (16, 14, 14))
            self.assertEqual(result.temporal_mask.shape, (16,))

    def test_local_masks_cover_exactly_25_percent_of_patch_grid(self) -> None:
        result = inject(self.frames, "L1_local_freeze4")
        self.assertEqual(int(result.patch_mask[6].sum()), 49)
        self.assertEqual(int(result.patch_mask.sum()), 4 * 49)
        self.assertEqual(int(result.pixel_mask[6].sum()), 20 * 30)

    def test_freeze_and_shift_use_declared_sources(self) -> None:
        frozen = inject(self.frames, "L1_local_freeze4").frames
        for frame_index in range(6, 10):
            np.testing.assert_array_equal(
                frozen[frame_index, 10:30, 15:45], self.frames[5, 10:30, 15:45]
            )
        shifted = inject(self.frames, "L5_local_temporal_shift").frames
        np.testing.assert_array_equal(
            shifted[6:10, 10:30, 15:45], self.frames[8:12, 10:30, 15:45]
        )

    def test_scene_cut_is_a_difficult_negative(self) -> None:
        donor = np.full_like(self.frames, 23)
        result = inject(self.frames, "L7_scene_cut", donor=donor)
        self.assertFalse(result.positive_anomaly)
        np.testing.assert_array_equal(result.frames[:8], self.frames[:8])
        np.testing.assert_array_equal(result.frames[8:], donor[8:])


if __name__ == "__main__":
    unittest.main()
