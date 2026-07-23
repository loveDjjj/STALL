from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from build_multi_order_baselines import (
    add_baselines,
    conditional_two_sided_realness,
    d3_statistics,
    empirical_cdf,
    two_sided_realness,
    window_positions,
)
from evaluate_d3_robustness import transformed_windows
from evaluate_d3_pixel_robustness import perturb_frames, select_eval_rows
from prepare_genvideo_d3_exact_protocol import select_planned_rows
from summarize_d3_exact_genvideo_metrics import load_current_reference, load_d3_exact_scores


class MultiOrderBaselineTest(unittest.TestCase):
    def test_window_positions_maps_native_indices(self) -> None:
        downsample = [0, 4, 8, 11, 15, 19]
        self.assertEqual(window_positions(downsample, [8, 11, 15]), [2, 3, 4])

    def test_window_positions_rejects_missing_native_index(self) -> None:
        with self.assertRaises(ValueError):
            window_positions([0, 4, 8], [4, 7])

    def test_empirical_and_two_sided_scores(self) -> None:
        calibration = np.array([1.0, 2.0, 3.0, 4.0])
        values = np.array([0.0, 2.0, 5.0])
        np.testing.assert_allclose(empirical_cdf(values, calibration), [0.0, 0.5, 1.0])
        np.testing.assert_allclose(two_sided_realness(values, calibration), [0.0, 1.0, 0.0])

    def test_conditional_calibration_uses_motion_bins(self) -> None:
        calibration = np.array([1.0, 2.0, 10.0, 20.0])
        motion = np.array([0.0, 0.1, 1.0, 1.1])
        scores, bins = conditional_two_sided_realness(
            np.array([1.5, 15.0]),
            np.array([0.05, 1.05]),
            calibration,
            motion,
            num_bins=2,
        )
        np.testing.assert_array_equal(bins, [0, 1])
        np.testing.assert_allclose(scores, [1.0, 1.0])

    def test_d3_time_normalization_uses_actual_timestamps(self) -> None:
        emb = np.array([[0.0], [1.0], [3.0], [6.0]], dtype=np.float32)
        uniform = d3_statistics(emb, np.array([0.0, 1.0, 2.0, 3.0]))
        stretched = d3_statistics(emb, np.array([0.0, 2.0, 4.0, 6.0]))
        self.assertAlmostEqual(uniform["d3_raw"], stretched["d3_raw"])
        self.assertAlmostEqual(uniform["d3_time_raw"], 4.0 * stretched["d3_time_raw"])

    def test_baseline_weights(self) -> None:
        key = {"subset": "real", "source_model": "real", "filename": "x.mp4"}
        global_d3 = pd.DataFrame(
            [{**key, "global_spatial": 0.8, "global_temporal_t1": 0.4, "global_stall": 0.6,
              "d3_one_sided": 0.2, "d3_two_sided": 0.3, "d3_motion_conditional": 0.5,
              "d3_time_conditional": 0.7}]
        )
        patch = pd.DataFrame(
            [{**key, "patch_spat_percentile": 0.1, "patch_temp_percentile": 0.9}]
        )
        result = add_baselines(global_d3, patch).iloc[0]
        self.assertAlmostEqual(result["B4"], 0.4)
        self.assertAlmostEqual(result["B7"], 0.82)
        self.assertAlmostEqual(result["B8"], 0.688)
        self.assertAlmostEqual(result["G2_two_sided"], 0.575)

    def test_robustness_transforms_are_deterministic_and_shape_stable(self) -> None:
        emb = np.arange(16, dtype=np.float32)[:, None]
        timestamps = np.arange(16, dtype=np.float64) / 8.0
        first = transformed_windows(emb, timestamps, "video-key")
        second = transformed_windows(emb, timestamps, "video-key")
        self.assertEqual(len(first["uniform_4fps_2s"][0]), 8)
        self.assertEqual(len(first["central_8fps_1s"][0]), 8)
        self.assertEqual(len(first["frame_drop_one"][0]), 15)
        self.assertEqual(len(first["frame_duplicate_two"][0]), 16)
        np.testing.assert_array_equal(first["frame_drop_one"][0], second["frame_drop_one"][0])
        self.assertEqual(first["frame_duplicate_two"][0][5, 0], emb[4, 0])

    def test_d3_runlist_only_selects_manifest_csv_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            real_csv = root / "real.csv"
            fake_csv = root / "fake.csv"
            pd.DataFrame({"content_path": ["frames/r1"]}).to_csv(real_csv, index=False)
            pd.DataFrame({"content_path": ["frames/f1", "frames/f2"]}).to_csv(fake_csv, index=False)
            manifest = pd.DataFrame(
                {
                    "csv_path": [str(real_csv), str(fake_csv)],
                    "rows": [1, 2],
                }
            )
            plan = pd.DataFrame(
                {"content_path": ["frames/r1", "frames/f1", "frames/f2", "frames/unused"]}
            )
            selected = select_planned_rows(manifest, plan)
            self.assertEqual(selected["content_path"].tolist(), ["frames/r1", "frames/f1", "frames/f2"])

    def test_d3_summary_allows_missing_comparison_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            missing = Path(temp) / "missing.csv"
            self.assertTrue(load_current_reference(missing, "protocol", "method").empty)

    def test_pixel_perturbations_have_declared_shapes(self) -> None:
        frames = np.full((2, 12, 20, 3), 127, dtype=np.uint8)
        self.assertEqual(perturb_frames(frames, "reference_pixels").shape, frames.shape)
        self.assertEqual(perturb_frames(frames, "jpeg_q30").shape, frames.shape)
        self.assertEqual(perturb_frames(frames, "resize_half").shape, (2, 6, 10, 3))

    def test_pixel_eval_sampling_is_generator_bounded(self) -> None:
        rows = []
        for subset, source, count in (("real", "real", 8), ("annotated", "a", 5), ("annotated", "b", 5)):
            for index in range(count):
                rows.append(
                    {
                        "subset": subset,
                        "source_model": source,
                        "filename": f"{source}-{index}.mp4",
                    }
                )
        index = pd.DataFrame(rows)
        selected = select_eval_rows(index, index, real_count=4, fake_per_generator=2, seed=42)
        counts = selected.groupby(["subset", "source_model"]).size().to_dict()
        self.assertEqual(counts, {("annotated", "a"): 2, ("annotated", "b"): 2, ("real", "real"): 4})

    def test_official_scores_deduplicate_shared_real_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            common = {
                "content_path": "real/r1",
                "label": 0,
                "source_model": "real",
                "filename": "r1.mp4",
                "d3_second_order_std": 1.0,
            }
            for source in ("a", "b"):
                pd.DataFrame(
                    [
                        common,
                        {
                            "content_path": f"fake/{source}",
                            "label": 1,
                            "source_model": source,
                            "filename": f"{source}.mp4",
                            "d3_second_order_std": 2.0,
                        },
                    ]
                ).to_csv(root / f"{source}_head1000_XCLIP-16_l2_scores.csv", index=False)
            scores = load_d3_exact_scores(root, "*_XCLIP-16_l2_scores.csv")
            self.assertEqual(len(scores), 3)
            self.assertEqual(int(scores["label"].eq(0).sum()), 1)


if __name__ == "__main__":
    unittest.main()
