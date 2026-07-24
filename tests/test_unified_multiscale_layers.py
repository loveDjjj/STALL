from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_patch_fast import FastPatchScorer
from analyze_intermediate_layers import merge_layer_windows
from tools.analyze_unified_multiscale import merge_window_scores


class UnifiedMultiscaleLayerTests(unittest.TestCase):
    def test_multiscale_fuses_calibrated_temporal_percentiles(self) -> None:
        keys = {
            "dataset": "demo",
            "protocol_split": "evaluation",
            "subset": "annotated",
            "source_model": "fake",
            "filename": "video.mp4",
            "window_id": 0,
            "frame_indices": "[0,1,2]",
        }
        baseline = pd.DataFrame(
            [
                {
                    **keys,
                    "duration_seconds": 2.0,
                    "effective_k": 1,
                    "G_k": 0.4,
                    "patch_spatial": 0.2,
                    "L_k": 0.5,
                }
            ]
        )
        region = pd.DataFrame(
            [
                {
                    **keys,
                    "region1_temporal": 0.1,
                    "region2_temporal": 0.5,
                    "region3_temporal": 0.9,
                }
            ]
        )
        merged = merge_window_scores(baseline, region)
        self.assertAlmostEqual(float(merged.loc[0, "MS3_temporal"]), 0.3)
        self.assertAlmostEqual(float(merged.loc[0, "MS4_temporal"]), 0.5)
        self.assertAlmostEqual(float(merged.loc[0, "MS3_L_k"]), 0.29)
        self.assertAlmostEqual(float(merged.loc[0, "MS4_L_k"]), 0.47)

    def test_layer23_must_reproduce_frozen_temporal_score(self) -> None:
        keys = {
            "dataset": "demo",
            "protocol_split": "evaluation",
            "subset": "real",
            "source_model": "real",
            "filename": "video.mp4",
            "window_id": 0,
            "frame_indices": "[0,1,2]",
        }
        baseline = pd.DataFrame(
            [
                {
                    **keys,
                    "patch_d2": 0.3,
                    "patch_spatial": 0.4,
                }
            ]
        )
        layers = pd.DataFrame(
            [
                {
                    **keys,
                    "layer11_temporal": 0.1,
                    "layer17_temporal": 0.2,
                    "layer23_temporal": 0.31,
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "layer 23 score differs"):
            merge_layer_windows(baseline, layers)

    def test_temporal_only_scorer_matches_full_temporal_percentile(self) -> None:
        scorer = FastPatchScorer.__new__(FastPatchScorer)
        scorer.device = torch.device("cpu")
        scorer.devices = [scorer.device]
        scorer.patch_grid_size = (2, 2)
        scorer.params_aggregation_region_size = 1
        identity = torch.eye(3)
        zero = torch.zeros(3)
        scorer._param_cache = {
            scorer.device: (zero, identity, zero, identity)
        }
        scorer.calib_spat = np.array([-10.0, 0.0], dtype=np.float32)
        scorer.calib_temp = np.array([-10.0, 0.0], dtype=np.float32)
        patch = np.random.default_rng(4).normal(size=(2, 4, 4, 3)).astype(np.float32)
        kwargs = {
            "patch_temp_mode": "same_grid_second_order",
            "aggregation": "mean",
            "bottomk_ratio": 0.2,
            "temporal_run_length": 3,
            "patch_region_size": 1,
        }
        temporal = scorer.score_temporal_batch(patch, **kwargs)
        full = scorer.score_batch_on_device(
            patch,
            None,
            scorer.device,
            patch_spat_weight=0.1,
            patch_temp_weight=0.9,
            **kwargs,
        )
        np.testing.assert_array_equal(
            temporal["patch_temp_percentile"], full["patch_temp_percentile"]
        )


if __name__ == "__main__":
    unittest.main()
