"""CAES候选窗口、selector和manifest的纯函数回归测试。"""

from __future__ import annotations

from dataclasses import replace
import inspect
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from config import apply_overrides, load_config
from data.cache_contract import CacheContractContext
from data.coarse_global_cache import (
    coarse_cache_path,
    coarse_positions,
    load_coarse_entry,
    write_coarse_entry,
)
from data.sampling import uniform_windows
from temporal_selection.candidates import (
    generate_candidate_windows,
    uniform_candidate_windows,
)
from temporal_selection.models import WindowManifest
from temporal_selection.selectors import select_windows, temporal_iou
from features import AlphaStallFeatureExtractor
import numpy as np
import torch


class TemporalSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.indices = list(range(40))

    def _scored(self, count: int = 40):
        candidates = generate_candidate_windows(list(range(count)))
        return [
            replace(
                item,
                scores={
                    "feature_change_mean": float(item.candidate_id),
                    "real_anomaly_mean": float(item.candidate_id),
                },
            )
            for item in candidates
        ]

    def test_uniform_selector_exactly_reuses_historical_windows(self) -> None:
        expected = uniform_windows(self.indices, requested_k=3, window_frames=16)
        candidates = uniform_candidate_windows(self.indices, requested_k=3)
        self.assertEqual(
            [list(item.frame_indices) for item in candidates], expected
        )
        manifest = select_windows(
            video_id="demo:v1", duration_seconds=5.0,
            downsample_indices=self.indices, selector_name="uniform",
        )
        selected = {item.candidate_id for item in manifest.selected}
        self.assertEqual(
            [list(item.frame_indices) for item in manifest.candidates if item.candidate_id in selected],
            expected,
        )

    def test_candidate_stride_and_tail_alignment(self) -> None:
        candidates = generate_candidate_windows(list(range(25)))
        self.assertEqual([item.start_position for item in candidates], [0, 4, 8, 9])
        self.assertTrue(all(len(item.frame_indices) == 16 for item in candidates))
        self.assertEqual(len({item.frame_indices for item in candidates}), len(candidates))

    def test_short_video_and_effective_k(self) -> None:
        self.assertEqual(generate_candidate_windows(list(range(15))), [])
        manifest = select_windows(
            video_id="demo:short", duration_seconds=1.875,
            downsample_indices=list(range(15)), selector_name="uniform",
        )
        self.assertEqual(manifest.effective_k, 0)
        one = select_windows(
            video_id="demo:exact", duration_seconds=2.0,
            downsample_indices=list(range(16)), selector_name="random",
        )
        self.assertEqual(one.effective_k, 1)

    def test_duplicate_or_nonmonotonic_dense_frames_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "重复"):
            generate_candidate_windows([0, 1, 1, 2])
        with self.assertRaisesRegex(ValueError, "严格递增"):
            generate_candidate_windows([0, 2, 1, 3])

    def test_random_selector_is_video_deterministic(self) -> None:
        first = select_windows(
            video_id="demo:a", duration_seconds=10.0,
            downsample_indices=list(range(80)), selector_name="random", seed=17,
        )
        repeat = select_windows(
            video_id="demo:a", duration_seconds=10.0,
            downsample_indices=list(range(80)), selector_name="random", seed=17,
        )
        other = select_windows(
            video_id="demo:b", duration_seconds=10.0,
            downsample_indices=list(range(80)), selector_name="random", seed=17,
        )
        self.assertEqual(first.selected, repeat.selected)
        self.assertNotEqual(
            {item.candidate_id for item in first.selected},
            {item.candidate_id for item in other.selected},
        )

    def test_temporal_nms_rejects_overlapping_top_candidate(self) -> None:
        scored = self._scored()
        ranked = sorted(scored, key=lambda item: item.candidate_id, reverse=True)
        self.assertGreater(temporal_iou(ranked[0], ranked[1]), 0.5)
        manifest = select_windows(
            video_id="demo:nms", duration_seconds=5.0,
            downsample_indices=self.indices,
            selector_name="real_anomaly_nms",
            scored_candidates=scored,
        )
        mapping = {item.candidate_id: item for item in scored}
        chosen = [mapping[item.candidate_id] for item in manifest.selected]
        for index, left in enumerate(chosen):
            for right in chosen[index + 1 :]:
                self.assertLessEqual(temporal_iou(left, right), 0.5)

    def test_stratified_selector_covers_three_time_regions(self) -> None:
        scored = self._scored(count=80)
        manifest = select_windows(
            video_id="demo:strata", duration_seconds=10.0,
            downsample_indices=list(range(80)),
            selector_name="stratified_real_anomaly",
            scored_candidates=scored,
        )
        mapping = {item.candidate_id: item for item in scored}
        starts = sorted(mapping[item.candidate_id].start_position for item in manifest.selected)
        self.assertEqual(len(starts), 3)
        self.assertLess(starts[0], starts[1])
        self.assertLess(starts[1], starts[2])

    def test_manifest_round_trip_and_digest(self) -> None:
        manifest = select_windows(
            video_id="demo:roundtrip", duration_seconds=5.0,
            downsample_indices=self.indices, selector_name="uniform",
        )
        restored = WindowManifest.from_dict(manifest.to_dict())
        self.assertEqual(restored, manifest)
        self.assertEqual(restored.digest(), manifest.digest())

    def test_selector_api_has_no_label_or_generator_input(self) -> None:
        parameters = inspect.signature(select_windows).parameters
        self.assertNotIn("subset", parameters)
        self.assertNotIn("label", parameters)
        self.assertNotIn("source_model", parameters)
        with self.assertRaises(TypeError):
            select_windows(
                video_id="demo:v", duration_seconds=5.0,
                downsample_indices=self.indices, selector_name="uniform",
                subset="annotated",
            )

    def test_caes_config_rejects_fps_drift_and_cache_alias(self) -> None:
        config = load_config(ROOT / "configs/benchmark.yaml")
        with self.assertRaisesRegex(ValueError, "coarse_fps"):
            apply_overrides(config, ["temporal_selection.coarse_fps=2"])
        with self.assertRaisesRegex(ValueError, "不得与 Patch cache"):
            apply_overrides(
                config,
                [
                    "runtime.coarse_global_cache_dir=cache/patch_embeddings_k3_2s_8fps"
                ],
            )

    def test_coarse_positions_keep_exact_one_second_intervals(self) -> None:
        self.assertEqual(coarse_positions(list(range(23))), [0, 8, 16])
        self.assertEqual(coarse_positions(list(range(25))), [0, 8, 16, 24])
        with self.assertRaisesRegex(ValueError, "整除"):
            coarse_positions(list(range(25)), base_fps=8, coarse_fps=3)

    def test_coarse_cache_key_includes_dataset_and_split(self) -> None:
        root = Path("cache")
        paths = {
            coarse_cache_path(root, dataset=dataset, split=split, video_id="same")
            for dataset, split in (
                ("a", "calibration"), ("a", "evaluation"), ("b", "evaluation")
            )
        }
        self.assertEqual(len(paths), 3)

    def test_coarse_entry_round_trip_is_float16_and_strict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            root.mkdir()
            source = Path(temporary) / "video.mp4"
            source.write_bytes(b"mock-video")
            context = CacheContractContext(
                root=root,
                policy="strict",
                strict=True,
                contract={"identity": {"cache_kind": "global_embeddings"}},
                contract_sha256="a" * 64,
            )
            path = coarse_cache_path(
                root, dataset="demo", split="evaluation", video_id="demo:v1"
            )
            write_coarse_entry(
                context=context,
                path=path,
                source_video_path=source,
                video_id="demo:v1",
                frame_indices=[0, 8],
                downsample_positions=[0, 8],
                global_features=np.zeros((2, 1024), dtype=np.float32),
            )
            payload = load_coarse_entry(
                context=context,
                path=path,
                source_video_path=source,
                frame_indices=[0, 8],
            )
            self.assertEqual(payload["global"].dtype, torch.float16)
            self.assertEqual(tuple(payload["global"].shape), (2, 1024))

    def test_global_only_extractor_does_not_require_patch_output(self) -> None:
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.marker = torch.nn.Parameter(torch.zeros(()))
                self.head = torch.nn.Identity()

            def forward_features(self, values):
                return {"x_norm_clstoken": torch.ones(len(values), 1024)}

        extractor = AlphaStallFeatureExtractor.__new__(AlphaStallFeatureExtractor)
        extractor.model = FakeModel()
        extractor.transform = lambda _image: torch.zeros(3, 2, 2)
        extractor.device = "cpu"
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        outputs = extractor.frames_to_global_embeddings(
            [np.stack([frame, frame]), np.stack([frame])], batch_size=2
        )
        self.assertEqual([item.shape for item in outputs], [(2, 1024), (1, 1024)])


if __name__ == "__main__":
    unittest.main()
