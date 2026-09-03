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
from temporal_selection.manifest import read_window_manifests, write_window_manifests
from temporal_selection.selectors import select_windows, temporal_iou
from temporal_selection.reference import (
    SelectorReference,
    attach_candidate_scores,
    fit_selector_reference,
    score_coarse_sequence,
)
from temporal_selection.calibration import (
    FrozenLocalD2Reference,
    audit_reconstructed_scores,
    load_frozen_local_reference,
    save_frozen_local_reference,
)
from temporal_selection.dense_scoring import (
    request_feature_arrays,
    score_fixed_windows,
    selected_window_requests,
    union_frame_indices,
)
from math_utils import StableGaussianParams
from features import AlphaStallFeatureExtractor
import numpy as np
import pandas as pd
import torch

from temporal_selection.evaluation import (
    align_selector_scores,
    evaluate_selector_gate,
    paired_selector_bootstrap,
)


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

    def test_window_manifest_jsonl_round_trip_and_duplicate_guard(self) -> None:
        first = select_windows(
            video_id="demo:first", duration_seconds=5.0,
            downsample_indices=self.indices, selector_name="uniform",
        )
        second = select_windows(
            video_id="demo:second", duration_seconds=5.0,
            downsample_indices=self.indices, selector_name="uniform",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "windows.jsonl"
            digest = write_window_manifests(path, [first, second])
            self.assertEqual(len(digest), 64)
            self.assertEqual(read_window_manifests(path), [first, second])
            with self.assertRaisesRegex(ValueError, "重复video_id"):
                write_window_manifests(path, [first, first])

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

    def test_selector_reference_is_real_only_and_zero_diff_is_not_anomaly(self) -> None:
        sequences = [
            np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]], dtype=np.float32),
            np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 2.0]], dtype=np.float32),
        ]
        reference = fit_selector_reference(
            sequences,
            ["real:a", "real:b"],
            coarse_contract_sha256="b" * 64,
            device="cpu",
        )
        self.assertEqual(reference.calibration_ids, ("real:a", "real:b"))
        scored = score_coarse_sequence(
            np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            [0, 8, 16],
            reference,
        )
        self.assertEqual(float(scored["anomaly"][0]), 0.0)
        self.assertTrue(np.isfinite(scored["anomaly"]).all())
        self.assertNotIn("label", inspect.signature(fit_selector_reference).parameters)

    def test_candidate_scores_use_transition_midpoints(self) -> None:
        candidates = generate_candidate_windows(list(range(24)))
        scored = attach_candidate_scores(
            candidates,
            {
                "midpoint_position": np.array([4.0, 12.0, 20.0]),
                "feature_change": np.array([1.0, 3.0, 5.0]),
                "anomaly": np.array([0.1, 0.3, 0.9]),
            },
        )
        self.assertAlmostEqual(scored[0].scores["feature_change_mean"], 2.0)
        self.assertAlmostEqual(scored[0].scores["real_anomaly_max"], 0.3)
        self.assertAlmostEqual(scored[-1].scores["real_anomaly_max"], 0.9)

    def test_selector_reference_rejects_fps_mismatch(self) -> None:
        reference = SelectorReference(
            mean=np.zeros(2),
            whitening=np.eye(2),
            calibration_likelihood=np.array([-1.0, 0.0]),
            calibration_ids=("real:a",),
            coarse_contract_sha256="c" * 64,
            coarse_fps=2,
        )
        with self.assertRaisesRegex(ValueError, "FPS"):
            reference.validate()

    def test_frozen_local_reference_round_trip(self) -> None:
        reference = FrozenLocalD2Reference(
            dataset="demo",
            params=StableGaussianParams(
                mean=np.zeros(1024),
                whitening=np.eye(1024),
                calibration_raw=np.array([-2.0, -1.0]),
                shrinkage=0.0,
            ),
            calibration_ids=("demo:real1", "demo:real2"),
            source_run="c0",
            source_config_hash="d" * 64,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reference.npz"
            file_sha = save_frozen_local_reference(path, reference)
            restored = load_frozen_local_reference(path)
        self.assertEqual(len(file_sha), 64)
        self.assertEqual(restored.digest(), reference.digest())
        np.testing.assert_array_equal(restored.params.whitening, reference.params.whitening)

    def test_reconstructed_score_audit_reports_raw_and_rank_drift(self) -> None:
        source = np.array([-3.0, -2.0, -1.0])
        percentile = np.array([1 / 3, 2 / 3, 1.0])
        audit = audit_reconstructed_scores(
            source + np.array([1e-6, -1e-6, 2e-6]), source, percentile
        )
        self.assertAlmostEqual(audit["raw_max_abs_difference"], 2e-6)
        self.assertEqual(audit["window_cdf_max_rank_steps"], 0.0)
        self.assertGreater(audit["raw_pearson_correlation"], 0.999999)

    def test_dense_requests_deduplicate_windows_across_selectors(self) -> None:
        scored = self._scored(count=40)
        top = select_windows(
            video_id="demo:dense", duration_seconds=5.0,
            downsample_indices=self.indices, selector_name="real_anomaly",
            scored_candidates=scored,
        )
        change = select_windows(
            video_id="demo:dense", duration_seconds=5.0,
            downsample_indices=self.indices, selector_name="feature_change",
            scored_candidates=scored,
        )
        requests = selected_window_requests(
            {"real_anomaly": top, "feature_change": change}
        )
        self.assertEqual(len(requests), 3)
        self.assertEqual({len(item.uses) for item in requests}, {2})
        union = union_frame_indices(requests)
        self.assertLess(len(union), 3 * 16)

    def test_dense_feature_mapping_and_fixed_scoring(self) -> None:
        manifest = select_windows(
            video_id="demo:score", duration_seconds=5.0,
            downsample_indices=self.indices, selector_name="random", seed=17,
        )
        requests = selected_window_requests({"random": manifest})
        union = union_frame_indices(requests)
        rng = np.random.default_rng(53)
        global_features = rng.normal(size=(len(union), 3)).astype(np.float32)
        patch_features = rng.normal(size=(len(union), 4, 3)).astype(np.float32)
        global_windows, patch_windows = request_feature_arrays(
            requests,
            extracted_frame_indices=union,
            global_features=global_features,
            patch_features=patch_features,
        )
        params = StableGaussianParams(
            mean=np.zeros(3),
            whitening=np.eye(3),
            calibration_raw=np.linspace(-100.0, 100.0, 101),
        )
        records = score_fixed_windows(
            requests,
            global_windows=global_windows,
            patch_windows=patch_windows,
            global_parameters={"global_spatial": params, "global_t1": params},
            local_parameters=params,
            device="cpu",
        )
        self.assertEqual(len(records), manifest.effective_k)
        self.assertTrue(
            np.isfinite([item["patch_temporal_raw"] for item in records]).all()
        )
        self.assertTrue(all(len(item["frame_indices"]) == 16 for item in records))

    def test_selector_bootstrap_aligns_identity_and_reports_auc_ap(self) -> None:
        rows = []
        for dataset in ("a", "b", "c"):
            for index in range(8):
                subset = "real" if index < 4 else "annotated"
                source = "real_source" if subset == "real" else "generator"
                truth_score = 0.8 - index * 0.01 if subset == "real" else 0.2 + index * 0.01
                for selector, shift in (("uniform", 0.0), ("adaptive", 0.03)):
                    score = truth_score + (shift if subset == "real" else -shift)
                    rows.append({
                        "video_id": f"{dataset}:{index}",
                        "dataset": dataset,
                        "subset": subset,
                        "source_model": source,
                        "video_path": f"{dataset}/{index}.mp4",
                        "selector": selector,
                        "final_score": score,
                    })
        scores = pd.DataFrame(rows)
        aligned = align_selector_scores(scores)
        self.assertEqual(len(aligned), 24)
        bootstrap = paired_selector_bootstrap(scores, iterations=20)
        self.assertEqual(set(bootstrap["metric"]), {"auc", "ap_real"})
        self.assertIn("Macro-3", set(bootstrap["dataset"]))

        pairwise = []
        for selector, auc, ap in (
            ("uniform", 0.80, 0.79), ("adaptive", 0.81, 0.80)
        ):
            for dataset in ("a", "b", "c"):
                pairwise.append({
                    "selector": selector,
                    "scope": "generator_pairwise_dataset_macro",
                    "dataset": dataset,
                    "auc": auc,
                    "ap_real": ap,
                })
            pairwise.append({
                "selector": selector,
                "scope": "generator_pairwise_macro3",
                "dataset": "Macro-3",
                "auc": auc,
                "ap_real": ap,
            })
        stable_bootstrap = pd.DataFrame([
            {
                "selector": "adaptive", "dataset": "Macro-3",
                "metric": metric, "ci95_low": 0.006, "ci95_high": 0.014,
            }
            for metric in ("auc", "ap_real")
        ])
        gate = evaluate_selector_gate(pd.DataFrame(pairwise), stable_bootstrap)
        self.assertTrue(gate["passes_go_gate"].all())

    def test_selector_alignment_rejects_missing_video(self) -> None:
        frame = pd.DataFrame([
            {"video_id": "v1", "dataset": "d", "subset": "real",
             "source_model": "r", "video_path": "v1.mp4", "selector": "uniform",
             "final_score": 0.5},
            {"video_id": "v1", "dataset": "d", "subset": "real",
             "source_model": "r", "video_path": "v1.mp4", "selector": "adaptive",
             "final_score": 0.5},
            {"video_id": "v2", "dataset": "d", "subset": "annotated",
             "source_model": "g", "video_path": "v2.mp4", "selector": "uniform",
             "final_score": 0.5},
        ])
        with self.assertRaisesRegex(ValueError, "视频身份不一致"):
            align_selector_scores(frame)


if __name__ == "__main__":
    unittest.main()
