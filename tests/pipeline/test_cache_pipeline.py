"""严格缓存主链的无 GPU 端到端回归测试。"""

from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import load_config
from data.cache_contract import CacheContractContext
from data.sampling import cached_uniform_frame_indices, uniform_windows
_PIPELINE_SPEC = importlib.util.spec_from_file_location("alpha_stall_pipeline", SRC / "pipeline.py")
assert _PIPELINE_SPEC is not None and _PIPELINE_SPEC.loader is not None
_PIPELINE = importlib.util.module_from_spec(_PIPELINE_SPEC)
sys.modules[_PIPELINE_SPEC.name] = _PIPELINE
_PIPELINE_SPEC.loader.exec_module(_PIPELINE)
run_from_cache = _PIPELINE.run_from_cache
calibrate_and_aggregate = _PIPELINE._calibrate_and_aggregate


class CachePipelineTests(unittest.TestCase):
    """验证 cache -> calibration -> K 窗口 -> video score 的核心闭环。"""

    def _manifest(self, directory: Path, name: str, rows: list[dict]) -> Path:
        path = directory / name
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _rows(self, prefix: str, labels: list[str]) -> list[dict]:
        indices = json.dumps(list(range(48)))
        return [
            {
                "video_path": f"mock/{prefix}_{index}.mp4",
                "subset": label,
                "source_model": "real_source" if label == "real" else "fake_source",
                "downsample_idxs": indices,
            }
            for index, label in enumerate(labels)
        ]

    def test_k3_cache_union_covers_all_k1_to_k3_windows(self) -> None:
        downsample = list(range(64))
        cached = set(cached_uniform_frame_indices(downsample, cache_window_count=3))
        self.assertEqual(len(cached), 48)
        for requested_k in (1, 2, 3):
            for window in uniform_windows(downsample, requested_k=requested_k):
                self.assertTrue(set(window).issubset(cached))

    def test_full_pipeline_uses_calibration_only_and_k3_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            self._manifest(temp, "demo_calibration.csv", self._rows("cal", ["real"] * 4))
            self._manifest(temp, "demo_evaluation.csv", self._rows("eval", ["real", "real", "annotated", "annotated"]))
            config = load_config(ROOT / "configs/benchmark.yaml")
            config["data"] = {
                "datasets": ["demo"],
                "development_manifests": str(temp),
                "external_manifests": str(temp),
                "short_video_policy": "error",
            }
            config["calibration"]["real_videos_per_dataset"] = 4
            config["metrics"]["bootstrap_iterations"] = 4
            config["runtime"] = {
                "cache_dir": "cache/mock",
                "cache_policy": "strict",
                "device": "cpu",
                "max_features_for_fit": 256,
                "minimum_free_gib": 1,
            }

            def payload_for(row: pd.Series) -> dict:
                seed = abs(hash(str(row["video_path"]))) % (2**32)
                rng = np.random.default_rng(seed)
                # 维度故意很小，使测试只验证协议与数据流而非 DINO 数值规模。
                return {
                    "global": __import__("torch").from_numpy(rng.normal(size=(48, 3)).astype(np.float32)),
                    "patch": __import__("torch").from_numpy(rng.normal(size=(48, 4, 3)).astype(np.float32)),
                    "grid_size": [2, 2],
                    "frame_indices": list(range(48)),
                }

            context = CacheContractContext(temp / "cache", "strict", True, {"identity": {}}, "mock")
            with patch("alpha_stall_pipeline.prepare_feature_cache", return_value=context), patch(
                "alpha_stall_pipeline._load_cache_payload", side_effect=lambda _repo, _root, row, _context: payload_for(row)
            ):
                windows, videos, metadata = run_from_cache(ROOT, config)

        self.assertEqual(len(windows), (4 + 4) * 3)
        self.assertEqual(len(videos), 4)
        self.assertEqual(set(videos["subset"]), {"real", "annotated"})
        self.assertTrue(np.isfinite(videos["final_score"]).all())
        self.assertEqual(metadata["datasets"]["demo"]["calibration_videos"], 4)
        self.assertEqual(set(videos["effective_k"]), {3})

    def test_single_branch_ablations_do_not_create_nan_scores(self) -> None:
        base = load_config(ROOT / "configs/benchmark.yaml")
        common = {
            "dataset": "demo",
            "subset": "real",
            "source_model": "real_source",
            "video_path": "mock/cal.mp4",
            "window_id": 0,
        }
        calibration = pd.DataFrame([
            {**common, "video_id": "c1", "global_spatial_raw": 0.1, "global_t1_raw": 0.2, "patch_temporal_raw": 0.3},
            {**common, "video_id": "c2", "global_spatial_raw": 0.4, "global_t1_raw": 0.5, "patch_temporal_raw": 0.6},
        ])
        evaluation = pd.DataFrame([
            {**common, "video_id": "r1", "video_path": "mock/eval_real.mp4", "global_spatial_raw": 0.3, "global_t1_raw": 0.3, "patch_temporal_raw": 0.4},
            {**common, "video_id": "f1", "subset": "annotated", "source_model": "fake_source", "video_path": "mock/eval_fake.mp4", "global_spatial_raw": -0.3, "global_t1_raw": -0.2, "patch_temporal_raw": -0.1},
        ])
        for global_enabled, local_enabled in ((True, False), (False, True)):
            config = __import__("copy").deepcopy(base)
            config["method"]["global"]["enabled"] = global_enabled
            config["method"]["local"]["enabled"] = local_enabled
            if local_enabled:
                config["method"]["local"]["spatial_enabled"] = False
                config["method"]["local"]["temporal_enabled"] = True
            _, videos = calibrate_and_aggregate(calibration, evaluation, config)
            self.assertTrue(np.isfinite(videos["final_score"]).all())

    def test_k1_evaluation_uses_k1_subset_of_k3_calibration_reference(self) -> None:
        config = load_config(ROOT / "configs/benchmark.yaml")
        records = []
        for video_id, offset in (("c1", 0.0), ("c2", 1.0)):
            for window_id in range(3):
                records.append({
                    "video_id": video_id, "dataset": "demo", "subset": "real",
                    "source_model": "real_source", "video_path": f"mock/{video_id}.mp4", "window_id": window_id,
                    "global_spatial_raw": offset + window_id, "global_t1_raw": offset + window_id,
                    "patch_spatial_raw": offset + window_id, "patch_temporal_raw": offset + window_id,
                })
        calibration = pd.DataFrame(records)
        evaluation = pd.DataFrame([
            {**records[0], "video_id": "r1", "video_path": "mock/r1.mp4"},
            *[{**records[index], "video_id": "f1", "subset": "annotated", "source_model": "fake_source", "video_path": "mock/f1.mp4"} for index in range(3)],
        ])
        _, videos = calibrate_and_aggregate(calibration, evaluation, config)
        self.assertEqual(set(videos["effective_k"]), {1, 3})
        self.assertTrue(np.isfinite(videos["final_score"]).all())


if __name__ == "__main__":
    unittest.main()
