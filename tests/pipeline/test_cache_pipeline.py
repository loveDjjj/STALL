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
import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import load_config
from data.cache_contract import CacheContractContext
from data.packed_cache import FORMAT, PackedCacheReader, packed_root, write_index
from data.sampling import cached_uniform_frame_indices, uniform_windows
from math_utils import StableGaussianParams, score_gaussian_aggregate_float64
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

    def test_packed_reader_exposes_stable_entry_location(self) -> None:
        """顺序 shard 调度必须能先定位条目、后只读取一次对应 shard。"""

        with tempfile.TemporaryDirectory() as temporary:
            cache_root = Path(temporary)
            shard_name = "development/demo/evaluation/shard-00000.pt"
            root = packed_root(cache_root)
            target = root / shard_name
            target.parent.mkdir(parents=True)
            torch.save(
                {
                    "format": FORMAT,
                    "entries": [
                        {"cache_key": "real/source/demo.pt", "payload": {"value": torch.tensor(1)}}
                    ],
                },
                target,
            )
            write_index(
                cache_root,
                {
                    "format": FORMAT,
                    "entries": {"real/source/demo.pt": {"shard": shard_name, "position": 0}},
                    "shards": {shard_name: {"entries": 1}},
                },
            )
            reader = PackedCacheReader(cache_root, max_shards=1)
            self.assertEqual(reader.location("real/source/demo.pt"), (shard_name, 0))
            self.assertEqual(int(reader.get("real/source/demo.pt")["payload"]["value"]), 1)

    def test_packed_shard_split_keeps_each_shard_in_one_worker(self) -> None:
        """双卡任务边界不得切开一个物理 shard。"""

        with tempfile.TemporaryDirectory() as temporary:
            cache_root = Path(temporary)
            root = packed_root(cache_root)
            entries = {}
            shards = {}
            for shard_index, names in enumerate((("a", "b"), ("c",))):
                shard_name = f"development/demo/evaluation/shard-{shard_index:05d}.pt"
                target = root / shard_name
                target.parent.mkdir(parents=True, exist_ok=True)
                shard_entries = []
                for position, name in enumerate(names):
                    key = f"real/source/{name}.pt"
                    entries[key] = {"shard": shard_name, "position": position}
                    shard_entries.append({"cache_key": key, "payload": {}})
                torch.save({"format": FORMAT, "entries": shard_entries}, target)
                shards[shard_name] = {"entries": len(shard_entries)}
            write_index(cache_root, {"format": FORMAT, "entries": entries, "shards": shards})
            rows = pd.DataFrame([
                {"video_path": f"mock/{name}.mp4", "subset": "real", "source_model": "source"}
                for name in ("c", "a", "b")
            ])
            chunks = _PIPELINE._split_rows_by_packed_shard(rows, cache_root, workers=2)
            self.assertEqual([chunk.index.tolist() for chunk in chunks], [[1, 2], [0]])

    def test_batched_float64_scoring_matches_individual_windows(self) -> None:
        rng = np.random.default_rng(7)
        features = rng.normal(size=(4, 3, 5)).astype(np.float32)
        params = StableGaussianParams(
            mean=rng.normal(size=5), whitening=np.eye(5), calibration_raw=np.array([0.0, 1.0]),
        )
        batched, _ = score_gaussian_aggregate_float64(features, params, "mean", device="cpu", compute_percentile=False)
        individual = np.array([
            score_gaussian_aggregate_float64(item[None], params, "mean", device="cpu", compute_percentile=False)[0][0]
            for item in features
        ])
        np.testing.assert_allclose(batched, individual, rtol=0.0, atol=1e-12)

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
            # 该测试故意使用 3 维伪特征，只验证 Local 数据流；官方 VATEX
            # Global 参数固定为 1024 维，另由专门的一致性测试覆盖。
            config["method"]["global"]["enabled"] = False
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
                "alpha_stall_pipeline._load_cache_payload", side_effect=lambda _repo, _root, row, _context, _reader=None: payload_for(row)
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
        # Global 的窗口 CDF 已由官方 VATEX 参数在评分阶段生成；这里直接提供
        # 已校准列，以便本测试只覆盖视频级聚合的分支开关。
        for frame in (calibration, evaluation):
            frame["global_spatial"] = frame["global_spatial_raw"]
            frame["global_t1"] = frame["global_t1_raw"]
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
        for frame in (calibration, evaluation):
            frame["global_spatial"] = frame["global_spatial_raw"]
            frame["global_t1"] = frame["global_t1_raw"]
        _, videos = calibrate_and_aggregate(calibration, evaluation, config)
        self.assertEqual(set(videos["effective_k"]), {1, 3})
        self.assertTrue(np.isfinite(videos["final_score"]).all())


if __name__ == "__main__":
    unittest.main()
