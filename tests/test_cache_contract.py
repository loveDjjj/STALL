from __future__ import annotations

import subprocess
import sys
import tempfile
import time
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
TOOLS_DIR = ROOT / "tools"
for path in (SRC_DIR, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from alpha_stalled.cache_contract import (
    CONTRACT_FILENAME,
    LegacyFeatureCacheWarning,
    build_feature_cache_contract,
    cache_entry_is_complete,
    prepare_feature_cache,
    validate_cache_entry,
    validate_feature_cache_contract,
    write_cache_entry_metadata,
)
from alpha_stalled.backbone import clear_shared_dinov3_model
from stall import STALL
from create_patch_params import iter_real_patch_cache
from dataset_utils import count_cache_misses, load_csv_with_emb_cache, prefill_emb_cache
from dataset_utils_patch import (
    count_patch_cache_misses,
    load_csv_with_patch_cache,
    prefill_patch_emb_cache,
)
from eval_patch_fast import FastPatchScorer
from verify_feature_cache_contract import verify as verify_feature_cache


class _TinyDino(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)])
        self.embed_dim = 8


class _TinyExtractor:
    def __init__(self, dino: Path, weights: Path) -> None:
        self.model = _TinyDino()
        self.dino_repo_path = str(dino)
        self.dino_weights_path = str(weights)

    def _embed_flat_frames(self, frames, batch_size: int) -> np.ndarray:
        del batch_size
        return np.arange(len(frames) * 8, dtype=np.float32).reshape(len(frames), 8)

    def frames_to_global_patch_embeddings(self, videos, batch_size: int):
        del batch_size
        outputs = []
        for frames in videos:
            length = len(frames)
            outputs.append(
                {
                    "global": np.ones((length, 8), dtype=np.float32),
                    "patch": np.ones((length, 4, 8), dtype=np.float32),
                    "grid_size": [2, 2],
                }
            )
        return outputs


class CacheContractTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_shared_dinov3_model()
        self.addCleanup(clear_shared_dinov3_model)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dino = self.root / "dinov3"
        self.dino.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=self.dino, check=True)
        (self.dino / "model.py").write_text("MODEL = 'tiny'\n", encoding="utf-8")
        subprocess.run(["git", "add", "model.py"], cwd=self.dino, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Cache Contract Tests",
                "-c",
                "user.email=cache-contract@example.invalid",
                "commit",
                "-qm",
                "initial",
            ],
            cwd=self.dino,
            check=True,
        )
        self.weights = self.root / "tiny.pth"
        self.weights.write_bytes(b"tiny checkpoint")
        self.model = SimpleNamespace(
            model=_TinyDino(),
            dino_repo_path=str(self.dino),
            dino_weights_path=str(self.weights),
        )

    def contract(self, *, frame_batch: int = 4) -> dict:
        return build_feature_cache_contract(
            self.model,
            cache_kind="patch_embeddings",
            frame_batch_size=frame_batch,
            video_batch_size=2,
        )

    def index(self) -> tuple[Path, Path]:
        video = self.root / "video.mp4"
        video.write_bytes(b"fake video bytes")
        index = self.root / "index.csv"
        pd.DataFrame(
            [
                {
                    "video_path": str(video),
                    "subset": "real",
                    "source_model": "camera",
                    "downsample_idxs": "[0, 2]",
                    "2_sec_idxs": "[0, 2]",
                }
            ]
        ).to_csv(index, index=False)
        return index, video

    def test_contract_records_encoder_preprocessing_and_batch_identity(self) -> None:
        contract = self.contract()
        digest = validate_feature_cache_contract(contract)
        encoder = contract["identity"]["encoder"]
        extraction = contract["identity"]["extraction"]

        self.assertEqual(len(digest), 64)
        self.assertEqual(encoder["checkpoint_filename"], "tiny.pth")
        self.assertEqual(encoder["output_layer"], 3)
        self.assertEqual(encoder["feature_dimension"], 8)
        self.assertEqual(extraction["frame_batch_size"], 4)
        self.assertIn("backbone.py", extraction["extractor_source_sha256"])
        self.assertIn("video_io.py", extraction["extractor_source_sha256"])
        self.assertIn("stall_patch.py", extraction["extractor_source_sha256"])

    def test_strict_root_is_immutable_and_rejects_different_batching(self) -> None:
        cache_root = self.root / "cache"
        context = prepare_feature_cache(
            cache_root,
            expected_contract=self.contract(),
            policy="strict",
            create=True,
            required_cache_kind="patch_embeddings",
        )

        self.assertTrue(context.strict)
        self.assertTrue((cache_root / CONTRACT_FILENAME).is_file())
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            prepare_feature_cache(
                cache_root,
                expected_contract=self.contract(frame_batch=8),
                policy="strict",
                create=True,
                required_cache_kind="patch_embeddings",
            )
        with self.assertRaisesRegex(ValueError, "legacy policy"):
            prepare_feature_cache(
                cache_root,
                expected_contract=None,
                policy="legacy",
            )

    def test_entry_binds_source_frames_payload_and_cache_content(self) -> None:
        cache_root = self.root / "cache"
        context = prepare_feature_cache(
            cache_root,
            expected_contract=self.contract(),
            policy="strict",
            create=True,
        )
        source = self.root / "video.mp4"
        source.write_bytes(b"source video")
        cache = cache_root / "real/camera/video_2s.pt"
        cache.parent.mkdir(parents=True)
        cache.write_bytes(b"cache bytes")
        payload = {"format": "test", "shape": [4, 8], "dtype": "torch.float32"}

        written = write_cache_entry_metadata(
            context,
            cache_path=cache,
            source_video_path=source,
            frame_indices=[0, 2, 4, 6],
            payload=payload,
        )
        validated = validate_cache_entry(
            context,
            cache_path=cache,
            source_video_path=source,
            frame_indices=[0, 2, 4, 6],
            payload=payload,
            verify_cache_sha256=True,
        )

        self.assertEqual(validated.cache_sha256, written.cache_sha256)
        self.assertTrue(cache_entry_is_complete(cache, strict=True))
        audit = verify_feature_cache(
            cache_root,
            repository_root=self.root,
            verify_cache_hashes=True,
        )
        self.assertTrue(audit["strict_evidence"])
        self.assertTrue(audit["full_entry_coverage"])
        with self.assertRaisesRegex(ValueError, "frame indices differ"):
            validate_cache_entry(
                context,
                cache_path=cache,
                source_video_path=source,
                frame_indices=[0, 2, 4, 7],
            )
        cache.write_bytes(b"CACHE BYTES")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validate_cache_entry(
                context,
                cache_path=cache,
                source_video_path=source,
                frame_indices=[0, 2, 4, 6],
                verify_cache_sha256=True,
            )

    def test_source_change_is_detected_but_timestamp_only_change_is_accepted(self) -> None:
        cache_root = self.root / "cache"
        context = prepare_feature_cache(
            cache_root,
            expected_contract=self.contract(),
            policy="strict",
            create=True,
        )
        source = self.root / "video.mp4"
        source.write_bytes(b"same content")
        cache = cache_root / "real/camera/video.pt"
        cache.parent.mkdir(parents=True)
        cache.write_bytes(b"cache")
        write_cache_entry_metadata(
            context,
            cache_path=cache,
            source_video_path=source,
            frame_indices=[1, 3],
            payload={"format": "test"},
        )

        time.sleep(0.002)
        source.touch()
        validate_cache_entry(
            context,
            cache_path=cache,
            source_video_path=source,
            frame_indices=[1, 3],
        )
        source.write_bytes(b"DIFFERENT!!!")
        with self.assertRaisesRegex(ValueError, "source video differs"):
            validate_cache_entry(
                context,
                cache_path=cache,
                source_video_path=source,
                frame_indices=[1, 3],
            )

    def test_auto_mode_warns_for_legacy_and_strict_refuses_to_bless_it(self) -> None:
        legacy = self.root / "legacy"
        legacy.mkdir()
        (legacy / "old.pt").write_bytes(b"legacy")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            context = prepare_feature_cache(
                legacy,
                expected_contract=None,
                policy="auto",
            )
        self.assertFalse(context.strict)
        self.assertTrue(any(item.category is LegacyFeatureCacheWarning for item in caught))
        audit = verify_feature_cache(
            legacy,
            repository_root=self.root,
            allow_legacy=True,
        )
        self.assertFalse(audit["strict_evidence"])
        with self.assertRaisesRegex(ValueError, "legacy cache"):
            verify_feature_cache(legacy, repository_root=self.root)
        with self.assertRaisesRegex(ValueError, "refuses to bless"):
            prepare_feature_cache(
                legacy,
                expected_contract=self.contract(),
                policy="strict",
                create=True,
            )

    def test_stall_shared_model_cache_is_keyed_by_checkpoint(self) -> None:
        second_weights = self.root / "other.pth"
        second_weights.write_bytes(b"other checkpoint")
        data = {
            "W_spat": np.eye(2),
            "mu_spat": np.zeros(2),
            "W_temp": np.eye(2),
            "mu_temp": np.zeros(2),
            "calib_ll_spat": np.zeros((1, 1)),
            "calib_ll_temp": np.zeros((1, 1)),
        }
        previous = (STALL._shared_model, STALL._shared_transform, STALL._shared_model_key)
        self.addCleanup(setattr, STALL, "_shared_model", previous[0])
        self.addCleanup(setattr, STALL, "_shared_transform", previous[1])
        self.addCleanup(setattr, STALL, "_shared_model_key", previous[2])
        STALL._shared_model = None
        STALL._shared_transform = None
        STALL._shared_model_key = None
        fake_model = _TinyDino()

        with mock.patch(
            "alpha_stalled.backbone.load_dinov3_model",
            return_value=(fake_model, object()),
        ) as load:
            STALL("cpu", data, dino_repo=str(self.dino), dino_weights=str(self.weights))
            STALL("cpu", data, dino_repo=str(self.dino), dino_weights=str(self.weights))
            STALL("cpu", data, dino_repo=str(self.dino), dino_weights=str(second_weights))

        self.assertEqual(load.call_count, 2)

    def test_patch_prefill_and_reader_enforce_contract_and_frame_identity(self) -> None:
        index, _ = self.index()
        cache_root = self.root / "patch_cache"
        extractor = _TinyExtractor(self.dino, self.weights)
        frames = np.zeros((2, 2, 2, 3), dtype=np.uint8)

        self.assertEqual(
            count_patch_cache_misses(
                str(index),
                str(cache_root),
                duration_sec=2,
                compact=True,
                cache_policy="strict",
            ),
            1,
        )
        with mock.patch("dataset_utils_patch.decode_indexed_frames", return_value=frames):
            written = list(
                prefill_patch_emb_cache(
                    str(index),
                    str(cache_root),
                    model=extractor,
                    batch_size=2,
                    duration_sec=2,
                    compact=True,
                    video_batch=1,
                    cache_policy="strict",
                )
            )
        self.assertEqual(len(written), 1)
        self.assertEqual(
            count_patch_cache_misses(
                str(index),
                str(cache_root),
                duration_sec=2,
                compact=True,
                cache_policy="strict",
            ),
            0,
        )
        rows = list(
            load_csv_with_patch_cache(
                str(index),
                str(cache_root),
                model=extractor,
                batch_size=2,
                duration_sec=2,
                compact=True,
                video_batch=1,
                cache_policy="strict",
            )
        )
        self.assertEqual(rows[0]["patch"].shape, (2, 4, 8))
        params_rows = list(
            iter_real_patch_cache(
                str(index),
                str(cache_root),
                duration_sec=2,
                compact=True,
                cache_policy="strict",
            )
        )
        self.assertEqual(len(params_rows), 1)
        self.assertEqual(params_rows[0][1]["patch"].shape, (2, 4, 8))

        changed = pd.read_csv(index)
        changed["2_sec_idxs"] = "[0, 3]"
        changed.to_csv(index, index=False)
        with self.assertRaisesRegex(ValueError, "frame indices differ"):
            list(
                load_csv_with_patch_cache(
                    str(index),
                    str(cache_root),
                    model=extractor,
                    batch_size=2,
                    duration_sec=2,
                    compact=True,
                    video_batch=1,
                    cache_policy="strict",
                )
            )

    def test_fast_scorer_requires_params_and_cache_contract_to_match(self) -> None:
        strict_root = self.root / "strict_cache"
        strict_context = prepare_feature_cache(
            strict_root,
            expected_contract=self.contract(),
            policy="strict",
            create=True,
            required_cache_kind="patch_embeddings",
        )
        legacy_root = self.root / "legacy_cache"
        legacy_root.mkdir()
        (legacy_root / "old.pt").write_bytes(b"legacy")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LegacyFeatureCacheWarning)
            legacy_context = prepare_feature_cache(
                legacy_root,
                expected_contract=None,
                policy="auto",
                required_cache_kind="patch_embeddings",
            )

        scorer = FastPatchScorer.__new__(FastPatchScorer)
        scorer.feature_cache_contract_sha256 = strict_context.contract_sha256
        scorer.validate_cache_contract(strict_context)
        with self.assertRaisesRegex(ValueError, "legacy feature cache"):
            scorer.validate_cache_contract(legacy_context)

        scorer.feature_cache_contract_sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            scorer.validate_cache_contract(strict_context)

        scorer.feature_cache_contract_sha256 = "legacy_uncontracted"
        scorer.validate_cache_contract(legacy_context)
        with self.assertRaisesRegex(ValueError, "legacy/unbound patch params"):
            scorer.validate_cache_contract(strict_context)

    def test_global_prefill_and_reader_write_contract_metadata(self) -> None:
        index, _ = self.index()
        cache_root = self.root / "global_cache"
        extractor = _TinyExtractor(self.dino, self.weights)
        frames = np.zeros((2, 2, 2, 3), dtype=np.uint8)

        with mock.patch("dataset_utils.decode_indexed_frames", return_value=frames):
            written = list(
                prefill_emb_cache(
                    str(index),
                    str(cache_root),
                    model=extractor,
                    batch_size=2,
                    duration_sec=2,
                    compact=True,
                    video_batch=1,
                    cache_policy="strict",
                )
            )
        self.assertEqual(len(written), 1)
        self.assertEqual(
            count_cache_misses(
                str(index),
                str(cache_root),
                duration_sec=2,
                compact=True,
                cache_policy="strict",
            ),
            0,
        )
        rows = list(
            load_csv_with_emb_cache(
                str(index),
                str(cache_root),
                model=extractor,
                batch_size=2,
                duration_sec=2,
                compact=True,
                video_batch=1,
                cache_policy="strict",
            )
        )
        self.assertEqual(rows[0]["embs"].shape, (1, 2, 8))
        self.assertTrue((cache_root / CONTRACT_FILENAME).is_file())


if __name__ == "__main__":
    unittest.main()
