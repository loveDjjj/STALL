from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_stalled import backbone
from stall import STALL
from stall_patch import PatchSTALL
import stall


class _TinyDino(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))


class BackboneContractTests(unittest.TestCase):
    def setUp(self) -> None:
        backbone.clear_shared_dinov3_model()
        self.addCleanup(backbone.clear_shared_dinov3_model)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "dinov3"
        self.repo.mkdir()
        self.weights = self.root / "model.pth"
        self.weights.write_bytes(b"checkpoint-a")

    @staticmethod
    def stall_params() -> dict:
        return {
            "W_spat": np.eye(2),
            "mu_spat": np.zeros(2),
            "W_temp": np.eye(2),
            "mu_temp": np.zeros(2),
            "calib_ll_spat": np.zeros((1, 1)),
            "calib_ll_temp": np.zeros((1, 1)),
        }

    def test_stall_reexports_canonical_backbone_functions(self) -> None:
        self.assertIs(stall.create_dinov3_transform, backbone.create_dinov3_transform)
        self.assertIs(stall.resolve_dinov3_paths, backbone.resolve_dinov3_paths)
        self.assertIs(stall.dinov3_model_cache_key, backbone.dinov3_model_cache_key)
        self.assertIs(stall.load_dinov3_model, backbone.load_dinov3_model)
        self.assertEqual(stall.DINO_V3_WEIGHTS, backbone.DINO_V3_WEIGHTS)

    def test_cache_key_binds_paths_device_size_and_mtime(self) -> None:
        first = backbone.dinov3_model_cache_key(
            "cpu", str(self.repo), str(self.weights)
        )
        self.assertEqual(first[0], str(self.repo.resolve()))
        self.assertEqual(first[1], str(self.weights.resolve()))
        self.assertEqual(first[2], "cpu")
        self.weights.write_bytes(b"checkpoint-b-is-longer")
        second = backbone.dinov3_model_cache_key(
            "cpu", str(self.repo), str(self.weights)
        )
        self.assertNotEqual(first, second)

    def test_stall_and_patchstall_share_one_loaded_model(self) -> None:
        model = _TinyDino()
        transform = object()
        with mock.patch(
            "alpha_stalled.backbone.load_dinov3_model",
            return_value=(model, transform),
        ) as load:
            global_scorer = STALL(
                "cpu",
                self.stall_params(),
                dino_repo=str(self.repo),
                dino_weights=str(self.weights),
            )
            patch_scorer = PatchSTALL(
                "cpu",
                data_dict=None,
                dino_repo=str(self.repo),
                dino_weights=str(self.weights),
            )

        self.assertEqual(load.call_count, 1)
        self.assertIs(global_scorer.model, patch_scorer.model)
        self.assertIs(global_scorer.transform, patch_scorer.transform)
        self.assertEqual(STALL._shared_model_key, PatchSTALL._shared_patch_model_key)

    def test_checkpoint_change_reloads_shared_handle(self) -> None:
        second = self.root / "second.pth"
        second.write_bytes(b"checkpoint-b")
        models = [_TinyDino(), _TinyDino()]
        with mock.patch(
            "alpha_stalled.backbone.load_dinov3_model",
            side_effect=[(models[0], object()), (models[1], object())],
        ) as load:
            first = STALL(
                "cpu",
                self.stall_params(),
                dino_repo=str(self.repo),
                dino_weights=str(self.weights),
            )
            changed = PatchSTALL(
                "cpu",
                data_dict=None,
                dino_repo=str(self.repo),
                dino_weights=str(second),
            )

        self.assertEqual(load.call_count, 2)
        self.assertIs(first.model, models[0])
        self.assertIs(changed.model, models[1])

    def test_missing_checkpoint_fails_before_cache_lookup(self) -> None:
        with self.assertRaises(FileNotFoundError):
            backbone.dinov3_model_cache_key(
                "cpu", str(self.repo), str(self.root / "missing.pth")
            )


if __name__ == "__main__":
    unittest.main()
