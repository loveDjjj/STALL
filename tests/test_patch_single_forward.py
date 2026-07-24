from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stall_patch import PatchSTALL


class _FakeDino(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.forward_calls = 0
        self.feature_calls = 0
        self.intermediate_calls = 0
        self.head = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return torch.full((x.shape[0], 4), -1.0, device=x.device)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self.feature_calls += 1
        batch = x.shape[0]
        cls = torch.arange(batch * 4, device=x.device).reshape(batch, 4).float()
        patch = torch.arange(batch * 4 * 4, device=x.device).reshape(batch, 4, 4).float()
        return {"x_norm_clstoken": cls, "x_norm_patchtokens": patch}

    def get_intermediate_layers(self, x: torch.Tensor, n, **kwargs):
        self.intermediate_calls += 1
        batch = x.shape[0]
        return tuple(
            torch.full((batch, 4, 4), float(layer), device=x.device) for layer in n
        )


class PatchSingleForwardTests(unittest.TestCase):
    def test_global_and_patch_tokens_share_one_backbone_forward(self) -> None:
        scorer = PatchSTALL.__new__(PatchSTALL)
        scorer.model = _FakeDino()
        scorer.transform = lambda image: torch.zeros(3, 2, 2)
        frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(3)]

        global_emb, patch_emb, grid = scorer._embed_flat_frames_with_patches(frames, batch_size=2)

        self.assertEqual(scorer.model.forward_calls, 0)
        self.assertEqual(scorer.model.feature_calls, 2)
        self.assertEqual(global_emb.shape, (3, 4))
        self.assertEqual(patch_emb.shape, (3, 4, 4))
        self.assertEqual(grid, (2, 2))

    def test_requested_layers_share_one_forward_per_frame_batch(self) -> None:
        scorer = PatchSTALL.__new__(PatchSTALL)
        scorer.model = _FakeDino()
        scorer.transform = lambda image: torch.zeros(3, 2, 2)
        videos = [
            np.zeros((2, 2, 2, 3), dtype=np.uint8),
            np.zeros((1, 2, 2, 3), dtype=np.uint8),
        ]

        outputs = scorer.frames_to_layer_patch_embeddings(
            videos, layers=(1, 2, 3), batch_size=4
        )

        self.assertEqual(scorer.model.intermediate_calls, 1)
        self.assertEqual(outputs[0]["layers"][1].shape, (2, 4, 4))
        self.assertEqual(outputs[1]["layers"][3].shape, (1, 4, 4))
        np.testing.assert_allclose(outputs[0]["layers"][2], 2.0)
        self.assertEqual(outputs[0]["grid_size"], (2, 2))


if __name__ == "__main__":
    unittest.main()
