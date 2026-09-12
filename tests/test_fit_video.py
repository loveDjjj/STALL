"""原视频拟合位置身份、缓存无关性、有界预取和断点恢复。"""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import load_paper_config
from reference import file_digest
from reference_fit import prepare_fit_inputs, prepare_fit_assets, sample_d2_positions
from workflow import video_file_states
import features
import data.video
import data.prefetch


def test_sampling_preserves_legacy_path():
    values = torch.arange(400)
    identity = "datasets/example/real/clip.mp4"
    seed = int.from_bytes(hashlib.sha256(("17:" + identity).encode()).digest()[:8], "little")
    expected = np.random.default_rng(seed).choice(400, size=256, replace=False)
    np.testing.assert_array_equal(sample_d2_positions(values, identity).numpy(), expected)
    with pytest.raises(ValueError):
        sample_d2_positions(values[:200], identity)


@pytest.mark.parametrize("row_indices,numbers", [(None, [1, 2]), ([2, 4], [2, 4])])
def test_video_assets_resume_and_keep_legacy_identity(tmp_path, monkeypatch, row_indices, numbers):
    weights = tmp_path / "weights"
    weights.write_bytes(b"weights")
    config = load_paper_config(ROOT / "configs/paper.yaml")
    config["encoder"]["weights"] = str(weights)
    config["encoder"]["weights_sha256"] = file_digest(weights)
    config["runtime"]["device"] = "cpu"
    config["runtime"]["decode_workers"] = 2
    rows = []
    for i in range(2):
        path = tmp_path / f"{i}.mp4"
        path.write_bytes(b"video")
        rows.append(
            dict(
                video_id=f"v{i}",
                video_path=str(path),
                legacy_path=f"original/{i}.mp4",
                downsample_idxs=json.dumps(list(range(32))),
            )
        )
    frame = pd.DataFrame(rows)
    calls = []
    fail = [True]

    def arrays():
        t = np.arange(32, dtype=np.float32)
        # 让不同时间位置D2不同，能够检测采样位置或legacy_path变化。
        return {
            "global": np.broadcast_to(t[:, None], (32, 1024)).copy(),
            "patch": np.broadcast_to(t[:, None, None, None] ** 3, (32, 4, 4, 1024)).copy(),
        }

    class Model:
        def __init__(self, *args, **kwargs):
            calls.append("model")

        def prepare_frames(self, frames):
            return frames

        def frames_to_global_patch_embeddings(self, frames, batch_size):
            assert batch_size == 8
            calls.append("forward")
            if calls.count("forward") == 2 and fail[0]:
                raise RuntimeError("模拟前向中断")
            return [arrays()]

    monkeypatch.setattr(features, "AlphaStallFeatureExtractor", Model)
    monkeypatch.setattr(data.video, "decode_bounded", lambda path, indices: list(indices))
    monkeypatch.setattr(data.prefetch, "frame_reservation", lambda path, count: 1)
    states = video_file_states(tmp_path, frame)
    with pytest.raises(RuntimeError, match="中断"):
        prepare_fit_assets(
            tmp_path, config, frame, states, tmp_path, "run", row_indices=row_indices
        )
    assert (tmp_path / f"fit_features/{numbers[0]:06d}.json").exists() and not (
        tmp_path / f"fit_features/{numbers[1]:06d}.json"
    ).exists()
    fail[0] = False
    result = prepare_fit_assets(
        tmp_path, config, frame, states, tmp_path, "run", row_indices=row_indices
    )
    before = len(calls)
    again = prepare_fit_assets(
        tmp_path, config, frame, states, tmp_path, "run", row_indices=row_indices
    )
    assert len(calls) == before and calls.count("forward") == 3
    pd.testing.assert_frame_equal(result, again)
    for row in result.itertuples():
        with np.load(row.feature_asset) as z:
            assert z["raw_d2"].shape == (256, 1024) and z["global_windows"].shape == (3, 16, 1024)
            assert str(z["sampling_identity"]) == row.legacy_path
            source = arrays()["patch"]
            patch = torch.from_numpy(np.stack([source[idx] for idx in z["frame_indices"]]))
            raw = (patch[:, 2:] - 2.0 * patch[:, 1:-1] + patch[:, :-2]).reshape(-1, 1024)
            np.testing.assert_array_equal(
                z["raw_d2"], sample_d2_positions(raw, row.legacy_path).numpy()
            )
    (tmp_path / f"fit_features/{numbers[0]:06d}.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="损坏"):
        prepare_fit_assets(
            tmp_path, config, frame, states, tmp_path, "run", row_indices=row_indices
        )


def test_legacy_configs_keep_read_compatibility():
    from config import validate_paper_config

    config = load_paper_config(ROOT / "configs/paper.yaml")
    config.pop("fit")
    for field in (
        "decode_workers",
        "prefetch_depth",
        "prefetch_memory_mb",
        "devices",
        "prefetch_enabled",
    ):
        config["runtime"].pop(field)
    validate_paper_config(config)
