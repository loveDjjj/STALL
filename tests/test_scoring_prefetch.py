"""使用确定性伪编码器核对同步/两阶段预取的选窗、前向角色和原始评分。"""

import json
from pathlib import Path
import sys
import threading

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from math_utils import StableGaussianParams, GaussianMeanCandidateScorerFloat64
from reference import ReferenceBundle, SCHEMA, file_digest
from reference_fit import score_cdf_checkpointed
import workflow


def test_scoring_and_cdf_match_synchronous_path(tmp_path, monkeypatch):
    owner = threading.get_ident()
    forwards = []

    class Extractor:
        def prepare_frames(self, frames):
            return np.asarray(frames)

        def frames_to_global_embeddings(self, items, batch_size):
            assert threading.get_ident() == owner and batch_size == 8
            forwards.append(len(items[0]))
            x = np.asarray(items[0], dtype=np.float32)
            return [np.stack([x, x * x], axis=1)]

        def frames_to_global_patch_embeddings(self, items, batch_size):
            g = self.frames_to_global_embeddings(items, batch_size)[0]
            return [{"global": g, "patch": np.broadcast_to(g[:, None, :], (len(g), 16, 2)).copy()}]

    params = StableGaussianParams(np.zeros(2), np.eye(2), np.array([-100000.0, 0.0]))
    bundle = ReferenceBundle(
        params, params, params, {i: np.array([-10.0, -1.0]) for i in (1, 2, 3)}, dict(schema=SCHEMA)
    )
    scorer = workflow.VideoScorer.__new__(workflow.VideoScorer)
    scorer.reference = bundle
    scorer.gs = params
    scorer.gt = params
    scorer.device = "cpu"
    scorer.extractor = Extractor()
    scorer.local_scorer = GaussianMeanCandidateScorerFloat64([params], params.mean, "cpu")
    monkeypatch.setattr(
        workflow,
        "decode_bounded",
        lambda path, indices: np.asarray(indices) + 10 * int(Path(path).stem),
    )
    monkeypatch.setattr(workflow, "VideoScorer", lambda *a, **kw: scorer)
    monkeypatch.setattr(workflow, "RawVideoScorer", lambda *a, **kw: scorer)
    import data.prefetch

    monkeypatch.setattr(data.prefetch, "frame_reservation", lambda path, count: count * 10)
    rows = []
    for i, length in enumerate((48, 64, 32)):
        path = tmp_path / f"{i}.mp4"
        path.write_bytes(b"video")
        rows.append(
            dict(
                video_id=str(i),
                video_path=str(path),
                dataset="test",
                subset="real",
                source_model="real",
                downsample_idxs=json.dumps(list(range(length))),
            )
        )
    frame = pd.DataFrame(rows)
    states = workflow.video_file_states(tmp_path, frame)
    weights = tmp_path / "weights"
    weights.write_bytes(b"fixture")
    config = dict(
        runtime=dict(
            device="cpu",
            decode_workers=2,
            prefetch_depth=3,
            prefetch_memory_mb=1,
            prefetch_enabled=False,
        ),
        encoder=dict(repo="unused", weights=str(weights), weights_sha256=file_digest(weights)),
        selection=dict(name="feature_change", k=3),
        method={},
    )
    ordinary = workflow.score_manifest_checkpointed(
        tmp_path, config, frame, bundle, "reference", tmp_path / "ordinary", "ordinary", states
    )
    config["runtime"]["prefetch_enabled"] = True
    prefetched = workflow.score_manifest_checkpointed(
        tmp_path, config, frame, bundle, "reference", tmp_path / "prefetched", "prefetched", states
    )
    pd.testing.assert_frame_equal(ordinary[0], prefetched[0])
    pd.testing.assert_frame_equal(ordinary[1], prefetched[1])
    old_count = len(forwards)
    workflow.score_manifest_checkpointed(
        tmp_path, config, frame, bundle, "reference", tmp_path / "prefetched", "prefetched", states
    )
    assert len(forwards) == old_count
    config["runtime"]["prefetch_enabled"] = False
    regular_cdf = score_cdf_checkpointed(
        tmp_path,
        config,
        frame,
        dict(gs=params, gt=params, lt=params),
        states,
        tmp_path / "cdf_sync",
        "cdf_sync",
    )
    config["runtime"]["prefetch_enabled"] = True
    async_cdf = score_cdf_checkpointed(
        tmp_path,
        config,
        frame,
        dict(gs=params, gt=params, lt=params),
        states,
        tmp_path / "cdf_async",
        "cdf_async",
    )
    assert regular_cdf == async_cdf
