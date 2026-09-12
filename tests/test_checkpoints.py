"""checkpoint内容验证、Infinity序列化及懒加载恢复行为。"""

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from artifacts import checkpoint_write, checkpoint_read
import workflow


def test_checksum_and_positive_infinity(tmp_path):
    path = tmp_path / "point.json"
    checkpoint_write(path, "fixed", dict(video_id="0001", raw=float("inf")))
    value = checkpoint_read(path, "fixed", "0001")
    assert value["raw"] == "Infinity"
    broken = json.loads(path.read_text())
    broken["payload"]["raw"] = 0.0
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="损坏"):
        checkpoint_read(path, "fixed", "0001")


def test_score_checkpoint_does_not_load_model_again(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fixture")
    frame = pd.DataFrame(
        [
            dict(
                video_id="v",
                video_path=str(video),
                dataset="d",
                subset="real",
                source_model="r",
                downsample_idxs=json.dumps(list(range(16))),
            )
        ]
    )
    config = dict(
        runtime=dict(device="cpu"),
        encoder=dict(repo="unused", weights="unused"),
        selection=dict(name="feature_change", k=3),
        method={},
    )
    calls = []

    class Scorer:
        def __init__(self, *args, **kwargs):
            calls.append("model")

        def score(self, *args, **kwargs):
            return dict(
                final_score=0.75,
                global_score=0.5,
                local_score=1.0,
                local_raw=2.0,
                effective_k=1,
                windows=[
                    dict(
                        rank=0,
                        frame_indices=list(range(16)),
                        global_spatial_raw=1.0,
                        global_temporal_raw=float("inf"),
                        local_raw=2.0,
                    )
                ],
            )

    monkeypatch.setattr(workflow, "VideoScorer", Scorer)
    states = workflow.video_file_states(tmp_path, frame)
    first = workflow.score_manifest_checkpointed(
        tmp_path, config, frame, None, "reference", tmp_path, "identity", states
    )
    second = workflow.score_manifest_checkpointed(
        tmp_path, config, frame, None, "reference", tmp_path, "identity", states
    )
    assert calls == ["model"]
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
    assert np.isposinf(second[1].global_temporal_raw.iloc[0])
