"""与逐位置直接白化交叉校验，防止多模型共享计算改变评分。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from evaluation.representations import representation_vectors, RepresentationScorers
from math_utils import StableGaussianParams, l2_normalized_second_order


def test_shared_model_scores_match_direct_whitening():
    rng = np.random.default_rng(17)
    g = rng.normal(size=(2, 4, 5)).astype("float32")
    p = rng.normal(size=(2, 4, 3, 5)).astype("float32")
    p[0, :, 0] = 0
    values = representation_vectors(g, p)
    assert torch.equal(
        values["local_d2"], l2_normalized_second_order(torch.from_numpy(p)).reshape(2, -1, 5)
    )
    models = {}
    for name, x in values.items():
        d = x.shape[-1]
        models[name] = {
            str(i): StableGaussianParams(rng.normal(size=d), rng.normal(size=(d, d)), np.empty(0))
            for i in range(2)
        }
    actual = RepresentationScorers(models, "cpu").score(g, p)
    for name, entries in models.items():
        x = values[name].numpy().astype("float64")
        for key, m in entries.items():
            expected = -0.5 * (
                (((x - m.mean) @ m.whitening) ** 2).sum(-1) + x.shape[-1] * np.log(2 * np.pi)
            ).mean(1)
            np.testing.assert_allclose(actual[name][key], expected, rtol=1e-12, atol=1e-12)
