"""与独立公式对照：空间聚合、SPLIT几何、短窗和零变化边界。"""

import numpy as np
import pytest
import torch
from evaluation.direction_features import direction_features, pooled_score, fit_source_balanced
from math_utils import StableGaussianParams


@pytest.mark.parametrize("t", [8, 16])
def test_split_matches_independent_patch_loops(t):
    x = torch.from_numpy(np.random.default_rng(17).normal(size=(2, t, 4, 5)).astype("float32"))
    pooled, scores = direction_features(x)
    for w in range(2):
        local = []
        for i in range(4):
            y = x[w, :, i]
            a = (y[1:] - y[:-1]).norm(dim=-1).sum()
            b = (y[2:] - y[:-2]).norm(dim=-1).sum() / 2 * (t - 1) / (t - 2)
            local.append((torch.log(a + 1e-8) - torch.log(b + 1e-8)) / 0.693147)
        ttr = torch.stack(local).mean()
        v = (x[w, 1:] - x[w, :-1]).reshape(t - 1, 2, 2, 5)
        lsmi = (
            (v[:, :, 0] - v[:, :, 1]).norm(dim=-1).mean() + (v[:, 0] - v[:, 1]).norm(dim=-1).mean()
        ) / 2
        np.testing.assert_allclose(scores[w], [ttr, lsmi, ttr**8 * lsmi], rtol=2e-6, atol=1e-7)
    # 原始空间平均和差分可交换；浮点求和次序只比较合理容差。
    mean = x.mean(2)
    a = mean[:, 2:] - 2 * mean[:, 1:-1] + mean[:, :-2]
    b = (x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]).mean(2)
    torch.testing.assert_close(a, b)
    np.testing.assert_allclose(pooled, torch.nn.functional.normalize(a, dim=-1), atol=1e-7)


def test_stationary_and_constant_velocity_boundaries():
    x = torch.ones((1, 8, 4, 3))
    p, s = direction_features(x)
    assert np.array_equal(p, np.zeros_like(p)) and np.array_equal(s, np.zeros_like(s))
    x = torch.arange(8, dtype=torch.float32)[None, :, None, None].expand(1, 8, 4, 3)
    p, s = direction_features(x)
    assert np.array_equal(p, np.zeros_like(p))
    np.testing.assert_allclose(s, 0, atol=1e-6)


def test_pooled_score_and_direction_cancellation():
    rng = np.random.default_rng(19)
    u = rng.normal(size=(2, 6, 3)).astype("float32")
    m = StableGaussianParams(rng.normal(size=3), rng.normal(size=(3, 3)), np.empty(0))
    expected = []
    for window in u:
        expected.append(
            np.mean(
                [
                    -0.5
                    * (
                        np.linalg.norm((v.astype(float) - m.mean) @ m.whitening) ** 2
                        + 3 * np.log(2 * np.pi)
                    )
                    for v in window
                ]
            )
        )
    np.testing.assert_allclose(pooled_score(u, m), expected, atol=1e-12)
    x = torch.zeros(1, 8, 4, 3)
    x[:, :, 0, 0] = torch.arange(8) ** 2
    x[:, :, 1, 0] = -(torch.arange(8) ** 2)
    pooled, _ = direction_features(x)
    assert not torch.equal(x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2], torch.zeros(1, 6, 4, 3))
    assert np.array_equal(pooled, np.zeros_like(pooled))


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        direction_features(torch.zeros(1, 8, 3, 5))
    with pytest.raises(ValueError):
        direction_features(torch.full((1, 8, 4, 5), float("nan")))


def test_source_weights_reduce_to_clip_weights_when_unique():
    from reference import fit_gaussian

    rng = np.random.default_rng(4)
    videos = [rng.normal(size=(n, 5)) for n in (7, 9, 11)]
    a = fit_source_balanced(videos, ["a", "b", "c"])
    b = fit_gaussian(videos)
    np.testing.assert_array_equal(a.mean, b.mean)
    np.testing.assert_array_equal(a.whitening, b.whitening)
    c = fit_source_balanced(videos, ["a", "a", "b"])
    expected = 0.25 * videos[0].mean(0) + 0.25 * videos[1].mean(0) + 0.5 * videos[2].mean(0)
    np.testing.assert_allclose(c.mean, expected, atol=1e-14)
