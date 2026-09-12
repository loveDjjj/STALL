"""强局部标量与空间先聚合控制；不改变既有Local D2定义。"""

import numpy as np
import torch


def direction_features(patches, *, epsilon=1e-8, gamma=8.0):
    """CPU FP32的[窗口,帧,Patch,通道]→pooled方向与官方SPLIT统计。

    SPLIT公式核对官方SPLIT_model.py的ttr/_compute_components；
    epsilon及自然对数除数0.693147保持原实现，输出标量为high-fake。
    """
    p = torch.as_tensor(patches, dtype=torch.float32, device="cpu")
    if p.ndim != 4 or p.shape[1] < 3 or not torch.isfinite(p).all():
        raise ValueError("Patch应为有限[W,T,P,D]且T>=3")
    w, t, n, d = p.shape
    side = int(n**0.5)
    if side * side != n or side < 2 or epsilon <= 0 or gamma <= 0:
        raise ValueError("SPLIT需要至少2×2规则网格及正epsilon/gamma")
    mean = p.mean(dim=2)
    pooled = mean[:, 2:] - 2.0 * mean[:, 1:-1] + mean[:, :-2]
    pooled = torch.nn.functional.normalize(pooled, dim=-1, eps=1e-12)
    # 按官方Patch优先布局归约，不能把时间/位置平均顺序换掉。
    trajectories = p.permute(0, 2, 1, 3).reshape(w * n, t, d)
    l1 = torch.norm(trajectories[:, 1:] - trajectories[:, :-1], dim=-1).sum(dim=1)
    l2 = torch.norm(trajectories[:, 2:] - trajectories[:, :-2], dim=-1).sum(dim=1)
    l2 = (l2 / 2) * (t - 1) / (t - 2)
    ttr = ((torch.log(l1 + epsilon) - torch.log(l2 + epsilon)) / 0.693147).reshape(w, n).mean(dim=1)
    grid = p.reshape(w, t, side, side, d)
    motion = grid[:, 1:] - grid[:, :-1]
    dx = torch.norm(motion[:, :, :, :-1] - motion[:, :, :, 1:], dim=-1).mean(dim=(1, 2, 3))
    dy = torch.norm(motion[:, :, :-1] - motion[:, :, 1:], dim=-1).mean(dim=(1, 2, 3))
    lsmi = (dx + dy) / 2
    values = torch.stack([ttr, lsmi, ttr**gamma * lsmi], dim=1)
    if not torch.isfinite(values).all():
        raise ValueError("SPLIT产生非有限分数")
    return pooled.numpy(), values.numpy()


def pooled_score(vectors, params):
    """少量pooled向量逐位置直接FP64白化，避免构造1024²视频矩。"""
    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim != 3 or not np.isfinite(x).all():
        raise ValueError("pooled方向格式无效")
    z = (x - params.mean) @ params.whitening
    return -0.5 * (np.square(z).sum(-1) + params.whitening.shape[1] * np.log(2 * np.pi)).mean(1)


def fit_source_balanced(videos, groups, ridge=1e-5):
    """每源总权重相同、源内片段等权；不改变片段内位置平均。"""
    from collections import Counter
    from math_utils import StableGaussianParams

    if len(videos) != len(groups) or len(set(groups)) < 2:
        raise ValueError("源组支持不足")
    counts = Counter(groups)
    x = np.concatenate(videos).astype(np.float64)
    w = np.concatenate([np.full(len(v), 1 / (len(v) * counts[g])) for v, g in zip(videos, groups)])
    mu = np.average(x, axis=0, weights=w)
    cov = np.atleast_2d(np.cov(x.T, aweights=w, ddof=1))
    e, q = np.linalg.eigh(cov)
    if e.min() < -1e-8:
        raise ValueError("协方差非PSD")
    return StableGaussianParams(mu, q / np.sqrt(np.maximum(e, 0) + ridge), np.empty(0))
