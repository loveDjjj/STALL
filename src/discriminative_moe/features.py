"""不依赖数据拟合的窗口矩摘要，保留部分跨通道方向结构。"""
import numpy as np
import torch


def projection(dimension=1024, rank=64, seed=1729):
    if not 1 <= rank <= dimension:
        raise ValueError('投影维度非法')
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(dimension, rank)))
    q *= np.where(np.diag(r) < 0, -1., 1.)
    return torch.from_numpy(q.astype(np.float32))


def moments(vectors, basis=None):
    """输入二维观察矩阵；空T1显式置零，不给短窗补观察。"""
    if vectors.ndim != 2 or not torch.isfinite(vectors).all():
        raise ValueError('摘要输入须为有限二维矩阵')
    d = vectors.shape[1]
    width = 2*d + (0 if basis is None else basis.shape[1]*(basis.shape[1]-1)//2)
    if not len(vectors):
        return vectors.new_zeros(width)
    parts = [vectors.mean(0), vectors.square().mean(0)]
    if basis is not None:
        projected = vectors @ basis.to(device=vectors.device, dtype=vectors.dtype)
        cross = projected.T @ projected / len(vectors)
        idx = torch.triu_indices(cross.shape[0], cross.shape[1], offset=1, device=vectors.device)
        parts.append(cross[idx[0], idx[1]])
    return torch.cat(parts)


def global_descriptors(global_windows, basis):
    g = torch.as_tensor(global_windows, dtype=torch.float32)
    if g.ndim != 3 or g.shape[1] not in (8, 16):
        raise ValueError('Global须为K×8/16×D')
    appearance=[]; temporal=[]
    for window in g:
        appearance.append(moments(window))
        diff = window[1:] - window[:-1]
        valid = diff.norm(dim=-1) > 0
        unit = torch.nn.functional.normalize(diff[valid], dim=-1, eps=1e-12)
        temporal.append(moments(unit, basis))
    return torch.stack(appearance).numpy(), torch.stack(temporal).numpy()


def local_descriptors(patch_windows, basis):
    if patch_windows.ndim != 4 or patch_windows.shape[1] not in (8, 16):
        raise ValueError('Patch须为K×8/16×P×D')
    # 与主线一样先float32二阶差分和归一化；不删除零D2。
    p = patch_windows.float()
    d2 = p[:, 2:] - 2*p[:, 1:-1] + p[:, :-2]
    unit = torch.nn.functional.normalize(d2, dim=-1, eps=1e-12)
    return torch.stack([moments(x.reshape(-1,x.shape[-1]), basis) for x in unit])
