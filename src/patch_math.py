"""Alpha-STALLED patch 工具共享的轻量 numpy 数学函数。"""

from __future__ import annotations

import numpy as np


def bottomk_mean(arr: np.ndarray, ratio: float) -> np.ndarray:
    """逐样本取最低 ratio 部分数值的均值。

    ``arr`` 解释为 ``[N, ...]``。选择最低值前会展平所有非 batch 维度；
    每个样本至少选取一个值。
    """
    flat = arr.reshape(arr.shape[0], -1)
    k = max(1, int(np.ceil(flat.shape[1] * ratio)))
    part = np.partition(flat, kth=k - 1, axis=1)[:, :k]
    return part.mean(axis=1)


def empirical_percentile(scores: np.ndarray, calib_sorted: np.ndarray) -> np.ndarray:
    """使用右侧 tie 规则的经验 CDF 百分位。

    返回值表示小于等于当前分数的校准分数比例，符合 STALL 的分数方向约定：
    百分位越高，证据越接近真实视频。
    """
    return np.searchsorted(calib_sorted, scores, side="right") / len(calib_sorted)
