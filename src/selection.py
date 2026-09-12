"""新论文主线的固定候选与窗口选择，不依赖历史实验脚本。"""
from dataclasses import dataclass, replace
import numpy as np


@dataclass(frozen=True)
class Window:
    """起点属于8FPS离散轴；返回列表次序为选择排名，不自动按时间重排。"""
    start_position: int
    frame_indices: tuple[int, ...]
    change: float | None = None


def validate_indices(indices):
    values = list(indices)
    if any(not isinstance(x, (int, np.integer)) or isinstance(x, bool) or x < 0 for x in values):
        raise ValueError('帧索引必须是非负整数')
    if any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError('帧索引必须严格递增且互异')
    return [int(x) for x in values]


def candidates(indices, window_frames=16, stride_frames=4):
    indices = validate_indices(indices)
    if window_frames < 1 or stride_frames < 1:
        raise ValueError('窗口长度和步长必须为正')
    end = len(indices) - window_frames
    if end < 0:
        return []
    starts = list(range(0, end + 1, stride_frames))
    if starts[-1] != end:
        starts.append(end)
    return [Window(start, tuple(indices[start:start+window_frames])) for start in starts]


def uniform_windows(indices, k=3, window_frames=16):
    indices = validate_indices(indices)
    if k < 1:
        raise ValueError('窗口预算必须为正')
    end = len(indices) - window_frames
    if end < 0:
        return []
    starts = np.rint(np.linspace(0, end, k)).astype(int)
    return [Window(int(s), tuple(indices[s:s+window_frames])) for s in dict.fromkeys(starts)]


def feature_change_windows(indices, coarse_global, k=3):
    if k < 1:
        raise ValueError('窗口预算必须为正')
    windows = candidates(indices)
    if not windows:
        return []
    features = np.asarray(coarse_global, dtype=np.float32)
    positions = np.arange(0, len(indices), 8)
    if features.ndim != 2 or len(features) != len(positions) or not np.isfinite(features).all():
        raise ValueError('粗特征与1FPS索引不一致或非有限')
    change = np.linalg.norm(np.diff(features, axis=0), axis=1)
    midpoints = (positions[1:] + positions[:-1]) / 2
    scored = []
    for window in windows:
        mask = (midpoints >= window.start_position) & (midpoints < window.start_position + 16)
        value = change[mask].mean() if mask.any() else change[np.argmin(abs(midpoints-(window.start_position+8)))]
        scored.append(replace(window, change=float(value)))
    return sorted(scored, key=lambda w: (-w.change, w.start_position))[:k]
