"""完整23单元评价的8帧单窗路径；不改变已冻结16帧评分。"""
from pathlib import Path

from data.video import decode_bounded
from selection import Window, validate_indices


class ShortWindowMixin:
    """输入必须已由清单裁为共同1秒片段，禁止真假采用不同观察预算。"""

    def prepare_coarse(self, path, indices, *, selector='feature_change', k=3,
                       include_uniform=False, materialize=True):
        indices = validate_indices(indices)
        if len(indices) != 8:
            raise ValueError('短视频实验要求清单明确提供8个互异采样帧')
        if selector not in ('uniform', 'feature_change') or k not in (1, 2, 3):
            raise ValueError('未定义的短视频观察策略')
        window = Window(0, tuple(indices))
        uniform = None
        if include_uniform:
            frames = self.extractor.prepare_frames(decode_bounded(Path(path), indices))
            uniform = (window, frames)
        return dict(uniform=uniform, coarse=[], coarse_count=0)

    def plan_from_prepared(self, indices, prepared, *, selector='feature_change', k=3):
        indices = validate_indices(indices)
        if len(indices) != 8:
            raise ValueError('短视频单窗必须恰为8帧')
        uniform = self.global_from_prepared(*prepared['uniform']) if prepared['uniform'] is not None else None
        return dict(windows=[Window(0, tuple(indices))], union=indices,
                    coarse_count=0, selector=selector, uniform=uniform)
