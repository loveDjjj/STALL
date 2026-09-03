"""在现有8 FPS索引轴上生成连续2秒候选窗口。"""

from __future__ import annotations

from .models import CandidateWindow
from data.sampling import uniform_windows


def _validate_indices(indices: list[int]) -> None:
    if len(indices) != len(set(indices)):
        raise ValueError("downsample_indices包含重复帧")
    if any(right <= left for left, right in zip(indices, indices[1:])):
        raise ValueError("downsample_indices必须严格递增")


def _window(
    candidate_id: int,
    start: int,
    indices: list[int],
    *,
    base_fps: float,
    window_frames: int,
) -> CandidateWindow:
    start_seconds = start / base_fps
    end_seconds = (start + window_frames) / base_fps
    return CandidateWindow(
        candidate_id=candidate_id,
        start_position=start,
        frame_indices=tuple(indices[start : start + window_frames]),
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        center_seconds=0.5 * (start_seconds + end_seconds),
        scores={},
    )


def generate_candidate_windows(
    downsample_indices: list[int],
    *,
    base_fps: float = 8.0,
    window_seconds: float = 2.0,
    stride_seconds: float = 0.5,
) -> list[CandidateWindow]:
    """生成固定stride候选，并追加尾对齐窗口且删除完全重复项。"""

    _validate_indices(downsample_indices)
    window_frames = round(base_fps * window_seconds)
    stride_frames = round(base_fps * stride_seconds)
    if window_frames < 1 or stride_frames < 1:
        raise ValueError("窗口或stride必须至少对应一帧")
    max_start = len(downsample_indices) - window_frames
    if max_start < 0:
        return []
    starts = list(range(0, max_start + 1, stride_frames))
    if starts[-1] != max_start:
        starts.append(max_start)
    output = []
    seen = set()
    for start in starts:
        frames = tuple(downsample_indices[start : start + window_frames])
        if frames in seen:
            continue
        seen.add(frames)
        output.append(
            _window(
                len(output), start, downsample_indices,
                base_fps=base_fps, window_frames=window_frames,
            )
        )
    return output


def uniform_candidate_windows(
    downsample_indices: list[int],
    *,
    requested_k: int = 3,
    base_fps: float = 8.0,
    window_seconds: float = 2.0,
) -> list[CandidateWindow]:
    """直接包装历史uniform实现，保证FS0索引逐位一致。"""

    _validate_indices(downsample_indices)
    window_frames = round(base_fps * window_seconds)
    windows = uniform_windows(
        downsample_indices, requested_k=requested_k, window_frames=window_frames
    )
    position = {frame: index for index, frame in enumerate(downsample_indices)}
    return [
        _window(
            candidate_id,
            position[frames[0]],
            downsample_indices,
            base_fps=base_fps,
            window_frames=window_frames,
        )
        for candidate_id, frames in enumerate(windows)
    ]


__all__ = ["generate_candidate_windows", "uniform_candidate_windows"]
