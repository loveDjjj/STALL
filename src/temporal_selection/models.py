"""时间选择产物的数据模型与稳定序列化。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any


WINDOW_MANIFEST_SCHEMA = "caes_window_manifest_v1"


@dataclass(frozen=True)
class CandidateWindow:
    """8 FPS离散轴上的一个合法连续检测窗口。"""

    candidate_id: int
    start_position: int
    frame_indices: tuple[int, ...]
    start_seconds: float
    end_seconds: float
    center_seconds: float
    scores: dict[str, float]


@dataclass(frozen=True)
class SelectedWindow:
    """selector对候选窗口的选择决定。"""

    candidate_id: int
    rank: int
    score: float
    reason: str


@dataclass(frozen=True)
class WindowManifest:
    """selector与dense detector之间唯一的数据契约。"""

    video_id: str
    duration_seconds: float
    base_fps: float
    candidate_stride_seconds: float
    window_seconds: float
    requested_k: int
    candidates: tuple[CandidateWindow, ...]
    selected: tuple[SelectedWindow, ...]
    selector: dict[str, Any]
    schema_version: str = WINDOW_MANIFEST_SCHEMA

    @property
    def effective_k(self) -> int:
        return len(self.selected)

    def validate(self) -> None:
        if self.schema_version != WINDOW_MANIFEST_SCHEMA:
            raise ValueError("不支持的 WindowManifest schema")
        if not self.video_id or self.base_fps <= 0 or self.requested_k < 1:
            raise ValueError("WindowManifest 视频身份/FPS/K无效")
        ids = [item.candidate_id for item in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("候选窗口ID重复")
        frames = [item.frame_indices for item in self.candidates]
        if len(frames) != len(set(frames)):
            raise ValueError("候选窗口帧索引重复")
        mapping = {item.candidate_id: item for item in self.candidates}
        selected_ids = [item.candidate_id for item in self.selected]
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("selected窗口重复")
        if any(item not in mapping for item in selected_ids):
            raise ValueError("selected引用不存在的候选窗口")
        if len(self.selected) > self.requested_k:
            raise ValueError("effective-K超过请求预算")
        ranks = [item.rank for item in self.selected]
        if sorted(ranks) != list(range(1, len(ranks) + 1)):
            raise ValueError("selected rank必须从1连续编号")
        for item in self.candidates:
            if len(item.frame_indices) != round(self.window_seconds * self.base_fps):
                raise ValueError("候选窗口帧数与时间合同不一致")
            if len(item.frame_indices) != len(set(item.frame_indices)):
                raise ValueError("候选窗口包含重复dense frame")
            if item.start_position < 0 or item.start_seconds < 0:
                raise ValueError("候选窗口越过视频起点")
            if item.end_seconds > self.duration_seconds + 1.0 / self.base_fps + 1e-9:
                raise ValueError("候选窗口越过视频末端")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value["effective_k"] = self.effective_k
        for item in value["candidates"]:
            item["frame_indices"] = list(item["frame_indices"])
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WindowManifest":
        payload = dict(value)
        payload.pop("effective_k", None)
        payload["candidates"] = tuple(
            CandidateWindow(
                **{
                    **item,
                    "frame_indices": tuple(int(frame) for frame in item["frame_indices"]),
                }
            )
            for item in payload["candidates"]
        )
        payload["selected"] = tuple(SelectedWindow(**item) for item in payload["selected"])
        result = cls(**payload)
        result.validate()
        return result

    def digest(self) -> str:
        canonical = json.dumps(
            self.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "CandidateWindow",
    "SelectedWindow",
    "WINDOW_MANIFEST_SCHEMA",
    "WindowManifest",
]
