"""CAES在8 FPS detector轴上构建1 FPS Global-only严格缓存。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import torch

from .cache_contract import (
    cache_entry_is_complete,
    prepare_model_feature_cache,
    tensor_descriptor,
    validate_cache_entry,
    write_cache_entry_metadata,
)
from .sampling import parse_indices


COARSE_CACHE_FORMAT = "caes_coarse_global_v1"


def coarse_positions(
    downsample_indices: list[int], *, base_fps: int = 8, coarse_fps: int = 1
) -> list[int]:
    """返回严格等间隔coarse位置，不追加不足一个间隔的尾帧。"""

    if base_fps < 1 or coarse_fps < 1 or base_fps % coarse_fps != 0:
        raise ValueError("coarse_fps必须正数且整除base_fps")
    if len(downsample_indices) != len(set(downsample_indices)):
        raise ValueError("downsample_indices包含重复帧")
    stride = base_fps // coarse_fps
    return list(range(0, len(downsample_indices), stride))


def coarse_cache_path(
    root: Path, *, dataset: str, split: str, video_id: str
) -> Path:
    """使用完整video identity哈希，避免跨数据集同stem覆盖。"""

    if not dataset or split not in {"calibration", "evaluation"} or not video_id:
        raise ValueError("coarse cache identity无效")
    digest = hashlib.sha256(video_id.encode("utf-8")).hexdigest()
    return root / dataset / split / digest[:2] / f"{digest}.pt"


def frame_selection_identity(
    *, base_fps: int = 8, coarse_fps: int = 1
) -> dict[str, object]:
    return {
        "mode": "coarse_stride_on_detector_grid",
        "base_fps": base_fps,
        "coarse_fps": coarse_fps,
        "stride_positions": base_fps // coarse_fps,
        "include_partial_tail": False,
        "deduplicate": True,
    }


def prepare_coarse_cache(
    root: Path,
    *,
    model,
    frame_batch_size: int,
    video_batch_size: int,
    policy: str = "strict",
    create: bool = False,
    base_fps: int = 8,
    coarse_fps: int = 1,
):
    return prepare_model_feature_cache(
        root,
        model=model,
        cache_kind="global_embeddings",
        frame_batch_size=frame_batch_size,
        video_batch_size=video_batch_size,
        frame_selection=frame_selection_identity(
            base_fps=base_fps, coarse_fps=coarse_fps
        ),
        feature_dtype="float16",
        policy=policy,
        create=create,
    )


def expected_coarse_indices(
    row: pd.Series, *, base_fps: int = 8, coarse_fps: int = 1
) -> tuple[list[int], list[int]]:
    downsample = parse_indices(row["downsample_idxs"])
    positions = coarse_positions(
        downsample, base_fps=base_fps, coarse_fps=coarse_fps
    )
    return positions, [downsample[position] for position in positions]


def write_coarse_entry(
    *,
    context,
    path: Path,
    source_video_path: Path,
    video_id: str,
    frame_indices: list[int],
    downsample_positions: list[int],
    global_features,
    base_fps: int = 8,
) -> None:
    """原子写入float16 Global与严格逐条元数据。"""

    features = torch.as_tensor(global_features, dtype=torch.float16, device="cpu")
    if tuple(features.shape) != (len(frame_indices), 1024):
        raise ValueError(f"coarse Global shape无效：{tuple(features.shape)}")
    if len(frame_indices) != len(set(frame_indices)):
        raise ValueError("coarse frame indices重复")
    payload = {
        "format": COARSE_CACHE_FORMAT,
        "video_id": video_id,
        "frame_indices": [int(item) for item in frame_indices],
        "downsample_positions": [int(item) for item in downsample_positions],
        "timestamps_seconds": torch.tensor(
            downsample_positions, dtype=torch.float32
        ) / float(base_fps),
        "global": features,
        "contract_sha256": context.contract_sha256,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.pt")
    torch.save(payload, temporary)
    temporary.replace(path)
    write_cache_entry_metadata(
        context,
        cache_path=path,
        source_video_path=source_video_path,
        frame_indices=frame_indices,
        payload={
            "format": COARSE_CACHE_FORMAT,
            "global": tensor_descriptor(features),
            "timestamps_seconds": tensor_descriptor(payload["timestamps_seconds"]),
            "video_id_sha256": hashlib.sha256(video_id.encode("utf-8")).hexdigest(),
        },
    )


def load_coarse_entry(
    *, context, path: Path, source_video_path: Path, frame_indices: list[int]
) -> dict:
    if not cache_entry_is_complete(path, strict=True):
        raise FileNotFoundError(f"coarse cache条目不完整：{path}")
    payload = torch.load(path, weights_only=True)
    if payload.get("format") != COARSE_CACHE_FORMAT:
        raise ValueError("coarse cache payload格式不匹配")
    validate_cache_entry(
        context,
        cache_path=path,
        source_video_path=source_video_path,
        frame_indices=frame_indices,
        payload={
            "format": COARSE_CACHE_FORMAT,
            "global": tensor_descriptor(payload["global"]),
            "timestamps_seconds": tensor_descriptor(payload["timestamps_seconds"]),
            "video_id_sha256": hashlib.sha256(
                str(payload["video_id"]).encode("utf-8")
            ).hexdigest(),
        },
    )
    if payload.get("contract_sha256") != context.contract_sha256:
        raise ValueError("coarse cache payload属于不同contract")
    return payload


__all__ = [
    "COARSE_CACHE_FORMAT",
    "coarse_cache_path",
    "coarse_positions",
    "expected_coarse_indices",
    "frame_selection_identity",
    "load_coarse_entry",
    "prepare_coarse_cache",
    "write_coarse_entry",
]
