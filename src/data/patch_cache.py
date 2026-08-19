"""Alpha STALL 的 Global+patch 特征缓存工具。

每个视频缓存为如下 dict：

    {
        "global": Tensor[T, D],
        "patch": Tensor[T, P, D],
        "grid_size": [Gh, Gw],
        "frame_indices": [...],
    }
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import torch

from .cache_contract import (
    cache_entry_is_complete,
    cache_policy_uses_strict_entries,
    prepare_model_feature_cache,
    tensor_descriptor,
    validate_cache_entry,
    write_cache_entry_metadata,
)
from .video import decode_indexed_frames, load_video_frames
from .manifest import is_missing_window, load_manifest
from .sampling import WINDOW_FRAMES, cached_uniform_frame_indices


def _get_patch_cache_path(
    cache_root: Path,
    subset: str,
    source_model: str,
    stem: str,
    duration_sec: int,
    compact: bool,
) -> Path:
    if compact:
        return cache_root / subset / source_model / f"{stem}_{duration_sec}s.pt"
    return cache_root / subset / source_model / f"{stem}.pt"


def cache_frame_indices(
    row: pd.Series,
    *,
    duration_sec: int,
    compact: bool,
    cache_window_count: int | None,
) -> list[int]:
    """依据缓存模式返回当前视频必须保存的帧索引。"""

    if compact and cache_window_count is not None:
        raise ValueError("compact 缓存与多窗口缓存不能同时启用")
    if compact:
        value = row.get(f"{duration_sec}_sec_idxs")
        return [] if is_missing_window(value) else json.loads(value)
    downsample = json.loads(row["downsample_idxs"])
    if cache_window_count is None:
        return downsample
    return cached_uniform_frame_indices(
        downsample,
        cache_window_count=cache_window_count,
        window_frames=WINDOW_FRAMES,
    )


def cache_frame_selection_identity(cache_window_count: int | None) -> dict[str, object] | str:
    """生成写入根级 contract 的帧选择协议。"""

    if cache_window_count is None:
        return "external_native_frame_indices"
    if cache_window_count < 1:
        raise ValueError("cache_window_count 必须是正整数")
    return {
        "mode": "uniform_window_union",
        "window_count": cache_window_count,
        "window_frames": WINDOW_FRAMES,
        "strategy": "uniform",
        "deduplicate": True,
    }


def _belongs_to_shard(row: pd.Series, shard_index: int, shard_count: int) -> bool:
    """按稳定缓存键分片，确保并行任务不会写入同一条目。"""

    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index 必须满足 0 <= shard_index < shard_count")
    key = "\0".join((str(row["subset"]), str(row["source_model"]), Path(str(row["video_path"])).stem))
    value = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")
    return value % shard_count == shard_index


def count_patch_cache_misses(
    csv_path: str,
    patch_emb_cache_dir: str,
    duration_sec: int = 2,
    debug_n: Optional[int] = None,
    compact: bool = False,
    cache_window_count: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    cache_policy: str = "auto",
) -> int:
    df = load_manifest(csv_path)
    if debug_n is not None:
        df = (
            df.groupby(["subset", "source_model"], group_keys=False)
            .apply(lambda g: g.head(debug_n))
            .reset_index(drop=True)
        )

    cache_root = Path(patch_emb_cache_dir)
    strict_entries = cache_policy_uses_strict_entries(cache_root, cache_policy)
    count = 0
    for _, row in df.iterrows():
        if not _belongs_to_shard(row, shard_index, shard_count):
            continue
        stem = Path(row["video_path"]).stem
        if not cache_frame_indices(
            row,
            duration_sec=duration_sec,
            compact=compact,
            cache_window_count=cache_window_count,
        ):
            continue
        cache_path = _get_patch_cache_path(
            cache_root, row["subset"], row["source_model"], stem, duration_sec, compact
        )
        if not cache_entry_is_complete(cache_path, strict=strict_entries):
            count += 1
    return count


def prefill_patch_emb_cache(
    csv_path: str,
    patch_emb_cache_dir: str,
    model,
    batch_size: int = 32,
    duration_sec: int = 2,
    debug_n: Optional[int] = None,
    compact: bool = False,
    cache_window_count: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    num_workers: int = 4,
    video_batch: int = 8,
    cache_policy: str = "auto",
) -> Iterator[str]:
    df = load_manifest(csv_path)

    idx_col = "downsample_idxs"
    window_col = f"{duration_sec}_sec_idxs"
    missing_cols = [c for c in (idx_col, window_col) if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CSV '{csv_path}' 缺少列: {missing_cols}\n"
            f"请运行 scripts/build_manifest.py，生成包含帧索引的 manifest。"
        )

    if debug_n is not None:
        df = (
            df.groupby(["subset", "source_model"], group_keys=False)
            .apply(lambda g: g.head(debug_n))
            .reset_index(drop=True)
        )

    cache_root = Path(patch_emb_cache_dir)
    cache_context = prepare_model_feature_cache(
        cache_root,
        model=model,
        cache_kind="patch_embeddings",
        frame_batch_size=batch_size,
        video_batch_size=video_batch,
        frame_selection=cache_frame_selection_identity(cache_window_count),
        policy=cache_policy,
        create=True,
    )
    misses = []  # (video_path, frame_idxs, cache_path) 元组

    for _, row in df.iterrows():
        if not _belongs_to_shard(row, shard_index, shard_count):
            continue
        video_path = row["video_path"]
        stem = Path(video_path).stem
        subset = row["subset"]
        source_model = row["source_model"]

        frame_indices = cache_frame_indices(
            row,
            duration_sec=duration_sec,
            compact=compact,
            cache_window_count=cache_window_count,
        )
        if not frame_indices:
            continue

        cache_path = _get_patch_cache_path(
            cache_root, subset, source_model, stem, duration_sec, compact
        )
        if not cache_entry_is_complete(cache_path, strict=cache_context.strict):
            misses.append((video_path, frame_indices, cache_path))

    if not misses:
        return

    def _decode(job):
        path, frame_idxs, _ = job
        return decode_indexed_frames(
            path,
            frame_idxs,
            require_all=cache_context.strict,
        )

    for chunk_start in range(0, len(misses), video_batch):
        chunk = misses[chunk_start : chunk_start + video_batch]

        with ThreadPoolExecutor(max_workers=min(num_workers, len(chunk))) as executor:
            chunk_frames = list(executor.map(_decode, chunk))

        valid_items = [
            (frames, job)
            for frames, job in zip(chunk_frames, chunk)
            if len(frames) > 0
        ]
        if not valid_items:
            continue

        video_arrays = [frames for frames, _ in valid_items]
        outputs = model.frames_to_global_patch_embeddings(video_arrays, batch_size=batch_size)

        for output, (_, (video_path, frame_idxs, cache_path)) in zip(outputs, valid_items):
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "global": torch.from_numpy(output["global"]),
                "patch": torch.from_numpy(output["patch"]),
                "grid_size": output["grid_size"],
                "frame_indices": frame_idxs,
            }
            tmp = cache_path.with_suffix(".tmp.pt")
            torch.save(payload, tmp)
            tmp.rename(cache_path)
            if cache_context.strict:
                write_cache_entry_metadata(
                    cache_context,
                    cache_path=cache_path,
                    source_video_path=video_path,
                    frame_indices=frame_idxs,
                    payload={
                        "format": "torch_dict_global_patch_v1",
                        "global": tensor_descriptor(payload["global"]),
                        "patch": tensor_descriptor(payload["patch"]),
                        "grid_size": [int(item) for item in payload["grid_size"]],
                    },
                )
            yield video_path


def load_csv_with_patch_cache(
    csv_path: str,
    patch_emb_cache_dir: str,
    model,
    batch_size: int = 32,
    duration_sec: int = 2,
    debug_n: Optional[int] = None,
    compact: bool = False,
    video_batch: int = 8,
    cache_policy: str = "auto",
):
    df = load_manifest(csv_path)

    idx_col = "downsample_idxs"
    window_col = f"{duration_sec}_sec_idxs"
    missing_cols = [c for c in (idx_col, window_col) if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CSV '{csv_path}' 缺少列: {missing_cols}\n"
            f"请运行 scripts/build_manifest.py，生成包含帧索引的 manifest。"
        )

    if debug_n is not None:
        df = (
            df.groupby(["subset", "source_model"], group_keys=False)
            .apply(lambda g: g.head(debug_n))
            .reset_index(drop=True)
        )

    cache_root = Path(patch_emb_cache_dir)
    cache_context = prepare_model_feature_cache(
        cache_root,
        model=model,
        cache_kind="patch_embeddings",
        frame_batch_size=batch_size,
        video_batch_size=video_batch,
        policy=cache_policy,
        create=False,
    )

    for _, row in df.iterrows():
        video_path = row["video_path"]
        subset = row["subset"]
        source_model = row["source_model"]
        filename = Path(video_path).name

        window_idxs_raw = row[window_col]
        if is_missing_window(window_idxs_raw):
            continue
        downsample_idxs = json.loads(row[idx_col])
        window_idxs = json.loads(window_idxs_raw)

        stem = Path(video_path).stem
        cache_path = _get_patch_cache_path(
            cache_root, subset, source_model, stem, duration_sec, compact
        )

        if cache_path.exists():
            payload = torch.load(cache_path, weights_only=True)
            if cache_context.strict:
                expected_indices = window_idxs if compact else downsample_idxs
                validate_cache_entry(
                    cache_context,
                    cache_path=cache_path,
                    source_video_path=video_path,
                    frame_indices=expected_indices,
                    payload={
                        "format": "torch_dict_global_patch_v1",
                        "global": tensor_descriptor(payload["global"]),
                        "patch": tensor_descriptor(payload["patch"]),
                        "grid_size": [int(item) for item in payload["grid_size"]],
                    },
                )
        else:
            if cache_context.strict:
                raise FileNotFoundError(
                    f"strict patch cache entry missing after prefill: {cache_path}"
                )
            frame_idxs = window_idxs if compact else downsample_idxs
            frames = load_video_frames(video_path, frame_idxs)
            if len(frames) == 0:
                continue
            output = model.frames_to_global_patch_embeddings([frames], batch_size=batch_size)[0]
            payload = {
                "global": torch.from_numpy(output["global"]),
                "patch": torch.from_numpy(output["patch"]),
                "grid_size": output["grid_size"],
                "frame_indices": frame_idxs,
            }
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(".tmp.pt")
            torch.save(payload, tmp)
            tmp.rename(cache_path)

        yield {
            "global": payload["global"].numpy(),
            "patch": payload["patch"].numpy(),
            "grid_size": payload["grid_size"],
            "frame_indices": payload["frame_indices"],
            "subset": subset,
            "source_model": source_model,
            "filename": filename,
            "video_path": video_path,
            "downsample_idxs": downsample_idxs,
            "window_idxs": window_idxs,
        }
