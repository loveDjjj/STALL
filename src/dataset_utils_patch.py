"""PatchSTALL 的 patch 级数据集 cache 工具。

本模块沿用 `dataset_utils.py` 的 CSV + embedding-cache 流程，但每个视频保存为
如下 dict：

    {
        "global": Tensor[T, D],
        "patch": Tensor[T, P, D],
        "grid_size": [Gh, Gw],
        "frame_indices": [...],
    }
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import torch

from dataset_utils import _is_missing_window, load_csv
from stall import load_video_frames


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


def count_patch_cache_misses(
    csv_path: str,
    patch_emb_cache_dir: str,
    duration_sec: int = 2,
    debug_n: Optional[int] = None,
    compact: bool = False,
) -> int:
    df = load_csv(csv_path)
    window_col = f"{duration_sec}_sec_idxs"

    if debug_n is not None:
        df = (
            df.groupby(["subset", "source_model"], group_keys=False)
            .apply(lambda g: g.head(debug_n))
            .reset_index(drop=True)
        )

    cache_root = Path(patch_emb_cache_dir)
    count = 0
    for _, row in df.iterrows():
        stem = Path(row["video_path"]).stem
        if compact and _is_missing_window(row.get(window_col)):
            continue
        cache_path = _get_patch_cache_path(
            cache_root, row["subset"], row["source_model"], stem, duration_sec, compact
        )
        if not cache_path.exists():
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
    num_workers: int = 4,
    video_batch: int = 8,
) -> Iterator[str]:
    df = load_csv(csv_path)

    idx_col = "downsample_idxs"
    window_col = f"{duration_sec}_sec_idxs"
    missing_cols = [c for c in (idx_col, window_col) if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CSV '{csv_path}' 缺少列: {missing_cols}\n"
            f"请重新运行 video_index.py，生成包含帧索引的 enriched CSV。"
        )

    if debug_n is not None:
        df = (
            df.groupby(["subset", "source_model"], group_keys=False)
            .apply(lambda g: g.head(debug_n))
            .reset_index(drop=True)
        )

    cache_root = Path(patch_emb_cache_dir)
    misses = []  # (video_path, frame_idxs, cache_path) 元组

    for _, row in df.iterrows():
        video_path = row["video_path"]
        stem = Path(video_path).stem
        subset = row["subset"]
        source_model = row["source_model"]

        if compact:
            window_idxs_raw = row[window_col]
            if _is_missing_window(window_idxs_raw):
                continue
            frame_idxs_raw = window_idxs_raw
        else:
            frame_idxs_raw = row[idx_col]

        cache_path = _get_patch_cache_path(
            cache_root, subset, source_model, stem, duration_sec, compact
        )
        if not cache_path.exists():
            misses.append((video_path, json.loads(frame_idxs_raw), cache_path))

    if not misses:
        return

    def _decode(job):
        path, frame_idxs, _ = job
        return load_video_frames(path, frame_idxs)

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
            yield video_path


def load_csv_with_patch_cache(
    csv_path: str,
    patch_emb_cache_dir: str,
    model,
    batch_size: int = 32,
    duration_sec: int = 2,
    debug_n: Optional[int] = None,
    compact: bool = False,
):
    df = load_csv(csv_path)

    idx_col = "downsample_idxs"
    window_col = f"{duration_sec}_sec_idxs"
    missing_cols = [c for c in (idx_col, window_col) if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CSV '{csv_path}' 缺少列: {missing_cols}\n"
            f"请重新运行 video_index.py，生成包含帧索引的 enriched CSV。"
        )

    if debug_n is not None:
        df = (
            df.groupby(["subset", "source_model"], group_keys=False)
            .apply(lambda g: g.head(debug_n))
            .reset_index(drop=True)
        )

    cache_root = Path(patch_emb_cache_dir)

    for _, row in df.iterrows():
        video_path = row["video_path"]
        subset = row["subset"]
        source_model = row["source_model"]
        filename = Path(video_path).name

        window_idxs_raw = row[window_col]
        if _is_missing_window(window_idxs_raw):
            continue
        downsample_idxs = json.loads(row[idx_col])
        window_idxs = json.loads(window_idxs_raw)

        stem = Path(video_path).stem
        cache_path = _get_patch_cache_path(
            cache_root, subset, source_model, stem, duration_sec, compact
        )

        if cache_path.exists():
            payload = torch.load(cache_path, weights_only=True)
        else:
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
