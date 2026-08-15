"""STALL 评测的数据加载工具。

四种加载模式：
  load_hf_dataset        — HuggingFace Hub（预计算 embedding，无需 DINOv3）
  load_local_dir         — 目录约定：root/real/<model>/*.mp4, root/fake/<model>/*.mp4
  load_csv               — 显式 CSV，包含 video_path、subset、source_model
  load_csv_with_emb_cache — video_index.py 生成的 enriched CSV + 磁盘 embedding cache
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator, Optional

import warnings

import numpy as np
import pandas as pd

from alpha_stalled.cache_contract import (
    cache_entry_is_complete,
    cache_policy_uses_strict_entries,
    prepare_model_feature_cache,
    tensor_descriptor,
    validate_cache_entry,
    write_cache_entry_metadata,
)
from alpha_stalled.video_io import decode_indexed_frames, load_video_frames

SUBSET_TO_FOLDER = {"real": "real", "annotated": "fake"}


def _is_missing_window(val) -> bool:
    """当窗口索引值缺失时返回 True（None 或 pandas NaN）。"""
    return val is None or (isinstance(val, float) and np.isnan(val))


def _get_cache_path(cache_root: Path, subset: str, source_model: str, stem: str, duration_sec: int, compact: bool) -> Path:
    """返回某个视频对应的 .pt cache 文件路径。"""
    if compact:
        return cache_root / subset / source_model / f"{stem}_{duration_sec}s.pt"
    return cache_root / subset / source_model / f"{stem}.pt"


def _slice_window(full_emb, downsample_idxs, window_idxs, video_path):
    """使用 downsample 索引映射从 full_emb 中切出目标窗口。

    返回切片数组；如果任一窗口索引缺失则返回 None，调用方应跳过。
    """
    idx_to_pos = {native: pos for pos, native in enumerate(downsample_idxs)}
    positions = [idx_to_pos[i] for i in window_idxs if i in idx_to_pos]
    if len(positions) != len(window_idxs):
        warnings.warn(
            f"{video_path}: {len(window_idxs) - len(positions)} window indices "
            "未在 downsample_idxs 中找到，跳过该视频。"
        )
        return None
    return full_emb[positions]


def load_hf_dataset(repo_id: str, split: str = "train", duration: int = 2, verbose: bool = False) -> list[dict]:
    """使用预计算 embeddings.parquet 从 HuggingFace Hub 加载。

    不需要 DINOv3；分数直接由已存储 embedding 计算。

    返回 dict 列表（返回前完成全部加载）：
        {"embs": np.ndarray [1, T, D], "subset": str, "source_model": str, "filename": str}

    Args:
        duration: 使用几秒窗口（1/2/3/4），选择对应 ``<duration>_sec_idxs`` 列。
                  默认 2。
        verbose:  若为 True，打印缺失目标索引列的视频元信息。

    Dependencies: huggingface_hub, datasets, pandas, pyarrow
    """
    from huggingface_hub import hf_hub_download
    import datasets as hf_datasets

    # 只下载一次 embeddings.parquet（缓存在 HF cache 目录）
    print("下载 embeddings.parquet…", flush=True)
    parquet_path = hf_hub_download(
        repo_id=repo_id, filename="embeddings.parquet", repo_type="dataset"
    )
    print("加载 embeddings 到内存…", flush=True)
    emb_df = pd.read_parquet(parquet_path)
    # 每个 "dino_embedding" 单元格是形状为 (T,) 的 numpy object array，
    # 其中每个元素是一维 D 长度 float array。np.stack 将其转换为 (T, D)。
    emb_lookup: dict[str, np.ndarray] = {
        row["file_name"]: np.stack(row["dino_embedding"]).astype(np.float32)
        for _, row in emb_df.iterrows()
    }

    ds = hf_datasets.load_dataset(repo_id, split=split, streaming=True)
    # 禁用视频解码，避免下载原始 .mp4 文件。
    if "video" in ds.features:
        ds = ds.cast_column("video", hf_datasets.Video(decode=False))

    samples = []
    n_missing_idxs = 0
    missing_idxs_meta = []
    for sample in ds:
        filename = sample["filename"]
        folder = SUBSET_TO_FOLDER.get(sample["subset"], "fake")
        key = f"videos/{folder}/{sample['source_model']}/{filename}"
        emb = emb_lookup.get(key)
        if emb is None:
            continue

        # 在 8 FPS 下选择目标时长窗口。
        # 如果视频短于目标窗口（primary key 为 None），则跳过。
        _PRIMARY_KEY = f"{duration}_sec_idxs"
        frame_idxs = sample.get(_PRIMARY_KEY)
        if frame_idxs is not None:
            idxs = np.array(frame_idxs, dtype=int)
            emb = emb[idxs]  # (T_native, D) → (duration*8, D)
        else:
            # 视频短于目标窗口，跳过。
            n_missing_idxs += 1
            missing_idxs_meta.append({k: v for k, v in sample.items() if k != "video"})
            continue

        samples.append({
            "embs": emb[np.newaxis],  # [1, T, D]
            "subset": sample["subset"],
            "source_model": sample["source_model"],
            "filename": filename,
        })

    if n_missing_idxs:
        print(
            f"  警告：{n_missing_idxs} 个视频缺少 '{duration}_sec_idxs' "
            f"（短于目标窗口，已跳过）。传入 --debug 可查看细节。",
            flush=True,
        )
        if verbose:
            for meta in missing_idxs_meta:
                meta_str = ", ".join(f"{k}={v!r}" for k, v in meta.items())
                print(f"    {meta_str}", flush=True)
    print(f"已加载 {len(samples)} 个视频，开始打分…", flush=True)
    return samples


def load_local_dir(root_dir: str) -> Iterator[dict]:
    """遍历 root_dir/real/<model>/*.mp4 和 root_dir/fake/<model>/*.mp4。

    subset 由顶层文件夹推断：
        real/  → "real"
        fake/  → "annotated"
    source_model 是直接子目录名。

    产出 dict：
        {"video_path": str, "subset": str, "source_model": str}
    """
    root = Path(root_dir)
    for folder, subset_val in [("real", "real"), ("fake", "annotated")]:
        subset_path = root / folder
        if not subset_path.exists():
            continue
        for model_dir in sorted(subset_path.iterdir()):
            if not model_dir.is_dir():
                continue
            for video_file in sorted(model_dir.glob("*.mp4")):
                yield {
                    "video_path": str(video_file),
                    "subset": subset_val,
                    "source_model": model_dir.name,
                }


def load_csv(csv_path: str) -> pd.DataFrame:
    """加载必须包含 video_path、subset、source_model 列的 CSV。

    直接返回 pandas DataFrame。

    抛出：
        ValueError: 若缺少任一必需列。
    """
    df = pd.read_csv(csv_path)
    required = {"video_path", "subset", "source_model"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV '{csv_path}' 缺少必需列: {sorted(missing)}\n"
            f"必需列: video_path, subset ('real'/'annotated'), source_model"
        )
    return df


def count_cache_misses(
    csv_path: str,
    emb_cache_dir: str,
    duration_sec: int = 2,
    debug_n: Optional[int] = None,
    compact: bool = False,
    cache_policy: str = "auto",
) -> int:
    """Count how many videos in the CSV are missing from the embedding cache.

    Fast: only checks file existence, no I/O or model calls.
    """
    df = load_csv(csv_path)
    window_col = f"{duration_sec}_sec_idxs"

    if debug_n is not None:
        df = (
            df.groupby(["subset", "source_model"], group_keys=False)
            .apply(lambda g: g.head(debug_n))
            .reset_index(drop=True)
        )

    cache_root = Path(emb_cache_dir)
    strict_entries = cache_policy_uses_strict_entries(cache_root, cache_policy)
    count = 0
    for _, row in df.iterrows():
        stem = Path(row["video_path"]).stem
        if compact and _is_missing_window(row.get(window_col)):
            continue
        cache_path = _get_cache_path(cache_root, row["subset"], row["source_model"], stem, duration_sec, compact)
        if not cache_entry_is_complete(cache_path, strict=strict_entries):
            count += 1
    return count


def prefill_emb_cache(
    csv_path: str,
    emb_cache_dir: str,
    model,
    batch_size: int = 32,
    duration_sec: int = 2,
    debug_n: Optional[int] = None,
    compact: bool = False,
    num_workers: int = 4,
    video_batch: int = 8,
    cache_policy: str = "auto",
):
    """Phase 1：为所有 cache-miss 视频提取并缓存 DINOv3 embeddings。

    cache 命中会直接产出。cache miss 会按 video-batch 并行处理：``num_workers``
    个线程同时解码视频，然后一次 GPU pass 处理 batch 中展平后的全部帧
    （跨视频 batching 可提高 GPU 利用率），最后原子写入结果。

    Args:
        csv_path:      video_index.py 生成的 CSV 路径。
        emb_cache_dir: 逐视频 .pt embedding 文件根目录。
        model:         用于 cache-miss 提取的 STALL 实例。
        batch_size:    每次 DINOv3 forward 的帧数。
        duration_sec:  只用于验证窗口列是否存在。
        debug_n:       若设置，每个 (subset, source_model) 最多处理这么多行。
        compact:       若为 True，只抽取 --duration 秒窗口帧，而不是完整降采样视频。
                       保存为 compact cache：
                       ``{stem}_{duration_sec}s.pt``.
        num_workers:   并行视频解码 CPU 线程数（默认 4）。
        video_batch:   单次 GPU pass 合并处理的视频数（默认 8）。

    Yields:
        每个已处理行对应的 video_path (str)。
    """
    import torch

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

    cache_root = Path(emb_cache_dir)
    cache_context = prepare_model_feature_cache(
        cache_root,
        model=model,
        cache_kind="global_embeddings",
        frame_batch_size=batch_size,
        video_batch_size=video_batch,
        policy=cache_policy,
        create=True,
    )

    # 第 1 遍：不做 I/O，只判断每行是 cache hit 还是 miss。
    misses = []  # 需要 decode+embed 的 (video_path, frame_idxs, cache_path) 元组

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

        cache_path = _get_cache_path(cache_root, subset, source_model, stem, duration_sec, compact)
        if not cache_entry_is_complete(cache_path, strict=cache_context.strict):
            misses.append((video_path, json.loads(frame_idxs_raw), cache_path))

    if not misses:
        return

    # 第 2 遍：按并行 video-batch 处理 cache miss。
    def _decode(job):
        path, frame_idxs, _ = job
        return decode_indexed_frames(
            path,
            frame_idxs,
            require_all=cache_context.strict,
        )

    for chunk_start in range(0, len(misses), video_batch):
        chunk = misses[chunk_start : chunk_start + video_batch]

        # 并行解码当前 chunk 中的所有视频。
        with ThreadPoolExecutor(max_workers=min(num_workers, len(chunk))) as executor:
            chunk_frames = list(executor.map(_decode, chunk))

        # 丢弃没有解码出任何帧的视频，避免单个坏视频中断整个提取流程。
        valid_items = [
            (frames, job)
            for frames, job in zip(chunk_frames, chunk)
            if len(frames) > 0
        ]
        if not valid_items:
            continue

        # 将所有帧展平为一个序列，并记录逐视频长度。
        lengths = [len(frames) for frames, _ in valid_items]
        flat_frames = [fr for frames, _ in valid_items for fr in frames]

        # 对所有展平帧执行一次 GPU pass。
        flat_embs = model._embed_flat_frames(flat_frames, batch_size)

        # 将 embeddings 拆回逐视频，并原子保存。
        cursor = 0
        for emb_len, (_, (video_path, frame_idxs, cache_path)) in zip(lengths, valid_items):
            emb = flat_embs[cursor : cursor + emb_len]
            cursor += emb_len
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(".tmp.pt")
            torch.save(torch.from_numpy(emb), tmp)
            tmp.rename(cache_path)
            if cache_context.strict:
                cached = torch.from_numpy(emb)
                write_cache_entry_metadata(
                    cache_context,
                    cache_path=cache_path,
                    source_video_path=video_path,
                    frame_indices=frame_idxs,
                    payload={
                        "format": "torch_tensor_v1",
                        "embedding": tensor_descriptor(cached),
                    },
                )
            yield video_path


def load_csv_with_emb_cache(
    csv_path: str,
    emb_cache_dir: str,
    model,
    batch_size: int = 32,
    duration_sec: int = 2,
    debug_n: Optional[int] = None,
    compact: bool = False,
    video_batch: int = 8,
    cache_policy: str = "auto",
):
    """加载 video_index.py 生成的 enriched CSV，并通过 embedding cache 产出样本。

    对每个视频行：
      - cache miss：加载帧、提取 DINOv3 embeddings、保存到磁盘。
      - cache hit：从磁盘加载 embeddings。
    随后切到目标时长窗口。

    Args:
        csv_path:      video_index.py 生成的 CSV 路径。
        emb_cache_dir: 逐视频 .pt embedding 文件根目录。
        model:         STALL 实例（用于 cache-miss 提取）。
        batch_size:    cache miss 时每次 DINOv3 forward 的帧数。
        duration_sec:  使用哪个窗口：1、2、3 或 4。
        debug_n:       若设置，每个 (subset, source_model) 最多产出这么多行。
        compact:       若为 True，查找由相同 flag 写出的 compact cache
                       （``{stem}_{duration_sec}s.pt``）。命中时 cache 已经只包含
                       目标窗口帧，不需要重新索引。

    Yields:
        {"embs": np.ndarray [1, T, D], "subset": str, "source_model": str, "filename": str}

    抛出：
        ValueError: 若必需索引列不存在（需要重新运行 video_index.py）。
    """
    import torch

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

    cache_root = Path(emb_cache_dir)
    cache_context = prepare_model_feature_cache(
        cache_root,
        model=model,
        cache_kind="global_embeddings",
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

        # Parse stored JSON index lists
        window_idxs_raw = row[window_col]
        if _is_missing_window(window_idxs_raw):
            # Video too short for this duration — skip
            continue
        downsample_idxs = json.loads(row[idx_col])
        window_idxs = json.loads(window_idxs_raw)

        # Resolve cache paths
        stem = Path(video_path).stem
        compact_cache_path = _get_cache_path(cache_root, subset, source_model, stem, duration_sec, compact=True)
        full_cache_path = _get_cache_path(cache_root, subset, source_model, stem, duration_sec, compact=False)

        if compact and compact_cache_path.exists():
            # Compact cache: already contains exactly the window frames, no re-indexing needed
            cached = torch.load(compact_cache_path, weights_only=True)
            if cache_context.strict:
                validate_cache_entry(
                    cache_context,
                    cache_path=compact_cache_path,
                    source_video_path=video_path,
                    frame_indices=window_idxs,
                    payload={
                        "format": "torch_tensor_v1",
                        "embedding": tensor_descriptor(cached),
                    },
                )
            emb = cached.numpy()  # (T_window, D)
        elif full_cache_path.exists():
            cached = torch.load(full_cache_path, weights_only=True)
            if cache_context.strict:
                validate_cache_entry(
                    cache_context,
                    cache_path=full_cache_path,
                    source_video_path=video_path,
                    frame_indices=downsample_idxs,
                    payload={
                        "format": "torch_tensor_v1",
                        "embedding": tensor_descriptor(cached),
                    },
                )
            full_emb = cached.numpy()  # (N_8fps, D)
            emb = _slice_window(full_emb, downsample_idxs, window_idxs, video_path)
            if emb is None:
                continue
        else:
            if cache_context.strict:
                raise FileNotFoundError(
                    f"strict cache entry missing after prefill: {full_cache_path}"
                )
            # Cache miss: extract embeddings for the downsampled frames
            frames = load_video_frames(video_path, downsample_idxs)
            if len(frames) == 0:
                continue
            full_emb = model.frames_to_embeddings([frames], batch_size=batch_size)[0]
            full_cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic write via temp file to avoid partial .pt files
            tmp = full_cache_path.with_suffix(".tmp.pt")
            torch.save(torch.from_numpy(full_emb), tmp)
            tmp.rename(full_cache_path)
            emb = _slice_window(full_emb, downsample_idxs, window_idxs, video_path)
            if emb is None:
                continue

        yield {
            "embs": emb[np.newaxis],  # [1, T, D]
            "subset": subset,
            "source_model": source_model,
            "filename": filename,
        }
