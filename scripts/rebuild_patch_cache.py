#!/usr/bin/env python3
"""按当前 manifest 重建可追溯的 DINOv3 Global+patch 特征缓存。

此脚本只处理 manifest 中的样本，并使用严格 cache contract 写入根级和
逐文件元数据。它可以重复执行：已验证完整的条目会跳过，未完成条目继续构建。
默认保存支撑 K=1/K=2/K=3 的 2 秒均匀窗口并集；主 runner 不再依赖完整下采样序列。
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import io
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.manifest import is_missing_window, load_manifest
from data.cache_contract import (
    cache_entry_is_complete,
    prepare_model_feature_cache,
    tensor_descriptor,
    write_cache_entry_metadata,
)
from data.patch_cache import (
    _belongs_to_shard,
    _get_patch_cache_path,
    cache_frame_indices,
    cache_frame_selection_identity,
    count_patch_cache_misses,
)
from data.video import decode_all_frames, decode_indexed_frames
from features import AlphaStallFeatureExtractor


DEFAULT_MANIFESTS = (
    ROOT / "data/manifests/development/comgenvid_calibration.csv",
    ROOT / "data/manifests/development/comgenvid_evaluation.csv",
    ROOT / "data/manifests/development/genvideo_calibration.csv",
    ROOT / "data/manifests/development/genvideo_evaluation.csv",
    ROOT / "data/manifests/development/videofeedback_calibration.csv",
    ROOT / "data/manifests/development/videofeedback_evaluation.csv",
    ROOT / "data/manifests/external/calibration.csv",
    ROOT / "data/manifests/external/evaluation.csv",
)


@dataclass(frozen=True)
class ManifestAudit:
    path: Path
    total_rows: int
    cacheable_rows: int
    short_video_rows: int
    missing_videos: int
    duplicate_keys: int


class ManifestFrameCountMismatch(ValueError):
    """视频顺序解码的实际帧数少于 manifest 请求的帧索引。"""

    def __init__(self, video_path: str, decoded_frames: int, seek_error: ValueError):
        self.video_path = video_path
        self.decoded_frames = decoded_frames
        self.seek_error = seek_error
        super().__init__(
            f"视频 {video_path} 随机取帧失败（{seek_error}），且顺序解码仅得到 "
            f"{decoded_frames} 帧。"
        )


def _resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _read_manifests(paths: list[Path], duration_sec: int) -> tuple[list[ManifestAudit], list[dict[str, str]]]:
    window_col = f"{duration_sec}_sec_idxs"
    audits: list[ManifestAudit] = []
    unavailable: list[dict[str, str]] = []
    all_keys: Counter[tuple[str, str, str]] = Counter()

    for path in paths:
        frame = load_manifest(str(path))
        if window_col not in frame.columns:
            raise ValueError(f"{path} 缺少当前采样所需的列 {window_col}")
        missing_video = ~frame["video_path"].map(lambda value: _resolve_path(value).is_file())
        short_video = frame[window_col].map(is_missing_window)
        for _, row in frame[short_video].iterrows():
            unavailable.append(
                {
                    "manifest": path.relative_to(ROOT).as_posix(),
                    "video_path": str(row["video_path"]),
                    "reason": f"缺少 {window_col}",
                }
            )
        keys = [
            (str(row["subset"]), str(row["source_model"]), Path(str(row["video_path"])).stem)
            for _, row in frame.iterrows()
        ]
        all_keys.update(keys)
        audits.append(
            ManifestAudit(
                path=path,
                total_rows=len(frame),
                cacheable_rows=len(frame),
                short_video_rows=int(short_video.sum()),
                missing_videos=int(missing_video.sum()),
                duplicate_keys=0,
            )
        )

    duplicate_keys = sum(count - 1 for count in all_keys.values() if count > 1)
    if duplicate_keys:
        raise ValueError(
            f"当前 manifest 在缓存路径键 subset/source_model/stem 上有 {duplicate_keys} 个冲突；"
            "请先消除同名视频冲突，不能覆盖写入。"
        )
    if any(item.missing_videos for item in audits):
        missing = sum(item.missing_videos for item in audits)
        raise FileNotFoundError(f"当前 manifest 有 {missing} 个视频路径不存在，拒绝构建不完整缓存。")
    return audits, unavailable


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _append_csv(path: Path, row: dict[str, str], fields: list[str]) -> None:
    """立即记录单条降级事件，避免后续进程异常时丢失审计信息。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    has_header = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not has_header:
            writer.writeheader()
        writer.writerow(row)


def _decode_cached_frames(video_path: str, frame_indices: list[int]) -> tuple[object, str | None]:
    """优先随机取帧；对 H.264 seek 边界问题回退到完整顺序解码。"""

    try:
        return decode_indexed_frames(video_path, frame_indices, require_all=True), None
    except ValueError as seek_error:
        frames = decode_all_frames(video_path, require_open=True)
        if len(frames) <= max(frame_indices):
            raise ManifestFrameCountMismatch(video_path, len(frames), seek_error) from seek_error
        return frames[frame_indices], str(seek_error)


def _downsample_indices(num_frames: int, fps: float, target_fps: float = 8.0) -> list[int]:
    """与 manifest 生成器一致地重算原始帧降采样索引。"""

    if fps < target_fps:
        raise ValueError(f"视频 fps={fps} 低于目标采样率 {target_fps}")
    ratio = fps / target_fps
    indices: list[int] = []
    step = 0
    while True:
        index = round(ratio * step)
        if index >= num_frames:
            return indices
        indices.append(index)
        step += 1


def _compute_windows(indices: list[int], target_fps: float = 8.0) -> dict[str, str | None]:
    """与 manifest 生成器保持一致，固定种子生成 1-4 秒窗口。"""

    rng = np.random.RandomState(42)
    output: dict[str, str | None] = {}
    for seconds in (1, 2, 3, 4):
        size = int(round(target_fps * seconds))
        name = f"{seconds}_sec_idxs"
        if len(indices) < size:
            output[name] = None
            continue
        start = int(rng.randint(0, len(indices) - size + 1))
        output[name] = json.dumps(indices[start : start + size])
    return output


def _repair_manifest_frame_count(manifest: Path, row: pd.Series, decoded_frames: int) -> pd.Series:
    """用顺序解码的实际帧数修正一个 manifest 行，并返回更新后的行。"""

    fps = float(row["fps"])
    indices = _downsample_indices(decoded_frames, fps)
    updated = row.copy()
    updated["num_frames"] = decoded_frames
    updated["duration_seconds"] = decoded_frames / fps
    updated["downsample_idxs"] = json.dumps(indices)
    for key, value in _compute_windows(indices).items():
        updated[key] = value

    lock_path = Path(f"{manifest}.lock")
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            lines = manifest.read_text(encoding="utf-8").splitlines(keepends=True)
            header = next(csv.reader([lines[0].rstrip("\r\n")]))
            prefix = f"{row['video_path']},"
            matches = [index for index, line in enumerate(lines[1:], start=1) if line.startswith(prefix)]
            if len(matches) != 1:
                raise ValueError(f"无法唯一定位待修正 manifest 行：{row['video_path']}")
            record = next(csv.reader([lines[matches[0]].rstrip("\r\n")]))
            values = dict(zip(header, record))
            for key, value in updated.items():
                if key in values:
                    values[key] = "" if value is None else str(value)
            buffer = io.StringIO()
            csv.writer(buffer, lineterminator="\n").writerow([values[key] for key in header])
            lines[matches[0]] = buffer.getvalue()
            temporary = Path(f"{manifest}.tmp")
            temporary.write_text("".join(lines), encoding="utf-8")
            temporary.replace(manifest)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return updated


def _write_cached_features(
    *,
    cache_path: Path,
    video_path: str,
    frame_indices: list[int],
    frames: object,
    model: AlphaStallFeatureExtractor,
    context,
    frame_batch_size: int,
) -> None:
    """提取一个视频的 DINO 特征并写入严格缓存与逐条元数据。"""

    output = model.frames_to_global_patch_embeddings([frames], batch_size=frame_batch_size)[0]
    payload = {
        "global": torch.from_numpy(output["global"]),
        "patch": torch.from_numpy(output["patch"]),
        "grid_size": output["grid_size"],
        "frame_indices": frame_indices,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp.pt")
    torch.save(payload, tmp)
    tmp.replace(cache_path)
    write_cache_entry_metadata(
        context,
        cache_path=cache_path,
        source_video_path=video_path,
        frame_indices=frame_indices,
        payload={
            "format": "torch_dict_global_patch_v1",
            "global": tensor_descriptor(payload["global"]),
            "patch": tensor_descriptor(payload["patch"]),
            "grid_size": [int(value) for value in payload["grid_size"]],
        },
    )


def _prefill_manifest_resilient(
    *,
    manifest: Path,
    cache_dir: Path,
    model: AlphaStallFeatureExtractor,
    context,
    duration_sec: int,
    cache_window_count: int,
    frame_batch_size: int,
    shard_index: int,
    shard_count: int,
    fallback_audit_path: Path,
    manifest_repair_audit_path: Path,
):
    """逐视频构建缓存，在随机定位失败时保留可审计的顺序解码回退。"""

    frame = load_manifest(str(manifest))
    for _, row in frame.iterrows():
        if not _belongs_to_shard(row, shard_index, shard_count):
            continue
        frame_indices = cache_frame_indices(
            row,
            duration_sec=duration_sec,
            compact=False,
            cache_window_count=cache_window_count,
        )
        if not frame_indices:
            continue
        video_path = str(row["video_path"])
        cache_path = _get_patch_cache_path(
            cache_dir,
            str(row["subset"]),
            str(row["source_model"]),
            Path(video_path).stem,
            duration_sec,
            False,
        )
        if cache_entry_is_complete(cache_path, strict=context.strict):
            continue
        try:
            frames, fallback_reason = _decode_cached_frames(video_path, frame_indices)
        except ManifestFrameCountMismatch as mismatch:
            original_num_frames = int(row["num_frames"])
            row = _repair_manifest_frame_count(manifest, row, mismatch.decoded_frames)
            frame_indices = cache_frame_indices(
                row,
                duration_sec=duration_sec,
                compact=False,
                cache_window_count=cache_window_count,
            )
            if not frame_indices:
                raise ValueError(
                    f"视频 {video_path} 修正后的实际帧数为 {mismatch.decoded_frames}，"
                    "不足以构建当前 K 窗口缓存。"
                ) from mismatch
            frames_all = decode_all_frames(video_path, require_open=True)
            if len(frames_all) <= max(frame_indices):
                raise RuntimeError(f"修正后的 manifest 仍请求不存在的帧：{video_path}") from mismatch
            frames = frames_all[frame_indices]
            fallback_reason = str(mismatch.seek_error)
            _append_csv(
                manifest_repair_audit_path,
                {
                    "manifest": manifest.relative_to(ROOT).as_posix(),
                    "video_path": video_path,
                    "manifest_num_frames": str(original_num_frames),
                    "decoded_num_frames": str(mismatch.decoded_frames),
                    "reason": str(mismatch.seek_error),
                },
                [
                    "manifest",
                    "video_path",
                    "manifest_num_frames",
                    "decoded_num_frames",
                    "reason",
                ],
            )
            print(
                f"已按顺序解码帧数修正 manifest：{video_path} "
                f"({original_num_frames} -> {mismatch.decoded_frames})",
                flush=True,
            )
        if fallback_reason is not None:
            _append_csv(
                fallback_audit_path,
                {
                    "manifest": manifest.relative_to(ROOT).as_posix(),
                    "video_path": video_path,
                    "reason": fallback_reason,
                    "max_frame_index": str(max(frame_indices)),
                },
                ["manifest", "video_path", "reason", "max_frame_index"],
            )
            print(f"随机取帧失败，已顺序解码恢复：{video_path}", flush=True)
        _write_cached_features(
            cache_path=cache_path,
            video_path=video_path,
            frame_indices=frame_indices,
            frames=frames,
            model=model,
            context=context,
            frame_batch_size=frame_batch_size,
        )
        yield video_path


def _check_gpu(device: str, minimum_free_gib: float) -> None:
    if not device.startswith("cuda"):
        return
    if not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 缓存构建，但当前 PyTorch 未检测到可用 CUDA。")
    index = torch.device(device).index or 0
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    free_gib = free_bytes / 1024**3
    total_gib = total_bytes / 1024**3
    print(f"GPU {index}: 空闲 {free_gib:.1f} GiB / 总计 {total_gib:.1f} GiB", flush=True)
    if free_gib < minimum_free_gib:
        raise RuntimeError(
            f"GPU {index} 仅剩 {free_gib:.1f} GiB，低于安全阈值 {minimum_free_gib:.1f} GiB；"
            "请等待其他任务释放显存后重试，脚本不会抢占其他任务。"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="要构建的 manifest；可重复指定。未指定时使用全部当前 development/external 清单。",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "cache/patch_embeddings_k3_2s_8fps",
        help="严格 K 窗口缓存根目录，绝不应指向 legacy patch_embeddings 或完整序列缓存。",
    )
    parser.add_argument("--duration-sec", type=int, default=2, choices=(1, 2, 3, 4))
    parser.add_argument(
        "--cache-window-count",
        type=int,
        default=3,
        help="每视频缓存的均匀窗口数；默认 K=3，同时覆盖 K=1/K=2 评分。",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-batch-size", type=int, default=8)
    parser.add_argument("--video-batch-size", type=int, default=1)
    parser.add_argument("--decode-workers", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0, help="当前并行分片编号，从 0 开始。")
    parser.add_argument("--shard-count", type=int, default=1, help="稳定哈希分片总数；两张卡并行时设为 2。")
    parser.add_argument("--minimum-free-gib", type=float, default=12.0)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--skip-audit-report",
        action="store_true",
        help="不重写根目录审计 CSV；用于多个 GPU 并行恢复同一缓存时保留已有全量审计。",
    )
    parser.add_argument(
        "--allow-short-videos",
        action="store_true",
        help="允许不足 2 秒的样本被记录为不适用；不设置时将其视为协议错误。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.frame_batch_size, args.video_batch_size, args.decode_workers, args.cache_window_count) < 1:
        raise ValueError("batch size、decode workers 和 cache_window_count 必须为正整数")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard_index 必须满足 0 <= shard_index < shard_count")
    manifests = [_resolve_path(value) for value in args.manifest] if args.manifest else list(DEFAULT_MANIFESTS)
    manifests = [path.resolve() for path in manifests]
    if len(set(manifests)) != len(manifests):
        raise ValueError("manifest 参数存在重复文件")
    for path in manifests:
        if not path.is_file():
            raise FileNotFoundError(f"找不到 manifest: {path}")

    audits, unavailable = _read_manifests(manifests, args.duration_sec)
    cache_dir = _resolve_path(args.cache_dir).resolve()
    if not args.skip_audit_report:
        report_dir = cache_dir / "audit"
        _write_csv(
            report_dir / "short_or_ineligible_videos.csv",
            unavailable,
            ["manifest", "video_path", "reason"],
        )
    summary = {
        "duration_sec": args.duration_sec,
        "cache_dir": str(cache_dir),
        "manifests": [
            {
                "path": item.path.relative_to(ROOT).as_posix(),
                "total_rows": item.total_rows,
                "cacheable_rows": item.cacheable_rows,
                "short_video_rows": item.short_video_rows,
                "missing_videos": item.missing_videos,
            }
            for item in audits
        ],
        "cacheable_rows": sum(item.cacheable_rows for item in audits),
        "short_video_rows": len(unavailable),
        "shard": {"index": args.shard_index, "count": args.shard_count},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if unavailable:
        print(
            f"发现 {len(unavailable)} 个不足 {args.duration_sec} 秒的样本；"
            "它们不会写入 K 窗口缓存，主实验将按 data.short_video_policy 明确排除。",
            flush=True,
        )
    if args.audit_only:
        return

    _check_gpu(args.device, args.minimum_free_gib)
    model = AlphaStallFeatureExtractor(device=args.device)
    cache_context = prepare_model_feature_cache(
        cache_dir,
        model=model,
        cache_kind="patch_embeddings",
        frame_batch_size=args.frame_batch_size,
        video_batch_size=args.video_batch_size,
        frame_selection=cache_frame_selection_identity(args.cache_window_count),
        policy="strict",
        create=True,
    )
    fallback_audit_path = (
        cache_dir / "audit" / f"sequential_decode_fallbacks_shard{args.shard_index}.csv"
    )
    manifest_repair_audit_path = (
        cache_dir / "audit" / f"manifest_frame_count_repairs_shard{args.shard_index}.csv"
    )
    completed = 0
    for manifest in manifests:
        missing_before = count_patch_cache_misses(
            str(manifest), str(cache_dir), duration_sec=args.duration_sec,
            compact=False, cache_window_count=args.cache_window_count, cache_policy="strict",
            shard_index=args.shard_index, shard_count=args.shard_count,
        )
        print(f"构建 {manifest.relative_to(ROOT)}：待处理 {missing_before} 条", flush=True)
        for video_path in _prefill_manifest_resilient(
            manifest=manifest,
            cache_dir=cache_dir,
            model=model,
            context=cache_context,
            duration_sec=args.duration_sec,
            cache_window_count=args.cache_window_count,
            frame_batch_size=args.frame_batch_size,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            fallback_audit_path=fallback_audit_path,
            manifest_repair_audit_path=manifest_repair_audit_path,
        ):
            completed += 1
            if completed % 25 == 0:
                print(f"已完成 {completed} 条，最近视频：{video_path}", flush=True)
        missing_after = count_patch_cache_misses(
            str(manifest), str(cache_dir), duration_sec=args.duration_sec,
            compact=False, cache_window_count=args.cache_window_count, cache_policy="strict",
            shard_index=args.shard_index, shard_count=args.shard_count,
        )
        if missing_after:
            raise RuntimeError(f"{manifest} 构建后仍缺少 {missing_after} 条严格缓存")
    print(f"严格缓存构建完成：本次新写入 {completed} 条，根目录：{cache_dir}", flush=True)


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
