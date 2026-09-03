#!/usr/bin/env python3
"""为CAES构建1 FPS、Global-only、float16严格缓存。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.cache_contract import cache_entry_is_complete
from data.coarse_global_cache import (
    coarse_cache_path,
    expected_coarse_indices,
    prepare_coarse_cache,
    write_coarse_entry,
)
from data.manifest import load_manifest
from data.video import decode_all_frames, decode_indexed_frames
from features import AlphaStallFeatureExtractor


@dataclass(frozen=True)
class ManifestSpec:
    dataset: str
    split: str
    path: Path
    scope: str


MANIFESTS = (
    ManifestSpec("comgenvid", "calibration", ROOT / "data/manifests/development/comgenvid_calibration.csv", "development"),
    ManifestSpec("comgenvid", "evaluation", ROOT / "data/manifests/development/comgenvid_evaluation.csv", "development"),
    ManifestSpec("videofeedback", "calibration", ROOT / "data/manifests/development/videofeedback_calibration.csv", "development"),
    ManifestSpec("videofeedback", "evaluation", ROOT / "data/manifests/development/videofeedback_evaluation.csv", "development"),
    ManifestSpec("genvideo", "calibration", ROOT / "data/manifests/development/genvideo_calibration.csv", "development"),
    ManifestSpec("genvideo", "evaluation", ROOT / "data/manifests/development/genvideo_evaluation.csv", "development"),
    ManifestSpec("genvidbench", "calibration", ROOT / "data/manifests/external/calibration.csv", "external"),
    ManifestSpec("genvidbench", "evaluation", ROOT / "data/manifests/external/evaluation.csv", "external"),
)


def _belongs(video_id: str, shard_index: int, shard_count: int) -> bool:
    value = int.from_bytes(
        hashlib.sha256(video_id.encode("utf-8")).digest()[:8], "little"
    )
    return value % shard_count == shard_index


def _decode(path: Path, indices: list[int]):
    try:
        return decode_indexed_frames(path, indices, require_all=True)
    except ValueError as error:
        frames = decode_all_frames(path, require_open=True)
        if not indices or max(indices) >= len(frames):
            raise ValueError(
                f"coarse cache顺序回退仍缺帧：{path}，解码={len(frames)}"
            ) from error
        return frames[indices]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["development", "external", "all"], default="development")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache/coarse_global_1fps")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--frame-batch-size", type=int, default=64)
    parser.add_argument("--video-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard必须满足0 <= index < count")
    specs = [
        item for item in MANIFESTS
        if args.scope == "all" or item.scope == args.scope
    ]
    jobs = []
    for spec in specs:
        frame = load_manifest(str(spec.path))
        for _, row in frame.iterrows():
            video_id = f"{spec.dataset}:{row['video_path']}"
            if not _belongs(video_id, args.shard_index, args.shard_count):
                continue
            positions, indices = expected_coarse_indices(row)
            if not indices:
                continue
            video_path = Path(str(row["video_path"]))
            if not video_path.is_absolute():
                video_path = ROOT / video_path
            path = coarse_cache_path(
                args.cache_dir,
                dataset=spec.dataset,
                split=spec.split,
                video_id=video_id,
            )
            jobs.append((spec, video_id, video_path, positions, indices, path))
    if args.limit is not None:
        jobs = jobs[: args.limit]
    completed = sum(cache_entry_is_complete(item[-1], strict=True) for item in jobs)
    misses = [item for item in jobs if not cache_entry_is_complete(item[-1], strict=True)]
    estimated_bytes = sum(len(item[4]) * 1024 * 2 for item in misses)
    print(
        f"[审计] jobs={len(jobs)}，complete={completed}，missing={len(misses)}，"
        f"新增Global张量约={estimated_bytes / 1024**3:.3f} GiB",
        flush=True,
    )
    if args.audit_only:
        return

    model = AlphaStallFeatureExtractor(args.device)
    context = prepare_coarse_cache(
        args.cache_dir,
        model=model,
        frame_batch_size=args.frame_batch_size,
        video_batch_size=args.video_batch_size,
        create=True,
    )
    done = 0
    for offset in range(0, len(misses), args.video_batch_size):
        batch = misses[offset : offset + args.video_batch_size]
        with ThreadPoolExecutor(max_workers=min(args.workers, len(batch))) as executor:
            frames = list(executor.map(lambda item: _decode(item[2], item[4]), batch))
        outputs = model.frames_to_global_embeddings(frames, batch_size=args.frame_batch_size)
        for item, output in zip(batch, outputs):
            _, video_id, video_path, positions, indices, path = item
            write_coarse_entry(
                context=context,
                path=path,
                source_video_path=video_path,
                video_id=video_id,
                frame_indices=indices,
                downsample_positions=positions,
                global_features=output,
            )
            done += 1
        print(
            f"[构建] shard={args.shard_index}/{args.shard_count} {done}/{len(misses)}",
            flush=True,
        )
    summary = {
        "scope": args.scope,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "entries_considered": len(jobs),
        "entries_preexisting": completed,
        "entries_written": done,
        "contract_sha256": context.contract_sha256,
    }
    audit = args.cache_dir / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    (audit / f"build_summary_shard_{args.shard_index:03d}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
