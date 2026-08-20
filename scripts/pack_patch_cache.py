#!/usr/bin/env python3
"""将严格 K=3 单视频缓存安全迁移为按数据集、split 排列的顺序 shard。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.packed_cache import FORMAT, packed_root, write_index
from data.patch_cache import _get_patch_cache_path, cache_frame_indices
from data.manifest import load_manifest


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="cache/patch_embeddings_k3_2s_8fps")
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-shards", type=int)
    return parser.parse_args()


def jobs(cache_root: Path):
    groups = (("development", ROOT / "data/manifests/development"), ("external", ROOT / "data/manifests/external"))
    for scope, directory in groups:
        for manifest in sorted(directory.glob("*.csv")):
            dataset = "genvidbench" if scope == "external" else manifest.stem.rsplit("_", 1)[0]
            split = "calibration" if manifest.stem.endswith("calibration") else "evaluation"
            for _, row in load_manifest(str(manifest)).iterrows():
                if not cache_frame_indices(row, duration_sec=2, compact=False, cache_window_count=3):
                    continue
                source = _get_patch_cache_path(cache_root, str(row["subset"]), str(row["source_model"]), Path(str(row["video_path"])).stem, 2, False)
                yield scope, dataset, split, source, row


def main() -> None:
    args = arguments()
    if args.shard_size < 1:
        raise ValueError("--shard-size 必须为正整数")
    cache_root = ROOT / args.cache_dir
    root = packed_root(cache_root)
    index_path = root / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {"format": FORMAT, "entries": {}, "shards": {}}
    pending = [item for item in jobs(cache_root) if item[3].relative_to(cache_root).as_posix() not in index["entries"]]
    print(f"待迁移视频：{len(pending)}；shard_size={args.shard_size}")
    if args.dry_run:
        return
    grouped: dict[tuple[str, str, str], list] = {}
    for item in pending:
        grouped.setdefault(item[:3], []).append(item)
    migrated = 0
    for (scope, dataset, split), items in grouped.items():
      for start in range(0, len(items), args.shard_size):
        if args.limit_shards is not None and migrated >= args.limit_shards:
            return
        group = items[start:start + args.shard_size]
        name = f"{scope}/{dataset}/{split}/shard-{len([key for key in index['shards'] if key.startswith(f'{scope}/{dataset}/{split}/')]):05d}.pt"
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        entries = []
        for _, _, _, source, _ in group:
            key = source.relative_to(cache_root).as_posix()
            metadata = json.loads(Path(f"{source}.meta.json").read_text(encoding="utf-8"))
            entries.append({"cache_key": key, "payload": torch.load(source, weights_only=True), "entry_metadata": metadata})
        temporary = destination.with_suffix(".tmp.pt")
        torch.save({"format": FORMAT, "entries": entries}, temporary)
        checked = torch.load(temporary, weights_only=True, mmap=True)
        if [entry["cache_key"] for entry in checked["entries"]] != [entry["cache_key"] for entry in entries]:
            raise RuntimeError("shard 写后校验失败，保留原始缓存")
        temporary.replace(destination)
        index["shards"][name] = {"entries": len(entries)}
        for position, entry in enumerate(entries):
            index["entries"][entry["cache_key"]] = {"shard": name, "position": position}
        write_index(cache_root, index)
        # index 已原子提交且 shard 已读回校验，才删除本 shard 的旧缓存。
        for _, _, _, source, _ in group:
            source.unlink()
            Path(f"{source}.meta.json").unlink()
        migrated += 1
        print(f"已完成 {name}：{len(entries)} 视频")


if __name__ == "__main__":
    main()
