"""严格特征缓存的顺序 shard 布局与可恢复迁移辅助函数。"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

PACKED_DIRNAME = "packed_v1"
INDEX_NAME = "index.json"
FORMAT = "alpha_stall_packed_patch_v1"


def packed_root(cache_root: Path) -> Path:
    return cache_root / PACKED_DIRNAME


def load_index(cache_root: Path) -> dict[str, Any] | None:
    path = packed_root(cache_root) / INDEX_NAME
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("format") != FORMAT:
        raise ValueError("不支持的 packed cache 索引格式")
    return value


def write_index(cache_root: Path, index: dict[str, Any]) -> None:
    root = packed_root(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / INDEX_NAME
    temporary = target.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


class PackedCacheReader:
    """每个评分进程保留有限 shard LRU，避免反复打开同一个大文件。"""

    def __init__(self, cache_root: Path, max_shards: int = 2) -> None:
        self.cache_root = cache_root
        self.index = load_index(cache_root)
        self.max_shards = max_shards
        self._shards: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, cache_key: str) -> dict[str, Any] | None:
        if self.index is None:
            return None
        item = self.index["entries"].get(cache_key)
        if item is None:
            return None
        shard_name = item["shard"]
        with self._lock:
            shard = self._shards.pop(shard_name, None)
            if shard is None:
                # 评分会访问一个 shard 内的大量非连续张量。mmap 会把这些访问变成同步缺页，
                # 双卡同时跨 shard 时容易把机械盘拖入长时间 I/O 等待；直接读入内存可由 LRU
                # 和操作系统页缓存共同复用可用内存。
                shard = torch.load(packed_root(self.cache_root) / shard_name, weights_only=True)
                if shard.get("format") != FORMAT:
                    raise ValueError(f"packed shard 格式不匹配：{shard_name}")
            self._shards[shard_name] = shard
            while len(self._shards) > self.max_shards:
                self._shards.popitem(last=False)
        entry = shard["entries"][int(item["position"])]
        if entry["cache_key"] != cache_key:
            raise ValueError("packed cache 索引与 shard 条目不一致")
        return entry
