"""WindowManifest的原子JSONL存储、读取与内容哈希。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from .models import WindowManifest


def write_window_manifests(path: Path, manifests: Iterable[WindowManifest]) -> str:
    """原子写入并返回文件SHA256，拒绝重复video_id。"""

    values = list(manifests)
    video_ids = [item.video_id for item in values]
    if len(video_ids) != len(set(video_ids)):
        raise ValueError("WindowManifest集合包含重复video_id")
    for item in values:
        item.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for item in values:
            handle.write(
                json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            )
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_window_manifests(path: Path) -> list[WindowManifest]:
    values = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                values.append(WindowManifest.from_dict(json.loads(line)))
            except Exception as error:
                raise ValueError(
                    f"WindowManifest第{line_number}行无效：{path}"
                ) from error
    if len({item.video_id for item in values}) != len(values):
        raise ValueError("WindowManifest文件包含重复video_id")
    return values


__all__ = ["read_window_manifests", "write_window_manifests"]
