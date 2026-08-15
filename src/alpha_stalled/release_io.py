"""Stable identity, path, hashing, and JSON helpers for release artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPOSITORY_ROOT.parent
IDENTITY_COLUMNS = ("dataset", "subset", "source_model", "filename")


def sha256_file(path: Path | str, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file without loading the complete artifact into memory."""
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def video_id(
    row: Mapping[str, Any],
    identity_columns: Sequence[str] = IDENTITY_COLUMNS,
) -> str:
    """Build the stable identity hash used by locked Alpha-STALLED manifests."""
    identity = "|".join(str(row[column]) for column in identity_columns)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def video_id_shard(value: str, num_shards: int) -> int:
    """Assign a locked hexadecimal video ID to its deterministic shard."""
    return int(value[:16], 16) % num_shards


def repository_relative(
    value: str | Path,
    repository_root: Path = REPOSITORY_ROOT,
) -> str:
    """Store absolute paths under the workspace root as portable relative paths."""
    path = Path(value)
    if path.is_absolute():
        try:
            return str(path.relative_to(repository_root.parent))
        except ValueError:
            return str(path)
    return str(path)


def resolve_video(
    value: str | Path,
    repository_root: Path = REPOSITORY_ROOT,
    *,
    require_file: bool = False,
) -> Path:
    """Resolve current, repository-relative, or workspace-relative video paths.

    Manifest builders keep the expected workspace path for a missing file so it
    can be recorded in an audit. Scorers set ``require_file=True`` and fail fast.
    """
    path = Path(value)
    candidates = (path, repository_root / path, repository_root.parent / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    if require_file:
        raise FileNotFoundError(f"video not found: {value}")
    return (repository_root.parent / path).resolve()


def resolve_required_video(
    value: str | Path,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    """Resolve a manifest video path and require the source file to exist."""

    return resolve_video(value, repository_root, require_file=True)


def write_json(path: Path | str, payload: object) -> None:
    """Atomically write deterministic, sorted, UTF-8 JSON with a final newline."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
