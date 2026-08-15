"""Validation for human-readable release directory indexes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReleaseIndexSummary:
    directory_count: int
    asset_count: int


def validate_release_indexes(release_root: Path) -> ReleaseIndexSummary:
    """Require every release directory and asset to be explicitly indexed."""
    root_index = release_root / "README.md"
    if not root_index.is_file():
        raise ValueError("release root lacks README.md")
    root_text = root_index.read_text(encoding="utf-8")

    directories = sorted(path for path in release_root.iterdir() if path.is_dir())
    missing_directories = [
        path.name for path in directories if f"`{path.name}/`" not in root_text
    ]
    if missing_directories:
        raise ValueError(
            f"release root index omits directories: {missing_directories}"
        )

    root_assets = sorted(
        path.name
        for path in release_root.iterdir()
        if path.is_file() and path.name != "README.md"
    )
    if root_assets:
        raise ValueError(
            f"release assets must live in protocol directories: {root_assets}"
        )

    asset_count = 0
    for directory in directories:
        index_path = directory / "README.md"
        if not index_path.is_file():
            raise ValueError(f"release directory lacks README: {directory.name}")
        index_text = index_path.read_text(encoding="utf-8")
        assets = sorted(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.name != "README.md"
        )
        missing_assets = [
            path.relative_to(directory).as_posix()
            for path in assets
            if f"`{path.relative_to(directory).as_posix()}`" not in index_text
        ]
        if missing_assets:
            raise ValueError(
                f"{directory.name}/README.md omits assets: {missing_assets}"
            )
        asset_count += len(assets)

    return ReleaseIndexSummary(
        directory_count=len(directories),
        asset_count=asset_count,
    )
