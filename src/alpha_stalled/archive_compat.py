"""Compatibility helpers for command-line tools moved into the research archive."""

from __future__ import annotations

import runpy
from pathlib import Path

from .release_io import REPOSITORY_ROOT


ARCHIVED_TOOL_ROOT = REPOSITORY_ROOT / "research_archive/tools"


def archived_tool_path(relative_path: str | Path) -> Path:
    """Resolve and validate a repository-owned archived tool implementation."""

    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"archived tool path must be relative: {relative_path}")
    target = (ARCHIVED_TOOL_ROOT / relative).resolve()
    if not target.is_relative_to(ARCHIVED_TOOL_ROOT.resolve()):
        raise ValueError(f"archived tool path escapes archive root: {relative_path}")
    if not target.is_file():
        raise FileNotFoundError(target)
    return target


def run_archived_tool(relative_path: str | Path) -> None:
    """Execute an archived implementation while preserving the caller's arguments."""

    runpy.run_path(str(archived_tool_path(relative_path)), run_name="__main__")


__all__ = ["ARCHIVED_TOOL_ROOT", "archived_tool_path", "run_archived_tool"]
