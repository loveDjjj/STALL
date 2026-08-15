"""Deterministic CSV shard and checkpoint-part loading primitives."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd


def expected_shard_paths(
    directory: str | Path,
    prefixes: Iterable[str],
    num_shards: int,
) -> list[Path]:
    """Return all declared shard paths in prefix-major, shard-minor order."""
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    root = Path(directory)
    paths = [
        root / f"{prefix}_shard{shard:02d}_of_{num_shards:02d}.csv"
        for prefix in prefixes
        for shard in range(num_shards)
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return paths


def read_csv_files(paths: Sequence[str | Path]) -> pd.DataFrame:
    """Concatenate declared CSV files with exact round-trip float parsing."""
    files = [Path(path) for path in paths]
    if not files:
        raise ValueError("at least one CSV file is required")
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return pd.concat(
        [pd.read_csv(path, float_precision="round_trip") for path in files],
        ignore_index=True,
    )


def read_expected_shards(
    directory: str | Path,
    prefixes: Iterable[str],
    num_shards: int,
) -> pd.DataFrame:
    """Read a complete declared shard set without wildcard discovery."""
    return read_csv_files(expected_shard_paths(directory, prefixes, num_shards))


def read_checkpoint_parts(
    root: str | Path,
    pattern: str = "part_*.csv",
    *,
    empty_error: str | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
    """Discover sorted checkpoint parts and concatenate them deterministically."""
    paths = sorted(Path(root).glob(pattern))
    if not paths:
        raise FileNotFoundError(empty_error or f"no checkpoint parts under {root}")
    return read_csv_files(paths), paths


def checkpoint_completed_ids(
    directory: str | Path,
    *,
    pattern: str = "part_*.csv",
    column: str = "video_id",
    cast_str: bool = False,
) -> tuple[set[object], list[Path]]:
    """Read only checkpoint identity columns for resumable scoring."""
    paths = sorted(Path(directory).glob(pattern))
    completed: set[object] = set()
    for path in paths:
        values = pd.read_csv(path, usecols=[column])[column]
        if cast_str:
            values = values.astype(str)
        completed.update(values.unique())
    return completed, paths


def checkpoint_completed_keys(
    directory: str | Path,
    key_columns: Sequence[str],
    *,
    pattern: str = "part_*.csv",
) -> tuple[set[tuple[str, ...]], list[Path]]:
    """Read deterministic multi-column identity keys from checkpoint parts."""

    if not key_columns:
        raise ValueError("at least one checkpoint key column is required")
    paths = sorted(Path(directory).glob(pattern))
    completed: set[tuple[str, ...]] = set()
    for path in paths:
        frame = pd.read_csv(
            path,
            usecols=list(key_columns),
            float_precision="round_trip",
        )
        completed.update(
            tuple(str(value) for value in row)
            for row in frame.drop_duplicates().itertuples(index=False, name=None)
        )
    return completed, paths


__all__ = [
    "checkpoint_completed_ids",
    "checkpoint_completed_keys",
    "expected_shard_paths",
    "read_checkpoint_parts",
    "read_csv_files",
    "read_expected_shards",
]
