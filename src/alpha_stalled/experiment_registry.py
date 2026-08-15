"""Validation primitives for the project experiment registry.

The registry is intentionally a small CSV so it remains reviewable in Git. This
module validates identity, lifecycle, lineage, metrics, and evidence paths without
importing the scoring stack.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


ALLOWED_STATUSES = frozenset(
    {
        "main_method_locked",
        "main_baseline",
        "valid_direct_control",
        "valid_ablation",
        "external_confirmation",
        "coverage_extension",
        "diagnostic_only",
        "rejected",
        "superseded",
        "invalid_leakage",
    }
)
REQUIRED_COLUMNS = (
    "experiment_id",
    "protocol_id",
    "parent_experiment_id",
    "experiment_name",
    "git_commit",
    "datasets",
    "evaluation_videos",
    "calibration_real_per_dataset",
    "evaluation_overlap",
    "uses_fake_for_selection",
    "region",
    "aggregation",
    "layer",
    "K",
    "alpha",
    "beta",
    "score_direction",
    "macro_auc",
    "macro_ap",
    "status",
    "paper_status",
    "result_path",
    "report_path",
    "reason",
)
EVIDENCE_REQUIRED_STATUSES = frozenset(
    {
        "main_method_locked",
        "main_baseline",
        "valid_direct_control",
        "valid_ablation",
        "external_confirmation",
        "coverage_extension",
    }
)
IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


@dataclass(frozen=True)
class RegistrySummary:
    row_count: int
    status_counts: Mapping[str, int]
    main_experiment_id: str


def read_registry(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    """Read the CSV as strings and preserve its declared column order."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"experiment registry has no header: {path}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"experiment registry has duplicate columns: {path}")
        rows = list(reader)
        for row_number, row in enumerate(rows, start=2):
            if None in row:
                raise ValueError(f"row {row_number}: more values than declared columns")
        return rows, tuple(reader.fieldnames)


def _parse_metric(value: str, *, row_id: str, field: str) -> float:
    try:
        metric = float(value)
    except ValueError as exc:
        raise ValueError(f"{row_id}: {field} is not numeric: {value!r}") from exc
    if not 0.0 <= metric <= 1.0:
        raise ValueError(f"{row_id}: {field} outside [0, 1]: {metric}")
    return metric


def _parse_nonnegative_int(value: str, *, row_id: str, field: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{row_id}: {field} is not an integer: {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{row_id}: {field} must be nonnegative: {parsed}")
    return parsed


def _validate_lineage(rows_by_id: Mapping[str, Mapping[str, str]]) -> None:
    for row_id, row in rows_by_id.items():
        parent = row["parent_experiment_id"].strip()
        if parent and parent not in rows_by_id:
            raise ValueError(f"{row_id}: unknown parent_experiment_id {parent!r}")
        if parent == row_id:
            raise ValueError(f"{row_id}: experiment cannot be its own parent")

        visited = {row_id}
        while parent:
            if parent in visited:
                raise ValueError(f"{row_id}: cycle in experiment lineage")
            visited.add(parent)
            parent = rows_by_id[parent]["parent_experiment_id"].strip()


def validate_registry(
    rows: Iterable[Mapping[str, str]],
    columns: Iterable[str],
    *,
    repository_root: Path | None = None,
) -> RegistrySummary:
    """Validate a registry and return its compact lifecycle summary."""

    columns = tuple(columns)
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in columns]
    if missing_columns:
        raise ValueError(f"experiment registry missing columns: {missing_columns}")

    rows = list(rows)
    if not rows:
        raise ValueError("experiment registry is empty")

    rows_by_id: dict[str, Mapping[str, str]] = {}
    names: set[str] = set()
    statuses: Counter[str] = Counter()
    for row_number, row in enumerate(rows, start=2):
        missing_values = [name for name in REQUIRED_COLUMNS if row.get(name) is None]
        if missing_values:
            raise ValueError(f"row {row_number}: missing values for {missing_values}")
        row_id = row["experiment_id"].strip()
        if not IDENTIFIER.fullmatch(row_id):
            raise ValueError(f"row {row_number}: invalid experiment_id {row_id!r}")
        if row_id in rows_by_id:
            raise ValueError(f"duplicate experiment_id: {row_id}")
        rows_by_id[row_id] = row

        protocol_id = row["protocol_id"].strip()
        if not IDENTIFIER.fullmatch(protocol_id):
            raise ValueError(f"{row_id}: invalid protocol_id {protocol_id!r}")

        name = row["experiment_name"].strip()
        if not name:
            raise ValueError(f"{row_id}: experiment_name is empty")
        if name in names:
            raise ValueError(f"duplicate experiment_name: {name}")
        names.add(name)

        status = row["status"].strip()
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"{row_id}: unsupported status {status!r}")
        statuses[status] += 1

        _parse_nonnegative_int(
            row["evaluation_videos"].strip(),
            row_id=row_id,
            field="evaluation_videos",
        )
        if not row["datasets"].strip():
            raise ValueError(f"{row_id}: datasets is empty")
        if not row["calibration_real_per_dataset"].strip():
            raise ValueError(f"{row_id}: calibration_real_per_dataset is empty")
        if row["uses_fake_for_selection"].strip().lower() not in {"true", "false"}:
            raise ValueError(f"{row_id}: uses_fake_for_selection must be true or false")
        if row["score_direction"].strip() != "higher_is_real":
            raise ValueError(f"{row_id}: unsupported score_direction")
        _parse_metric(row["macro_auc"].strip(), row_id=row_id, field="macro_auc")
        _parse_metric(row["macro_ap"].strip(), row_id=row_id, field="macro_ap")

        overlap = row["evaluation_overlap"].strip()
        if status != "invalid_leakage" and overlap != "0":
            raise ValueError(f"{row_id}: valid result must declare zero evaluation overlap")
        if status == "invalid_leakage" and overlap == "0":
            raise ValueError(f"{row_id}: invalid_leakage must declare positive/unknown overlap")

        for field in ("result_path", "report_path"):
            raw_path = row[field].strip()
            if status in EVIDENCE_REQUIRED_STATUSES and not raw_path:
                raise ValueError(f"{row_id}: {field} is required for status {status}")
            if raw_path and repository_root is not None:
                path = Path(raw_path)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"{row_id}: {field} must be repository-relative")
                if not (repository_root / path).is_file():
                    raise ValueError(f"{row_id}: missing {field}: {raw_path}")

        if not row["reason"].strip():
            raise ValueError(f"{row_id}: reason is empty")

    _validate_lineage(rows_by_id)

    main_ids = [
        row_id
        for row_id, row in rows_by_id.items()
        if row["status"].strip() == "main_method_locked"
    ]
    if len(main_ids) != 1:
        raise ValueError(f"expected exactly one locked main method, found {main_ids}")
    return RegistrySummary(
        row_count=len(rows),
        status_counts=dict(sorted(statuses.items())),
        main_experiment_id=main_ids[0],
    )


__all__ = [
    "ALLOWED_STATUSES",
    "EVIDENCE_REQUIRED_STATUSES",
    "IDENTIFIER",
    "REQUIRED_COLUMNS",
    "RegistrySummary",
    "read_registry",
    "validate_registry",
]
