"""Cache inventory, lifecycle policy, and non-destructive layout auditing."""

from __future__ import annotations

import copy
import hashlib
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .experiment_registry import IDENTIFIER
from .release_io import sha256_file


SCHEMA_VERSION = "alpha_stalled_cache_inventory_v1"
CLEANUP_PRIORITIES = (
    "P0_keep",
    "P1_keep_active",
    "P2_review",
    "P3_safe_delete_candidate",
)
RETENTION_CLASSES = frozenset(
    {"keep", "keep_active", "review_before_delete", "safe_delete_candidate"}
)
LIFECYCLES = frozenset(
    {"protocol_index", "feature_cache", "external_method_frames", "debug_benchmark"}
)
METADATA_STATUSES = frozenset({"self_describing", "partial", "external_only", "missing"})
SHA256_HEX_LENGTH = 64


@dataclass(frozen=True)
class CacheInventorySummary:
    snapshot_id: str
    group_count: int
    file_count: int
    logical_bytes: int
    safe_delete_candidate_bytes: int


def _required(mapping: Mapping[str, Any], fields: Iterable[str], context: str) -> None:
    missing = [field for field in fields if field not in mapping]
    if missing:
        raise ValueError(f"{context} missing fields: {missing}")


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a nonempty string")
    return value.strip()


def _strings(value: Any, context: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{context} must be a list")
    for index, item in enumerate(value):
        _string(item, f"{context}[{index}]")
    return value


def _nonnegative_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a nonnegative integer")
    return value


def _sha256(value: Any, context: str) -> str:
    text = _string(value, context)
    if len(text) != SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return text


def _count_mapping(value: Any, context: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    normalized: dict[str, int] = {}
    for key, count in value.items():
        normalized[_string(key, f"{context} key")] = _nonnegative_integer(
            count, f"{context}.{key}"
        )
    return normalized


def _relative_directory(value: Any, repository_root: Path, context: str) -> Path:
    relative = Path(_string(value, context))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{context} must be repository-relative")
    path = repository_root / relative
    if not path.is_dir():
        raise ValueError(f"{context} is not a directory: {relative}")
    try:
        path.resolve().relative_to(repository_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{context} resolves outside the repository") from exc
    return path


def _validate_spec(spec: Mapping[str, Any], repository_root: Path) -> None:
    _required(spec, ("schema_version", "snapshot_id", "policy", "groups"), "inventory")
    if spec["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported cache inventory schema: {spec['schema_version']!r}")
    snapshot_id = _string(spec["snapshot_id"], "snapshot_id")
    if not IDENTIFIER.fullmatch(snapshot_id):
        raise ValueError("snapshot_id must be a stable lowercase identifier")

    policy = spec["policy"]
    if not isinstance(policy, Mapping):
        raise ValueError("policy must be an object")
    _required(policy, ("cache_root", "deletion_mode"), "policy")
    _relative_directory(policy["cache_root"], repository_root, "policy.cache_root")
    if policy["deletion_mode"] != "manual_approval_only":
        raise ValueError("policy.deletion_mode must be manual_approval_only")

    groups = spec["groups"]
    if not isinstance(groups, list) or not groups:
        raise ValueError("groups must be a nonempty list")
    ids: set[str] = set()
    paths: list[tuple[str, Path]] = []
    for index, group in enumerate(groups):
        context = f"groups[{index}]"
        if not isinstance(group, Mapping):
            raise ValueError(f"{context} must be an object")
        _required(
            group,
            (
                "cache_id",
                "path",
                "family",
                "dataset_scope",
                "purpose",
                "lifecycle",
                "retention",
                "cleanup_priority",
                "release_dependency",
                "rebuildable",
                "producer",
                "rebuild_command",
                "current_consumers",
                "metadata_status",
                "payload_contract",
                "cache_key",
                "hash_contents",
                "notes",
            ),
            context,
        )
        cache_id = _string(group["cache_id"], f"{context}.cache_id")
        if not IDENTIFIER.fullmatch(cache_id) or cache_id in ids:
            raise ValueError(f"{context}.cache_id is invalid or duplicated: {cache_id}")
        ids.add(cache_id)
        path = _relative_directory(group["path"], repository_root, f"{context}.path")
        paths.append((cache_id, path.resolve()))
        for field in ("family", "dataset_scope", "purpose", "producer", "payload_contract", "notes"):
            _string(group[field], f"{context}.{field}")
        if group["lifecycle"] not in LIFECYCLES:
            raise ValueError(f"{context}.lifecycle is unsupported")
        if group["retention"] not in RETENTION_CLASSES:
            raise ValueError(f"{context}.retention is unsupported")
        if group["cleanup_priority"] not in CLEANUP_PRIORITIES:
            raise ValueError(f"{context}.cleanup_priority is unsupported")
        if group["metadata_status"] not in METADATA_STATUSES:
            raise ValueError(f"{context}.metadata_status is unsupported")
        for field in ("release_dependency", "rebuildable", "hash_contents"):
            if not isinstance(group[field], bool):
                raise ValueError(f"{context}.{field} must be boolean")
        command = group["rebuild_command"]
        if command is not None and not isinstance(command, str):
            raise ValueError(f"{context}.rebuild_command must be string or null")
        if group["rebuildable"] and not (isinstance(command, str) and command.strip()):
            raise ValueError(f"{context}: rebuildable cache requires rebuild_command")
        _strings(group["current_consumers"], f"{context}.current_consumers", allow_empty=True)
        cache_key = group["cache_key"]
        if not isinstance(cache_key, Mapping):
            raise ValueError(f"{context}.cache_key must be an object")
        _required(cache_key, ("present", "missing"), f"{context}.cache_key")
        _strings(cache_key["present"], f"{context}.cache_key.present", allow_empty=True)
        _strings(cache_key["missing"], f"{context}.cache_key.missing", allow_empty=True)

        if group["cleanup_priority"] == "P3_safe_delete_candidate":
            if group["retention"] != "safe_delete_candidate" or group["release_dependency"]:
                raise ValueError(f"{context}: P3 candidate must be non-release safe_delete_candidate")

    for index, (left_id, left) in enumerate(paths):
        for right_id, right in paths[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError(f"cache group paths overlap: {left_id} and {right_id}")


def _iter_files(root: Path) -> tuple[list[Path], list[Path]]:
    files: list[Path] = []
    symlinks: list[Path] = []
    stack = [root]
    while stack:
        directory = stack.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                if entry.is_symlink():
                    symlinks.append(path)
                elif entry.is_dir(follow_symlinks=False):
                    stack.append(path)
                elif entry.is_file(follow_symlinks=False):
                    files.append(path)
    files.sort(key=lambda path: path.as_posix())
    symlinks.sort(key=lambda path: path.as_posix())
    return files, symlinks


def _pt_naming_class(name: str) -> str:
    for duration in (1, 2, 3, 4):
        if name.endswith(f"_{duration}s.pt"):
            return f"compact_{duration}s"
    return "unsuffixed_pt"


def _scan_group(
    group: Mapping[str, Any], repository_root: Path
) -> tuple[dict[str, Any], set[str]]:
    root = repository_root / group["path"]
    files, symlinks = _iter_files(root)
    if symlinks:
        examples = [str(path.relative_to(repository_root)) for path in symlinks[:5]]
        raise ValueError(f"cache group contains symlinks: {group['cache_id']} {examples}")

    layout = hashlib.sha256()
    content = hashlib.sha256() if group["hash_contents"] else None
    extensions: Counter[str] = Counter()
    naming: Counter[str] = Counter()
    logical_bytes = 0
    seen: set[str] = set()
    for path in files:
        repository_relative = path.relative_to(repository_root).as_posix()
        group_relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        logical_bytes += size
        seen.add(repository_relative)
        extension = path.suffix.lower() or "[none]"
        extensions[extension] += 1
        if extension == ".pt":
            naming[_pt_naming_class(path.name)] += 1
        layout.update(group_relative.encode("utf-8"))
        layout.update(b"\0")
        layout.update(str(size).encode("ascii"))
        layout.update(b"\n")
        if content is not None:
            content.update(group_relative.encode("utf-8"))
            content.update(b"\0")
            content.update(sha256_file(path).encode("ascii"))
            content.update(b"\n")

    stats: dict[str, Any] = {
        "file_count": len(files),
        "logical_bytes": logical_bytes,
        "extension_counts": dict(sorted(extensions.items())),
        "pt_naming_counts": dict(sorted(naming.items())),
        "layout_sha256": layout.hexdigest(),
        "content_sha256": content.hexdigest() if content is not None else None,
    }
    return stats, seen


def _build(spec: Mapping[str, Any], repository_root: Path) -> dict[str, Any]:
    inventory = copy.deepcopy(dict(spec))
    cache_root = repository_root / inventory["policy"]["cache_root"]
    registered_files: set[str] = set()
    priority_files: Counter[str] = Counter()
    priority_bytes: Counter[str] = Counter()
    for group in inventory["groups"]:
        stats, seen = _scan_group(group, repository_root)
        group["stats"] = stats
        registered_files.update(seen)
        priority = group["cleanup_priority"]
        priority_files[priority] += stats["file_count"]
        priority_bytes[priority] += stats["logical_bytes"]

    all_files, symlinks = _iter_files(cache_root)
    if symlinks:
        examples = [str(path.relative_to(repository_root)) for path in symlinks[:5]]
        raise ValueError(f"cache root contains unregistered symlinks: {examples}")
    all_relative = {path.relative_to(repository_root).as_posix() for path in all_files}
    unregistered = sorted(all_relative - registered_files)
    multiply_registered = len(registered_files) - sum(
        group["stats"]["file_count"] for group in inventory["groups"]
    )
    if multiply_registered:
        raise ValueError("cache files are registered by overlapping groups")
    if unregistered:
        raise ValueError(
            f"unregistered cache files: count={len(unregistered)} examples={unregistered[:10]}"
        )

    inventory["summary"] = {
        "group_count": len(inventory["groups"]),
        "file_count": len(all_files),
        "logical_bytes": sum(path.stat().st_size for path in all_files),
        "registered_file_count": len(registered_files),
        "unregistered_file_count": 0,
        "unregistered_file_examples": [],
        "by_cleanup_priority": {
            priority: {
                "file_count": int(priority_files[priority]),
                "logical_bytes": int(priority_bytes[priority]),
            }
            for priority in CLEANUP_PRIORITIES
        },
    }
    return inventory


def build_cache_inventory(spec: Mapping[str, Any], repository_root: Path) -> dict[str, Any]:
    """Scan all declared groups and fail if any cache file is unregistered."""

    _validate_spec(spec, repository_root)
    inventory = _build(spec, repository_root)
    validate_cache_inventory(inventory, repository_root=repository_root, verify_layout=False)
    return inventory


def validate_cache_inventory(
    inventory: Mapping[str, Any],
    *,
    repository_root: Path,
    verify_layout: bool = True,
) -> CacheInventorySummary:
    """Validate lifecycle declarations and optionally rescan current cache layout."""

    _validate_spec(inventory, repository_root)
    _required(inventory, ("summary",), "inventory")
    summary = inventory["summary"]
    if not isinstance(summary, Mapping):
        raise ValueError("summary must be an object")
    _required(
        summary,
        (
            "group_count",
            "file_count",
            "logical_bytes",
            "registered_file_count",
            "unregistered_file_count",
            "unregistered_file_examples",
            "by_cleanup_priority",
        ),
        "summary",
    )
    for index, group in enumerate(inventory["groups"]):
        _required(group, ("stats",), f"groups[{index}]")
        stats = group["stats"]
        if not isinstance(stats, Mapping):
            raise ValueError(f"groups[{index}].stats must be an object")
        _required(
            stats,
            (
                "file_count",
                "logical_bytes",
                "extension_counts",
                "pt_naming_counts",
                "layout_sha256",
                "content_sha256",
            ),
            f"groups[{index}].stats",
        )
        file_count = _nonnegative_integer(
            stats["file_count"], f"groups[{index}].stats.file_count"
        )
        _nonnegative_integer(
            stats["logical_bytes"], f"groups[{index}].stats.logical_bytes"
        )
        extensions = _count_mapping(
            stats["extension_counts"], f"groups[{index}].stats.extension_counts"
        )
        naming = _count_mapping(
            stats["pt_naming_counts"], f"groups[{index}].stats.pt_naming_counts"
        )
        if sum(extensions.values()) != file_count:
            raise ValueError(f"groups[{index}]: extension counts do not sum to file count")
        if sum(naming.values()) > file_count:
            raise ValueError(f"groups[{index}]: PT naming counts exceed file count")
        _sha256(stats["layout_sha256"], f"groups[{index}].stats.layout_sha256")
        content_sha = stats["content_sha256"]
        if group["hash_contents"]:
            _sha256(content_sha, f"groups[{index}].stats.content_sha256")
        elif content_sha is not None:
            raise ValueError(
                f"groups[{index}].stats.content_sha256 must be null when disabled"
            )

    group_count = _nonnegative_integer(summary["group_count"], "summary.group_count")
    file_count = _nonnegative_integer(summary["file_count"], "summary.file_count")
    logical_bytes = _nonnegative_integer(
        summary["logical_bytes"], "summary.logical_bytes"
    )
    registered = _nonnegative_integer(
        summary["registered_file_count"], "summary.registered_file_count"
    )
    unregistered = _nonnegative_integer(
        summary["unregistered_file_count"], "summary.unregistered_file_count"
    )
    examples = summary["unregistered_file_examples"]
    _strings(examples, "summary.unregistered_file_examples", allow_empty=True)
    if group_count != len(inventory["groups"]):
        raise ValueError("summary.group_count does not match declared groups")
    if registered + unregistered != file_count:
        raise ValueError("summary registered/unregistered counts do not match file_count")
    if unregistered == 0 and examples:
        raise ValueError("summary has unregistered examples but zero unregistered files")

    priorities = summary["by_cleanup_priority"]
    if not isinstance(priorities, Mapping) or set(priorities) != set(CLEANUP_PRIORITIES):
        raise ValueError("summary.by_cleanup_priority has unexpected keys")
    expected_files: Counter[str] = Counter()
    expected_bytes: Counter[str] = Counter()
    for group in inventory["groups"]:
        priority = group["cleanup_priority"]
        expected_files[priority] += group["stats"]["file_count"]
        expected_bytes[priority] += group["stats"]["logical_bytes"]
    for priority in CLEANUP_PRIORITIES:
        item = priorities[priority]
        if not isinstance(item, Mapping):
            raise ValueError(f"summary.by_cleanup_priority.{priority} must be an object")
        _required(item, ("file_count", "logical_bytes"), priority)
        priority_files = _nonnegative_integer(
            item["file_count"], f"summary.by_cleanup_priority.{priority}.file_count"
        )
        priority_bytes = _nonnegative_integer(
            item["logical_bytes"],
            f"summary.by_cleanup_priority.{priority}.logical_bytes",
        )
        if priority_files != expected_files[priority] or priority_bytes != expected_bytes[priority]:
            raise ValueError(f"summary.by_cleanup_priority.{priority} is inconsistent")
    if sum(expected_files.values()) != registered:
        raise ValueError("group file counts do not match registered_file_count")
    if sum(expected_bytes.values()) != logical_bytes:
        raise ValueError("group logical bytes do not match summary.logical_bytes")
    if verify_layout:
        declared = copy.deepcopy(dict(inventory))
        declared.pop("summary", None)
        for group in declared["groups"]:
            group.pop("stats", None)
        rebuilt = _build(declared, repository_root)
        if rebuilt["summary"] != inventory["summary"]:
            raise ValueError("cache inventory summary is stale")
        current_groups = {group["cache_id"]: group["stats"] for group in rebuilt["groups"]}
        stored_groups = {group["cache_id"]: group["stats"] for group in inventory["groups"]}
        if current_groups != stored_groups:
            raise ValueError("cache inventory group stats are stale")

    safe_bytes = int(
        summary["by_cleanup_priority"]["P3_safe_delete_candidate"]["logical_bytes"]
    )
    return CacheInventorySummary(
        snapshot_id=inventory["snapshot_id"],
        group_count=int(summary["group_count"]),
        file_count=int(summary["file_count"]),
        logical_bytes=int(summary["logical_bytes"]),
        safe_delete_candidate_bytes=safe_bytes,
    )


def render_cache_inventory(inventory: Mapping[str, Any]) -> str:
    """Render a deterministic human-readable inventory and cleanup decision table."""

    summary = inventory["summary"]
    gib = 1024**3
    lines = [
        "# Cache inventory and retention audit",
        "",
        f"Snapshot: `{inventory['snapshot_id']}`. This report is non-destructive; "
        "deletion requires explicit manual approval.",
        "",
        f"Registered groups/files: {summary['group_count']}/{summary['file_count']}. "
        f"Logical size: {summary['logical_bytes'] / gib:.2f} GiB. "
        "Filesystem `du` may differ because it reports allocated blocks.",
        "",
        "| Cache group | Path | Files | GiB | Lifecycle | Priority | Metadata |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for group in inventory["groups"]:
        stats = group["stats"]
        lines.append(
            f"| `{group['cache_id']}` | `{group['path']}` | {stats['file_count']:,} | "
            f"{stats['logical_bytes'] / gib:.2f} | `{group['lifecycle']}` | "
            f"`{group['cleanup_priority']}` | `{group['metadata_status']}` |"
        )

    lines.extend(["", "## Cleanup classes", ""])
    for priority in CLEANUP_PRIORITIES:
        item = summary["by_cleanup_priority"][priority]
        lines.append(
            f"- `{priority}`: {item['file_count']:,} files, "
            f"{item['logical_bytes'] / gib:.2f} GiB."
        )

    lines.extend(
        [
            "",
            "## Key findings",
            "",
            "- No registered cache is required to validate the immutable locked U0 release; "
            "the release retains manifests, raw-shard hashes, parameters, and final scores.",
            "- Existing feature cache filenames do not encode checkpoint SHA, DINO commit, "
            "layer, preprocessing, frame grouping, or numerical implementation.",
            "- The current snapshot predates the strict feature-cache contract and remains "
            "legacy. New empty roots created by contract-aware producers receive immutable "
            "root and per-entry metadata; existing roots are never backfilled automatically.",
            "- Patch payloads retain frame indices and grid size, but that is not a complete "
            "cache key. Shape compatibility alone does not authorize reuse.",
            "- `P3_safe_delete_candidate` means protocol-independent evidence says the group "
            "is disposable; this report does not delete it.",
            "",
            "## Group decisions",
            "",
        ]
    )
    for group in inventory["groups"]:
        missing = ", ".join(group["cache_key"]["missing"]) or "none"
        lines.extend(
            [
                f"### `{group['cache_id']}`",
                "",
                f"- Purpose: {group['purpose']}",
                f"- Producer: `{group['producer']}`",
                f"- Retention: `{group['retention']}`; release dependency: "
                f"`{str(group['release_dependency']).lower()}`.",
                f"- Missing cache-key fields: {missing}.",
                f"- Decision note: {group['notes']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "CLEANUP_PRIORITIES",
    "LIFECYCLES",
    "METADATA_STATUSES",
    "RETENTION_CLASSES",
    "SCHEMA_VERSION",
    "CacheInventorySummary",
    "build_cache_inventory",
    "render_cache_inventory",
    "validate_cache_inventory",
]
