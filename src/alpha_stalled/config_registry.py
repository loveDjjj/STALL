"""Configuration-asset identity and lifecycle governance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


SCHEMA_VERSION = "alpha_stalled_config_registry_v1"
ALLOWED_LIFECYCLES = {
    "coverage_extension",
    "current_locked",
    "governance",
    "historical_frozen",
    "release_evidence",
    "release_input",
    "template",
}
ALLOWED_AUTHORITIES = {"current", "extension", "historical", "supporting", "template"}
CONFIG_SUFFIXES = {".csv", ".json", ".template", ".yaml", ".yml"}
CURRENT_CONFIG = "configs/alpha_stalled_u0_locked.yaml"


@dataclass(frozen=True)
class ConfigRegistrySummary:
    asset_count: int
    protocol_config_count: int
    historical_config_count: int
    current_path: str


def read_config_registry(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config registry must be a mapping")
    return payload


def discover_config_assets(config_root: Path) -> set[str]:
    repository_root = config_root.parent
    return {
        path.relative_to(repository_root).as_posix()
        for path in config_root.rglob("*")
        if path.is_file()
        and path.name != "README.md"
        and path.suffix.lower() in CONFIG_SUFFIXES
    }


def _required(mapping: Mapping[str, Any], fields: tuple[str, ...], context: str) -> None:
    missing = [field for field in fields if field not in mapping]
    if missing:
        raise ValueError(f"{context} missing fields: {missing}")


def _protocol_identity(path: Path) -> tuple[str | None, str | None, bool | None]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"protocol config must be a mapping: {path}")
    identity = payload.get("config_identity")
    if isinstance(identity, Mapping):
        return (
            str(identity.get("protocol_id", "")) or None,
            str(identity.get("lifecycle", "")) or None,
            identity.get("current_authority")
            if isinstance(identity.get("current_authority"), bool)
            else None,
        )
    release = payload.get("release")
    if isinstance(release, Mapping):
        return str(release.get("protocol_version", "")) or None, None, None
    return None, None, None


def validate_config_registry(
    payload: Mapping[str, Any], repository_root: Path
) -> ConfigRegistrySummary:
    _required(payload, ("schema_version", "lifecycle_classes", "entries"), "registry")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported config registry schema: {payload['schema_version']!r}")
    lifecycle_classes = payload["lifecycle_classes"]
    if not isinstance(lifecycle_classes, list) or set(lifecycle_classes) != ALLOWED_LIFECYCLES:
        raise ValueError("lifecycle_classes must declare every allowed lifecycle exactly once")
    if len(lifecycle_classes) != len(set(lifecycle_classes)):
        raise ValueError("lifecycle_classes contains duplicates")

    entries = payload["entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("entries must be a nonempty list")
    declared: dict[str, Mapping[str, Any]] = {}
    protocol_ids: dict[str, str] = {}
    current_paths: list[str] = []
    for index, entry in enumerate(entries):
        context = f"entries[{index}]"
        if not isinstance(entry, Mapping):
            raise ValueError(f"{context} must be a mapping")
        _required(
            entry,
            ("path", "kind", "lifecycle", "authority", "protocol_id", "mutable"),
            context,
        )
        relative_text = entry["path"]
        if not isinstance(relative_text, str) or not relative_text:
            raise ValueError(f"{context}.path must be nonempty")
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("configs",):
            raise ValueError(f"{context}.path must stay under configs/")
        if relative_text in declared:
            raise ValueError(f"duplicate config registry path: {relative_text}")
        if not (repository_root / relative).is_file():
            raise ValueError(f"registered config asset is missing: {relative_text}")
        declared[relative_text] = entry

        kind = entry["kind"]
        lifecycle = entry["lifecycle"]
        authority = entry["authority"]
        protocol_id = entry["protocol_id"]
        if not isinstance(kind, str) or not kind:
            raise ValueError(f"{context}.kind must be nonempty")
        if lifecycle not in ALLOWED_LIFECYCLES:
            raise ValueError(f"{context}.lifecycle is invalid: {lifecycle!r}")
        if authority not in ALLOWED_AUTHORITIES:
            raise ValueError(f"{context}.authority is invalid: {authority!r}")
        if protocol_id is not None and (not isinstance(protocol_id, str) or not protocol_id):
            raise ValueError(f"{context}.protocol_id must be null or nonempty")
        if not isinstance(entry["mutable"], bool):
            raise ValueError(f"{context}.mutable must be boolean")
        if authority == "current":
            current_paths.append(relative_text)
            if lifecycle != "current_locked" or entry["mutable"]:
                raise ValueError("current config must be current_locked and immutable")

        if kind == "protocol_config":
            if protocol_id is None:
                raise ValueError(f"{context} protocol config has no protocol_id")
            if protocol_id in protocol_ids:
                raise ValueError(
                    f"duplicate protocol config id {protocol_id!r}: "
                    f"{protocol_ids[protocol_id]} and {relative_text}"
                )
            protocol_ids[protocol_id] = relative_text
            actual_id, actual_lifecycle, actual_current = _protocol_identity(
                repository_root / relative
            )
            if actual_id != protocol_id:
                raise ValueError(
                    f"protocol identity mismatch for {relative_text}: "
                    f"registry={protocol_id!r} file={actual_id!r}"
                )
            if actual_lifecycle is not None and actual_lifecycle != lifecycle:
                raise ValueError(f"lifecycle mismatch for {relative_text}")
            if actual_current is not None and actual_current != (authority == "current"):
                raise ValueError(f"current_authority mismatch for {relative_text}")

    discovered = discover_config_assets(repository_root / "configs")
    missing = sorted(discovered.difference(declared))
    stale = sorted(set(declared).difference(discovered))
    if missing or stale:
        raise ValueError(f"config registry coverage mismatch: missing={missing} stale={stale}")
    if current_paths != [CURRENT_CONFIG]:
        raise ValueError(
            f"exactly one current config is required and must be {CURRENT_CONFIG}: "
            f"{current_paths}"
        )

    unknown_protocol_refs = sorted(
        {
            str(entry["protocol_id"])
            for entry in entries
            if entry["protocol_id"] is not None
        }.difference(protocol_ids)
    )
    if unknown_protocol_refs:
        raise ValueError(f"entries reference undeclared protocol configs: {unknown_protocol_refs}")

    return ConfigRegistrySummary(
        asset_count=len(entries),
        protocol_config_count=len(protocol_ids),
        historical_config_count=sum(
            entry["lifecycle"] == "historical_frozen" for entry in entries
        ),
        current_path=current_paths[0],
    )
