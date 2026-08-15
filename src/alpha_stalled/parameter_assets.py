"""Identity, schema, and lifecycle validation for frozen parameter assets."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


SCHEMA_VERSION = "alpha_stalled_parameter_assets_v1"
ALLOWED_FAMILIES = {"global_stall", "local_patch"}
ALLOWED_LIFECYCLES = {
    "current_release",
    "external_confirmation",
    "historical_frozen",
}


@dataclass(frozen=True)
class ParameterAssetSummary:
    governed_asset_count: int
    current_release_count: int
    external_confirmation_count: int
    historical_frozen_count: int
    local_unregistered_count: int
    local_unregistered_bytes: int


def read_parameter_assets(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("parameter asset catalog must be a mapping")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_fields(
    mapping: Mapping[str, Any], fields: tuple[str, ...], context: str
) -> None:
    missing = [field for field in fields if field not in mapping]
    if missing:
        raise ValueError(f"{context} missing fields: {missing}")


def _array_contract(
    data: np.lib.npyio.NpzFile,
    name: str,
    shape: tuple[int, ...],
    dtype: str,
    context: str,
) -> None:
    if name not in data.files:
        raise ValueError(f"{context} missing NPZ array: {name}")
    array = data[name]
    if array.shape != shape or str(array.dtype) != dtype:
        raise ValueError(
            f"{context} array contract mismatch for {name}: "
            f"shape={array.shape} dtype={array.dtype}, expected={shape}/{dtype}"
        )


def _validate_global_asset(data: np.lib.npyio.NpzFile, entry: Mapping[str, Any]) -> None:
    context = str(entry["path"])
    count = int(entry["calibration_count"])
    spatial_rank = int(entry["spatial_rank"])
    temporal_rank = int(entry["temporal_rank"])
    expected_names = {
        "mu_spat",
        "W_spat",
        "calib_ll_spat",
        "mu_temp",
        "W_temp",
        "calib_ll_temp",
    }
    if set(data.files) != expected_names:
        raise ValueError(f"{context} unexpected NPZ arrays: {sorted(data.files)}")
    _array_contract(data, "mu_spat", (1024,), "float32", context)
    _array_contract(data, "W_spat", (1024, spatial_rank), "float32", context)
    _array_contract(data, "calib_ll_spat", (count, 16), "float64", context)
    _array_contract(data, "mu_temp", (1024,), "float32", context)
    _array_contract(data, "W_temp", (1024, temporal_rank), "float32", context)
    _array_contract(data, "calib_ll_temp", (count, 15), "float64", context)


def _validate_local_asset(data: np.lib.npyio.NpzFile, entry: Mapping[str, Any]) -> None:
    context = str(entry["path"])
    count = int(entry["calibration_count"])
    spatial_rank = int(entry["spatial_rank"])
    temporal_rank = int(entry["temporal_rank"])
    expected_names = {
        "mu_patch_spat",
        "W_patch_spat",
        "calib_patch_spat_scores",
        "mu_patch_temp",
        "W_patch_temp",
        "calib_patch_temp_scores",
        "patch_grid_size",
        "duration",
        "aggregation_config",
    }
    if set(data.files) != expected_names:
        raise ValueError(f"{context} unexpected NPZ arrays: {sorted(data.files)}")
    _array_contract(data, "mu_patch_spat", (1024,), "float32", context)
    _array_contract(data, "W_patch_spat", (1024, spatial_rank), "float32", context)
    _array_contract(data, "calib_patch_spat_scores", (count,), "float32", context)
    _array_contract(data, "mu_patch_temp", (1024,), "float32", context)
    _array_contract(data, "W_patch_temp", (1024, temporal_rank), "float32", context)
    _array_contract(data, "calib_patch_temp_scores", (count,), "float32", context)
    _array_contract(data, "patch_grid_size", (2,), "int32", context)
    _array_contract(data, "duration", (1,), "int32", context)
    aggregation_raw = data["aggregation_config"]
    if aggregation_raw.shape != () or aggregation_raw.dtype.kind != "U":
        raise ValueError(f"{context} aggregation_config must be a scalar string")
    aggregation = json.loads(str(aggregation_raw.item()))
    expected_aggregation = entry.get("aggregation")
    if not isinstance(expected_aggregation, Mapping) or not expected_aggregation:
        raise ValueError(f"{context} must declare aggregation semantics")
    mismatches = {
        key: (aggregation.get(key), expected)
        for key, expected in expected_aggregation.items()
        if aggregation.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"{context} aggregation semantics mismatch: {mismatches}")


def _tracked_parameter_paths(repository_root: Path) -> set[str]:
    completed = subprocess.run(
        ["git", "-C", str(repository_root), "ls-files", "--", "precomputed", "release"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip().endswith(".npz")
    }


def _validate_locked_u0_references(
    entries: list[Mapping[str, Any]], repository_root: Path
) -> None:
    config_path = repository_root / "configs/alpha_stalled_u0_locked.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected = {
        str(config["global_branch"]["params"]): str(
            config["global_branch"]["params_sha256"]
        )
    }
    expected.update(
        {
            str(spec["path"]): str(spec["sha256"])
            for spec in config["local_branch"]["params_by_dataset"].values()
        }
    )
    declared = {
        str(entry["path"]): str(entry["sha256"])
        for entry in entries
        if entry["lifecycle"] == "current_release"
    }
    if declared != expected:
        raise ValueError(
            f"current parameter assets do not match locked U0: {declared} != {expected}"
        )

    hash_index = json.loads(
        (repository_root / "release/u0/config_and_checkpoint_hashes.json").read_text(
            encoding="utf-8"
        )
    )
    indexed = {
        str(record["path"]): str(record["sha256"])
        for name, record in hash_index["input_files"].items()
        if name == "global_params" or name.startswith("local_params_")
    }
    if declared != indexed:
        raise ValueError("parameter catalog and locked U0 hash index disagree")


def validate_parameter_assets(
    payload: Mapping[str, Any],
    repository_root: Path,
    *,
    verify_git_tracking: bool = True,
    verify_locked_u0: bool = True,
) -> ParameterAssetSummary:
    _require_fields(
        payload,
        ("schema_version", "local_precomputed_policy", "entries"),
        "catalog",
    )
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported parameter catalog schema: {payload['schema_version']!r}")
    policy = payload["local_precomputed_policy"]
    if not isinstance(policy, Mapping):
        raise ValueError("local_precomputed_policy must be a mapping")
    _require_fields(policy, ("root", "allowed_unregistered_globs"), "local policy")
    local_root_relative = Path(str(policy["root"]))
    allowed_globs = policy["allowed_unregistered_globs"]
    if (
        local_root_relative.is_absolute()
        or ".." in local_root_relative.parts
        or not isinstance(allowed_globs, list)
        or not allowed_globs
        or not all(isinstance(pattern, str) and pattern for pattern in allowed_globs)
    ):
        raise ValueError("invalid local precomputed policy")

    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("parameter entries must be a nonempty list")
    entries: list[Mapping[str, Any]] = []
    declared: dict[str, Mapping[str, Any]] = {}
    for index, entry in enumerate(raw_entries):
        context = f"entries[{index}]"
        if not isinstance(entry, Mapping):
            raise ValueError(f"{context} must be a mapping")
        _require_fields(
            entry,
            (
                "path",
                "family",
                "lifecycle",
                "role",
                "protocol_id",
                "dataset",
                "bytes",
                "sha256",
                "calibration_count",
                "spatial_rank",
                "temporal_rank",
            ),
            context,
        )
        relative_text = entry["path"]
        if not isinstance(relative_text, str) or not relative_text:
            raise ValueError(f"{context}.path must be nonempty")
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{context}.path must be repository-relative")
        if relative_text in declared:
            raise ValueError(f"duplicate parameter asset path: {relative_text}")
        if entry["family"] not in ALLOWED_FAMILIES:
            raise ValueError(f"{context}.family is invalid")
        if entry["lifecycle"] not in ALLOWED_LIFECYCLES:
            raise ValueError(f"{context}.lifecycle is invalid")
        path = repository_root / relative
        if not path.is_file():
            raise ValueError(f"parameter asset is missing: {relative_text}")
        if path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"parameter byte count mismatch: {relative_text}")
        if _sha256(path) != entry["sha256"]:
            raise ValueError(f"parameter SHA-256 mismatch: {relative_text}")
        with np.load(path, allow_pickle=False) as data:
            if entry["family"] == "global_stall":
                _validate_global_asset(data, entry)
            else:
                _validate_local_asset(data, entry)
        declared[relative_text] = entry
        entries.append(entry)

    release_params = {
        path.relative_to(repository_root).as_posix()
        for path in (repository_root / "release").glob("*/params/*.npz")
    }
    declared_release_params = {
        path for path in declared if path.startswith("release/")
    }
    if release_params != declared_release_params:
        raise ValueError(
            "release parameter coverage mismatch: "
            f"missing={sorted(release_params - declared_release_params)} "
            f"stale={sorted(declared_release_params - release_params)}"
        )

    local_root = repository_root / local_root_relative
    local_files = sorted(local_root.glob("*.npz"))
    unregistered = [
        path
        for path in local_files
        if path.relative_to(repository_root).as_posix() not in declared
    ]
    invalid_local = [
        path.name
        for path in unregistered
        if not any(fnmatch(path.name, pattern) for pattern in allowed_globs)
    ]
    if invalid_local:
        raise ValueError(f"unregistered precomputed assets are not allowed: {invalid_local}")

    if verify_git_tracking:
        tracked = _tracked_parameter_paths(repository_root)
        expected_tracked = set(declared)
        if tracked != expected_tracked:
            raise ValueError(
                "tracked parameter coverage mismatch: "
                f"unregistered={sorted(tracked - expected_tracked)} "
                f"untracked={sorted(expected_tracked - tracked)}"
            )
    if verify_locked_u0:
        _validate_locked_u0_references(entries, repository_root)

    lifecycle_counts = {
        lifecycle: sum(entry["lifecycle"] == lifecycle for entry in entries)
        for lifecycle in ALLOWED_LIFECYCLES
    }
    return ParameterAssetSummary(
        governed_asset_count=len(entries),
        current_release_count=lifecycle_counts["current_release"],
        external_confirmation_count=lifecycle_counts["external_confirmation"],
        historical_frozen_count=lifecycle_counts["historical_frozen"],
        local_unregistered_count=len(unregistered),
        local_unregistered_bytes=sum(path.stat().st_size for path in unregistered),
    )


def render_parameter_asset_report(
    payload: Mapping[str, Any], summary: ParameterAssetSummary
) -> str:
    lines = [
        "# Parameter asset inventory",
        "",
        "This report is generated from `configs/parameter_assets.yaml` and the current",
        "parameter files. Governed assets are content-addressed; local ignored sweeps are",
        "inventory only and cannot become release inputs without explicit registration.",
        "",
        "## Summary",
        "",
        f"- Governed assets: {summary.governed_asset_count}",
        f"- Current locked-U0 assets: {summary.current_release_count}",
        f"- External-confirmation assets: {summary.external_confirmation_count}",
        f"- Historical frozen assets: {summary.historical_frozen_count}",
        f"- Local unregistered sweep assets: {summary.local_unregistered_count}",
        f"- Local unregistered sweep bytes: {summary.local_unregistered_bytes}",
        "",
        "## Governed assets",
        "",
        "| Path | Family | Lifecycle | Protocol | Dataset | Calibration | SHA-256 |",
        "|---|---|---|---|---|---:|---|",
    ]
    for entry in payload["entries"]:
        lines.append(
            f"| `{entry['path']}` | `{entry['family']}` | `{entry['lifecycle']}` | "
            f"`{entry['protocol_id']}` | `{entry['dataset']}` | "
            f"{entry['calibration_count']} | `{entry['sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## Local sweep boundary",
            "",
            f"The local root is `{payload['local_precomputed_policy']['root']}`. Files not",
            "listed above are permitted only when their names match one declared local-sweep",
            "patterns; they remain ignored, rebuildable research assets and are not protocol",
            "authorities.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "ParameterAssetSummary",
    "read_parameter_assets",
    "render_parameter_asset_report",
    "validate_parameter_assets",
]
