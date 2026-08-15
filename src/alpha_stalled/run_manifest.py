"""Build and validate reproducible experiment run manifests."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .experiment_registry import ALLOWED_STATUSES, IDENTIFIER
from .release_io import sha256_file
from .run_capture import captured_provenance, read_run_capture, validate_run_capture


SCHEMA_VERSION = "alpha_stalled_run_manifest_v1"
ARTIFACT_GROUPS = ("inputs", "intermediates", "outputs")
PROVENANCE_MODES = frozenset({"captured", "reconstructed"})
OUTCOME_STATES = frozenset({"completed", "completed_with_recovered_exit", "failed"})


@dataclass(frozen=True)
class RunManifestSummary:
    experiment_id: str
    protocol_id: str
    provenance_mode: str
    artifact_count: int
    verified_hash_count: int


def _required(mapping: Mapping[str, Any], fields: tuple[str, ...], context: str) -> None:
    missing = [field for field in fields if field not in mapping]
    if missing:
        raise ValueError(f"{context} missing fields: {missing}")


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a nonempty string")
    return value.strip()


def _string_list(value: Any, context: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{context} must be a {'possibly empty ' if allow_empty else ''}list")
    for index, item in enumerate(value):
        _nonempty_string(item, f"{context}[{index}]")
    return value


def _timestamp(value: Any, context: str, *, allow_none: bool = False) -> datetime | None:
    if value is None and allow_none:
        return None
    raw = _nonempty_string(value, context)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{context} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{context} must include a timezone")
    return parsed


def _count(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a nonnegative integer")
    return value


def _metric(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{context} must be in [0, 1]")
    return parsed


def _artifact_path(raw_path: Any, repository_root: Path, context: str) -> Path:
    relative = Path(_nonempty_string(raw_path, f"{context}.path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{context}.path must be repository-relative")
    path = repository_root / relative
    if not path.is_file():
        raise ValueError(f"{context}.path does not exist: {relative}")
    try:
        path.resolve().relative_to(repository_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{context}.path resolves outside the repository") from exc
    return path


def build_run_manifest(spec: Mapping[str, Any], repository_root: Path) -> dict[str, Any]:
    """Expand an unhashed manifest specification into a deterministic manifest."""

    manifest = copy.deepcopy(dict(spec))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported run manifest schema: {manifest.get('schema_version')!r}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("artifacts must be an object")
    provenance = manifest.get("provenance")
    if isinstance(provenance, dict) and "capture_path" in provenance:
        if set(provenance) != {"capture_path"}:
            raise ValueError("capture-backed provenance may only declare capture_path")
        capture_relative = Path(
            _nonempty_string(provenance["capture_path"], "provenance.capture_path")
        )
        if capture_relative.is_absolute() or ".." in capture_relative.parts:
            raise ValueError("provenance.capture_path must be repository-relative")
        capture_path = repository_root / capture_relative
        if not capture_path.is_file():
            raise ValueError(f"provenance.capture_path does not exist: {capture_relative}")
        capture = read_run_capture(capture_path)
        capture_summary = validate_run_capture(
            capture,
            repository_root=repository_root,
            require_completed=True,
        )
        if capture_path.resolve() != capture_summary.capture_path.resolve():
            raise ValueError("provenance.capture_path is not the canonical experiment capture")
        identity = manifest.get("identity")
        if not isinstance(identity, Mapping):
            raise ValueError("identity must be an object")
        if (
            identity.get("experiment_id") != capture_summary.experiment_id
            or identity.get("protocol_id") != capture_summary.protocol_id
        ):
            raise ValueError("run manifest identity disagrees with provenance capture")
        outcome = manifest.get("outcome")
        if not isinstance(outcome, Mapping):
            raise ValueError("outcome must be an object")
        if capture_summary.exit_code == 0 and outcome.get("state") != "completed":
            raise ValueError("zero-exit capture requires completed outcome")
        if capture_summary.exit_code != 0 and outcome.get("state") == "completed":
            raise ValueError("nonzero-exit capture cannot have completed outcome")
        manifest["provenance"] = captured_provenance(
            capture,
            repository_root=repository_root,
        )
        inputs = artifacts.get("inputs")
        if not isinstance(inputs, list):
            raise ValueError("artifacts.inputs must be a list")
        if any(
            entry.get("role") == "run_capture"
            or entry.get("path") == capture_relative.as_posix()
            for entry in inputs
            if isinstance(entry, Mapping)
        ):
            raise ValueError("run capture artifact is added automatically and must not be declared")
        inputs.insert(
            0,
            {"role": "run_capture", "path": capture_relative.as_posix()},
        )
    for group in ARTIFACT_GROUPS:
        entries = artifacts.get(group)
        if not isinstance(entries, list):
            raise ValueError(f"artifacts.{group} must be a list")
        for index, entry in enumerate(entries):
            context = f"artifacts.{group}[{index}]"
            if not isinstance(entry, dict):
                raise ValueError(f"{context} must be an object")
            _required(entry, ("role", "path"), context)
            if "bytes" in entry or "sha256" in entry:
                raise ValueError(f"{context}: spec must not predeclare bytes or sha256")
            path = _artifact_path(entry["path"], repository_root, context)
            entry["bytes"] = path.stat().st_size
            entry["sha256"] = sha256_file(path)
    validate_run_manifest(manifest, repository_root=repository_root, verify_hashes=True)
    return manifest


def validate_run_manifest(
    manifest: Mapping[str, Any],
    *,
    repository_root: Path,
    verify_hashes: bool = True,
) -> RunManifestSummary:
    """Validate structure, semantics, paths, and optionally every artifact hash."""

    _required(
        manifest,
        (
            "schema_version",
            "identity",
            "purpose",
            "provenance",
            "factors",
            "selection",
            "data",
            "metrics",
            "artifacts",
            "outcome",
        ),
        "manifest",
    )
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {manifest['schema_version']!r}")

    identity = manifest["identity"]
    if not isinstance(identity, Mapping):
        raise ValueError("identity must be an object")
    _required(identity, ("experiment_id", "protocol_id", "parent_experiment_id", "status"), "identity")
    experiment_id = _nonempty_string(identity["experiment_id"], "identity.experiment_id")
    protocol_id = _nonempty_string(identity["protocol_id"], "identity.protocol_id")
    if not IDENTIFIER.fullmatch(experiment_id) or not IDENTIFIER.fullmatch(protocol_id):
        raise ValueError("experiment_id and protocol_id must be stable lowercase identifiers")
    parent = identity["parent_experiment_id"]
    if not isinstance(parent, str) or (parent and not IDENTIFIER.fullmatch(parent)):
        raise ValueError("identity.parent_experiment_id is invalid")
    if identity["status"] not in ALLOWED_STATUSES:
        raise ValueError(f"unsupported identity.status: {identity['status']!r}")
    _nonempty_string(manifest["purpose"], "purpose")

    provenance = manifest["provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("provenance must be an object")
    _required(
        provenance,
        (
            "mode",
            "code_git_commit",
            "worktree_dirty",
            "environment",
            "commands_kind",
            "commands",
            "started_at_utc",
            "completed_at_utc",
            "provenance_gaps",
        ),
        "provenance",
    )
    mode = provenance["mode"]
    if mode not in PROVENANCE_MODES:
        raise ValueError(f"unsupported provenance.mode: {mode!r}")
    _nonempty_string(provenance["code_git_commit"], "provenance.code_git_commit")
    if not isinstance(provenance["worktree_dirty"], bool):
        raise ValueError("provenance.worktree_dirty must be boolean")
    _nonempty_string(provenance["environment"], "provenance.environment")
    commands_kind = _nonempty_string(provenance["commands_kind"], "provenance.commands_kind")
    _string_list(provenance["commands"], "provenance.commands")
    gaps = _string_list(
        provenance["provenance_gaps"],
        "provenance.provenance_gaps",
        allow_empty=True,
    )
    started = _timestamp(
        provenance["started_at_utc"],
        "provenance.started_at_utc",
        allow_none=True,
    )
    completed = _timestamp(provenance["completed_at_utc"], "provenance.completed_at_utc")
    if started is not None and completed is not None and completed < started:
        raise ValueError("provenance.completed_at_utc precedes started_at_utc")
    if mode == "captured":
        if started is None or commands_kind != "exact_invocation" or gaps:
            raise ValueError("captured provenance requires start time, exact commands, and no gaps")
    else:
        if commands_kind != "canonical_reproduction" or not gaps:
            raise ValueError("reconstructed provenance requires canonical commands and explicit gaps")

    factors = manifest["factors"]
    if not isinstance(factors, Mapping):
        raise ValueError("factors must be an object")
    _required(factors, ("changed", "frozen"), "factors")
    _string_list(factors["changed"], "factors.changed")
    _string_list(factors["frozen"], "factors.frozen")

    selection = manifest["selection"]
    if not isinstance(selection, Mapping):
        raise ValueError("selection must be an object")
    _required(
        selection,
        ("uses_generated_for_fit", "uses_generated_for_selection", "selection_scope"),
        "selection",
    )
    for field in ("uses_generated_for_fit", "uses_generated_for_selection"):
        if not isinstance(selection[field], bool):
            raise ValueError(f"selection.{field} must be boolean")
    _nonempty_string(selection["selection_scope"], "selection.selection_scope")

    data = manifest["data"]
    if not isinstance(data, Mapping):
        raise ValueError("data must be an object")
    _required(
        data,
        (
            "calibration_real_count",
            "generated_calibration_count",
            "evaluation_video_count",
            "calibration_evaluation_overlap_count",
            "score_direction",
            "ap_positive_class",
            "macro_definition",
        ),
        "data",
    )
    for field in (
        "calibration_real_count",
        "generated_calibration_count",
        "evaluation_video_count",
        "calibration_evaluation_overlap_count",
    ):
        _count(data[field], f"data.{field}")
    if data["score_direction"] != "higher_is_real":
        raise ValueError("data.score_direction must be higher_is_real")
    if data["ap_positive_class"] not in {"real", "fake"}:
        raise ValueError("data.ap_positive_class must be real or fake")
    _nonempty_string(data["macro_definition"], "data.macro_definition")

    metrics = manifest["metrics"]
    if not isinstance(metrics, Mapping):
        raise ValueError("metrics must be an object")
    _required(metrics, ("macro_auc", "macro_ap"), "metrics")
    _metric(metrics["macro_auc"], "metrics.macro_auc")
    _metric(metrics["macro_ap"], "metrics.macro_ap")

    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise ValueError("artifacts must be an object")
    seen_roles: set[str] = set()
    seen_paths: set[str] = set()
    artifacts_by_role: dict[str, tuple[str, Path]] = {}
    artifact_count = 0
    verified_hash_count = 0
    for group in ARTIFACT_GROUPS:
        entries = artifacts.get(group)
        if not isinstance(entries, list):
            raise ValueError(f"artifacts.{group} must be a list")
        if group in {"inputs", "outputs"} and not entries:
            raise ValueError(f"artifacts.{group} cannot be empty")
        for index, entry in enumerate(entries):
            context = f"artifacts.{group}[{index}]"
            if not isinstance(entry, Mapping):
                raise ValueError(f"{context} must be an object")
            _required(entry, ("role", "path", "bytes", "sha256"), context)
            role = _nonempty_string(entry["role"], f"{context}.role")
            if role in seen_roles:
                raise ValueError(f"duplicate artifact role: {role}")
            seen_roles.add(role)
            raw_path = _nonempty_string(entry["path"], f"{context}.path")
            if raw_path in seen_paths:
                raise ValueError(f"duplicate artifact path: {raw_path}")
            seen_paths.add(raw_path)
            path = _artifact_path(raw_path, repository_root, context)
            artifacts_by_role[role] = (group, path)
            expected_bytes = _count(entry["bytes"], f"{context}.bytes")
            if path.stat().st_size != expected_bytes:
                raise ValueError(f"{context}: byte size mismatch")
            digest = _nonempty_string(entry["sha256"], f"{context}.sha256")
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"{context}.sha256 is invalid")
            if verify_hashes:
                if sha256_file(path) != digest:
                    raise ValueError(f"{context}: SHA-256 mismatch")
                verified_hash_count += 1
            artifact_count += 1

    outcome = manifest["outcome"]
    if not isinstance(outcome, Mapping):
        raise ValueError("outcome must be an object")
    _required(outcome, ("state", "failures"), "outcome")
    if outcome["state"] not in OUTCOME_STATES:
        raise ValueError(f"unsupported outcome.state: {outcome['state']!r}")
    failures = _string_list(outcome["failures"], "outcome.failures", allow_empty=True)
    if outcome["state"] == "completed" and failures:
        raise ValueError("completed outcome cannot contain failures")
    if outcome["state"] != "completed" and not failures:
        raise ValueError("non-clean outcome must record at least one failure")

    if mode == "captured":
        if "run_capture" not in artifacts_by_role:
            raise ValueError("captured provenance requires a run_capture input artifact")
        capture_group, capture_path = artifacts_by_role["run_capture"]
        if capture_group != "inputs":
            raise ValueError("run_capture must be an input artifact")
        capture = read_run_capture(capture_path)
        capture_summary = validate_run_capture(
            capture,
            repository_root=repository_root,
            require_completed=True,
        )
        if capture_path.resolve() != capture_summary.capture_path.resolve():
            raise ValueError("run_capture artifact is not at its canonical experiment path")
        if (
            capture_summary.experiment_id != experiment_id
            or capture_summary.protocol_id != protocol_id
        ):
            raise ValueError("run_capture identity disagrees with manifest identity")
        expected_provenance = captured_provenance(
            capture,
            repository_root=repository_root,
        )
        if dict(provenance) != expected_provenance:
            raise ValueError("manifest provenance disagrees with run_capture")
        if capture_summary.exit_code == 0 and outcome["state"] != "completed":
            raise ValueError("zero-exit run_capture requires completed outcome")
        if capture_summary.exit_code != 0 and outcome["state"] == "completed":
            raise ValueError("nonzero-exit run_capture cannot have completed outcome")

    return RunManifestSummary(
        experiment_id=experiment_id,
        protocol_id=protocol_id,
        provenance_mode=mode,
        artifact_count=artifact_count,
        verified_hash_count=verified_hash_count,
    )


__all__ = [
    "ARTIFACT_GROUPS",
    "OUTCOME_STATES",
    "PROVENANCE_MODES",
    "SCHEMA_VERSION",
    "RunManifestSummary",
    "build_run_manifest",
    "validate_run_manifest",
]
