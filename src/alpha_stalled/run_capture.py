"""Capture exact experiment execution provenance before a process starts."""

from __future__ import annotations

import json
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .experiment_registry import IDENTIFIER
from .release_io import write_json


SCHEMA_VERSION = "alpha_stalled_run_capture_v1"
CAPTURE_FILENAME = "run_capture.json"
EXECUTION_STATES = frozenset({"running", "completed"})


@dataclass(frozen=True)
class RunCaptureSummary:
    experiment_id: str
    protocol_id: str
    state: str
    exit_code: int | None
    capture_path: Path


def _required(mapping: Mapping[str, Any], fields: Sequence[str], context: str) -> None:
    missing = [field for field in fields if field not in mapping]
    if missing:
        raise ValueError(f"{context} missing fields: {missing}")


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a nonempty string")
    return value.strip()


def _timestamp(value: Any, context: str, *, allow_none: bool = False) -> datetime | None:
    if value is None and allow_none:
        return None
    raw = _string(value, context)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{context} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{context} must include a timezone")
    return parsed


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(repository_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.rstrip("\n")


def _git_state(repository_root: Path) -> tuple[str, list[str]]:
    commit = _git(repository_root, "rev-parse", "HEAD")
    status = _git(repository_root, "status", "--porcelain=v1", "--untracked-files=all")
    return commit, status.splitlines() if status else []


def _expected_output_directory(repository_root: Path, experiment_id: str) -> Path:
    return repository_root / "results" / "runs" / experiment_id


def _validate_identity(experiment_id: str, protocol_id: str) -> None:
    if not IDENTIFIER.fullmatch(experiment_id):
        raise ValueError("experiment_id must be a stable lowercase identifier")
    if not IDENTIFIER.fullmatch(protocol_id):
        raise ValueError("protocol_id must be a stable lowercase identifier")


def validate_run_capture(
    capture: Mapping[str, Any],
    *,
    repository_root: Path,
    require_completed: bool = False,
) -> RunCaptureSummary:
    """Validate a capture record without changing it or executing a command."""

    _required(
        capture,
        ("schema_version", "identity", "repository", "runtime", "execution"),
        "capture",
    )
    if capture["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported run capture schema: {capture['schema_version']!r}")

    identity = capture["identity"]
    if not isinstance(identity, Mapping):
        raise ValueError("identity must be an object")
    _required(identity, ("experiment_id", "protocol_id"), "identity")
    experiment_id = _string(identity["experiment_id"], "identity.experiment_id")
    protocol_id = _string(identity["protocol_id"], "identity.protocol_id")
    _validate_identity(experiment_id, protocol_id)

    repository = capture["repository"]
    if not isinstance(repository, Mapping):
        raise ValueError("repository must be an object")
    _required(
        repository,
        (
            "working_directory",
            "output_directory",
            "git_commit",
            "worktree_dirty",
            "git_status_porcelain",
        ),
        "repository",
    )
    if repository["working_directory"] != ".":
        raise ValueError("repository.working_directory must be '.'")
    expected_relative = Path("results") / "runs" / experiment_id
    if Path(_string(repository["output_directory"], "repository.output_directory")) != expected_relative:
        raise ValueError("repository.output_directory does not match experiment_id")
    commit = _string(repository["git_commit"], "repository.git_commit")
    if len(commit) not in {40, 64} or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("repository.git_commit is not a Git object ID")
    if not isinstance(repository["worktree_dirty"], bool):
        raise ValueError("repository.worktree_dirty must be boolean")
    status = repository["git_status_porcelain"]
    if not isinstance(status, list) or any(not isinstance(item, str) for item in status):
        raise ValueError("repository.git_status_porcelain must be a string list")
    if bool(status) != repository["worktree_dirty"]:
        raise ValueError("repository.worktree_dirty disagrees with captured status")

    runtime = capture["runtime"]
    if not isinstance(runtime, Mapping):
        raise ValueError("runtime must be an object")
    _required(
        runtime,
        (
            "environment",
            "python_executable",
            "python_version",
            "platform",
            "conda_default_env",
            "cuda_visible_devices",
        ),
        "runtime",
    )
    for field in ("environment", "python_executable", "python_version", "platform"):
        _string(runtime[field], f"runtime.{field}")
    for field in ("conda_default_env", "cuda_visible_devices"):
        if runtime[field] is not None and not isinstance(runtime[field], str):
            raise ValueError(f"runtime.{field} must be string or null")

    execution = capture["execution"]
    if not isinstance(execution, Mapping):
        raise ValueError("execution must be an object")
    _required(
        execution,
        (
            "command_argv",
            "exact_invocation",
            "started_at_utc",
            "completed_at_utc",
            "state",
            "exit_code",
            "error",
        ),
        "execution",
    )
    argv = execution["command_argv"]
    if not isinstance(argv, list) or not argv or any(
        not isinstance(item, str) or not item for item in argv
    ):
        raise ValueError("execution.command_argv must be a nonempty string list")
    exact_invocation = _string(execution["exact_invocation"], "execution.exact_invocation")
    if exact_invocation != shlex.join(argv):
        raise ValueError("execution.exact_invocation disagrees with command_argv")
    started = _timestamp(execution["started_at_utc"], "execution.started_at_utc")
    completed = _timestamp(
        execution["completed_at_utc"],
        "execution.completed_at_utc",
        allow_none=True,
    )
    state = execution["state"]
    if state not in EXECUTION_STATES:
        raise ValueError(f"unsupported execution.state: {state!r}")
    exit_code = execution["exit_code"]
    error = execution["error"]
    if error is not None and not isinstance(error, str):
        raise ValueError("execution.error must be string or null")
    if state == "running":
        if completed is not None or exit_code is not None or error is not None:
            raise ValueError("running capture cannot have completion fields")
    else:
        if completed is None or isinstance(exit_code, bool) or not isinstance(exit_code, int):
            raise ValueError("completed capture requires timestamp and integer exit_code")
        if completed < started:
            raise ValueError("execution.completed_at_utc precedes started_at_utc")
    if require_completed and state != "completed":
        raise ValueError("run capture is still running")

    capture_path = _expected_output_directory(repository_root, experiment_id) / CAPTURE_FILENAME
    return RunCaptureSummary(
        experiment_id=experiment_id,
        protocol_id=protocol_id,
        state=state,
        exit_code=exit_code,
        capture_path=capture_path,
    )


def captured_provenance(capture: Mapping[str, Any], *, repository_root: Path) -> dict[str, Any]:
    """Translate a completed capture record into run-manifest provenance fields."""

    validate_run_capture(capture, repository_root=repository_root, require_completed=True)
    repository = capture["repository"]
    runtime = capture["runtime"]
    execution = capture["execution"]
    return {
        "mode": "captured",
        "code_git_commit": repository["git_commit"],
        "worktree_dirty": repository["worktree_dirty"],
        "environment": runtime["environment"],
        "commands_kind": "exact_invocation",
        "commands": [execution["exact_invocation"]],
        "started_at_utc": execution["started_at_utc"],
        "completed_at_utc": execution["completed_at_utc"],
        "provenance_gaps": [],
    }


def read_run_capture(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("run capture must be a JSON object")
    return payload


def execute_captured_run(
    *,
    repository_root: Path,
    experiment_id: str,
    protocol_id: str,
    command: Sequence[str],
    environment: str | None = None,
    require_clean: bool = False,
) -> RunCaptureSummary:
    """Execute one exact argv in a fresh canonical run directory and record it."""

    repository_root = repository_root.resolve()
    if not (repository_root / ".git").exists():
        raise ValueError(f"repository root is not a Git worktree: {repository_root}")
    _validate_identity(experiment_id, protocol_id)
    argv = list(command)
    if not argv or any(not isinstance(item, str) or not item for item in argv):
        raise ValueError("command must be a nonempty sequence of nonempty strings")

    commit, status = _git_state(repository_root)
    if require_clean and status:
        raise ValueError("--require-clean was set but the Git worktree is dirty")

    output_directory = _expected_output_directory(repository_root, experiment_id)
    if output_directory.exists():
        raise FileExistsError(f"refusing to reuse experiment directory: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=False)
    capture_path = output_directory / CAPTURE_FILENAME
    runtime_environment = environment or (
        f"conda:{os.environ['CONDA_DEFAULT_ENV']}"
        if os.environ.get("CONDA_DEFAULT_ENV")
        else f"python:{sys.executable}"
    )
    started_at = _utc_now()
    capture: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "experiment_id": experiment_id,
            "protocol_id": protocol_id,
        },
        "repository": {
            "working_directory": ".",
            "output_directory": output_directory.relative_to(repository_root).as_posix(),
            "git_commit": commit,
            "worktree_dirty": bool(status),
            "git_status_porcelain": status,
        },
        "runtime": {
            "environment": runtime_environment,
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "execution": {
            "command_argv": argv,
            "exact_invocation": shlex.join(argv),
            "started_at_utc": started_at,
            "completed_at_utc": None,
            "state": "running",
            "exit_code": None,
            "error": None,
        },
    }
    validate_run_capture(capture, repository_root=repository_root)
    write_json(capture_path, capture)

    exit_code: int
    error: str | None = None
    try:
        completed = subprocess.run(argv, cwd=repository_root, check=False)
        exit_code = int(completed.returncode)
    except KeyboardInterrupt:
        exit_code = 130
        error = "KeyboardInterrupt"
    except OSError as exc:
        exit_code = 127
        error = f"{type(exc).__name__}: {exc}"
    except BaseException as exc:
        exit_code = 1
        error = f"{type(exc).__name__}: {exc}"
        capture["execution"].update(
            {
                "completed_at_utc": _utc_now(),
                "state": "completed",
                "exit_code": exit_code,
                "error": error,
            }
        )
        write_json(capture_path, capture)
        raise

    capture["execution"].update(
        {
            "completed_at_utc": _utc_now(),
            "state": "completed",
            "exit_code": exit_code,
            "error": error,
        }
    )
    validate_run_capture(capture, repository_root=repository_root, require_completed=True)
    write_json(capture_path, capture)
    return RunCaptureSummary(
        experiment_id=experiment_id,
        protocol_id=protocol_id,
        state="completed",
        exit_code=exit_code,
        capture_path=capture_path,
    )


__all__ = [
    "CAPTURE_FILENAME",
    "EXECUTION_STATES",
    "SCHEMA_VERSION",
    "RunCaptureSummary",
    "captured_provenance",
    "execute_captured_run",
    "read_run_capture",
    "validate_run_capture",
]
