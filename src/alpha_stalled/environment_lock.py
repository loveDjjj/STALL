"""Validation helpers for the curated and fully locked Conda environments."""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml


CONDA_PACKAGE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]*$")
PIP_REQUIREMENT = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^]]+\])?==(?P<version>[^;\s]+)$"
)
CUDA_WHEEL_INDEX = "https://download.pytorch.org/whl/cu128"
DIRECT_ENVIRONMENT_NAME = "stall"
LOCKED_ENVIRONMENT_NAME = "stall-locked-linux-64-cu128"

# Every non-local import under src/ and tools/ must have an explicit owner here.
IMPORT_DISTRIBUTIONS = {
    "PIL": "pillow",
    "cv2": "opencv-python",
    "datasets": "datasets",
    "huggingface_hub": "huggingface-hub",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "tabulate": "tabulate",
    "torch": "torch",
    "torchvision": "torchvision",
    "tqdm": "tqdm",
    "transformers": "transformers",
    "yaml": "pyyaml",
}
REQUIRED_CONDA_PACKAGES = {"ffmpeg", "pip", "python"}
ALLOWED_CROSS_MANAGER_PACKAGES = {"setuptools", "tzdata"}


def normalize_package_name(name: str) -> str:
    """Return the PEP 503 normalized package name."""

    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass(frozen=True)
class CondaRequirement:
    name: str
    version: str
    build: str | None


@dataclass(frozen=True)
class EnvironmentSpec:
    name: str
    channels: tuple[str, ...]
    conda: Mapping[str, CondaRequirement]
    pip: Mapping[str, str]
    pip_options: tuple[str, ...]


@dataclass(frozen=True)
class EnvironmentLockSummary:
    direct_conda_count: int
    direct_pip_count: int
    locked_conda_count: int
    locked_pip_count: int
    external_import_count: int


@dataclass(frozen=True)
class InstalledEnvironmentSummary:
    matched_conda_count: int
    matched_pip_count: int
    overlaid_conda_count: int
    extra_pip_count: int


def _parse_conda_requirement(value: str, *, path: Path) -> CondaRequirement:
    parts = value.split("=")
    if len(parts) not in (2, 3) or any(not part for part in parts):
        raise ValueError(
            f"{path}: conda dependency must use name=version or "
            f"name=version=build: {value!r}"
        )
    name = normalize_package_name(parts[0])
    if not CONDA_PACKAGE_NAME.fullmatch(parts[0]):
        raise ValueError(f"{path}: invalid conda package name: {parts[0]!r}")
    return CondaRequirement(
        name=name,
        version=parts[1],
        build=parts[2] if len(parts) == 3 else None,
    )


def _parse_pip_requirements(
    values: Sequence[object], *, path: Path
) -> tuple[dict[str, str], tuple[str, ...]]:
    requirements: dict[str, str] = {}
    options = []
    for raw_value in values:
        if not isinstance(raw_value, str):
            raise ValueError(f"{path}: pip dependency is not a string: {raw_value!r}")
        value = raw_value.strip()
        if value.startswith("--"):
            options.append(value)
            continue
        match = PIP_REQUIREMENT.fullmatch(value)
        if match is None:
            raise ValueError(
                f"{path}: pip dependency must use an exact name==version pin: {value!r}"
            )
        name = normalize_package_name(match.group("name"))
        if name in requirements:
            raise ValueError(f"{path}: duplicate pip dependency: {name}")
        requirements[name] = match.group("version")
    return requirements, tuple(options)


def read_environment_spec(path: Path) -> EnvironmentSpec:
    """Read the supported deterministic subset of a Conda environment YAML."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: environment must be a mapping")
    unsupported = set(payload) - {"name", "channels", "dependencies"}
    if unsupported:
        raise ValueError(f"{path}: unsupported top-level keys: {sorted(unsupported)}")
    name = payload.get("name")
    channels = payload.get("channels")
    dependencies = payload.get("dependencies")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{path}: missing environment name")
    if not isinstance(channels, list) or not all(
        isinstance(channel, str) and channel for channel in channels
    ):
        raise ValueError(f"{path}: channels must be a non-empty string list")
    if not isinstance(dependencies, list):
        raise ValueError(f"{path}: dependencies must be a list")

    conda: dict[str, CondaRequirement] = {}
    pip: dict[str, str] = {}
    pip_options: tuple[str, ...] = ()
    saw_pip_section = False
    for dependency in dependencies:
        if isinstance(dependency, str):
            requirement = _parse_conda_requirement(dependency.strip(), path=path)
            if requirement.name in conda:
                raise ValueError(f"{path}: duplicate conda dependency: {requirement.name}")
            conda[requirement.name] = requirement
            continue
        if not isinstance(dependency, dict) or set(dependency) != {"pip"}:
            raise ValueError(f"{path}: unsupported dependency entry: {dependency!r}")
        if saw_pip_section:
            raise ValueError(f"{path}: duplicate pip dependency section")
        values = dependency["pip"]
        if not isinstance(values, list):
            raise ValueError(f"{path}: pip dependencies must be a list")
        pip, pip_options = _parse_pip_requirements(values, path=path)
        saw_pip_section = True

    overlap = set(conda) & set(pip)
    invalid_overlap = overlap - ALLOWED_CROSS_MANAGER_PACKAGES
    if invalid_overlap:
        raise ValueError(
            f"{path}: packages declared by both conda and pip: {sorted(invalid_overlap)}"
        )
    return EnvironmentSpec(
        name=name,
        channels=tuple(channels),
        conda=conda,
        pip=pip,
        pip_options=pip_options,
    )


def discover_external_imports(repository_root: Path) -> tuple[str, ...]:
    """Discover top-level third-party imports in repository runtime code and tools."""

    roots = (repository_root / "src", repository_root / "tools")
    python_files = sorted(path for root in roots for path in root.rglob("*.py"))
    local_modules = {path.stem for path in python_files}
    local_modules.update({"alpha_stalled", "dinov3", "tools"})
    imported = set()
    for path in python_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".", 1)[0])
    return tuple(
        sorted(imported - set(sys.stdlib_module_names) - local_modules)
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_environment_files(
    repository_root: Path,
    direct_path: Path | None = None,
    lock_path: Path | None = None,
) -> EnvironmentLockSummary:
    """Validate exact pins, runtime import coverage, and direct/lock agreement."""

    direct_path = direct_path or repository_root / "environment.yml"
    lock_path = lock_path or repository_root / "environment.lock.yml"
    direct = read_environment_spec(direct_path)
    lock = read_environment_spec(lock_path)

    _require(
        direct.name == DIRECT_ENVIRONMENT_NAME,
        f"unexpected direct environment name: {direct.name}",
    )
    _require(
        lock.name == LOCKED_ENVIRONMENT_NAME,
        f"unexpected locked environment name: {lock.name}",
    )
    _require(
        direct.channels == ("conda-forge", "defaults"),
        f"unexpected direct channel order: {direct.channels}",
    )
    _require(
        lock.channels == direct.channels,
        "direct and locked environment channels differ",
    )
    _require(
        set(direct.conda) >= REQUIRED_CONDA_PACKAGES,
        f"direct environment is missing conda packages: "
        f"{sorted(REQUIRED_CONDA_PACKAGES - set(direct.conda))}",
    )
    _require(
        all(requirement.build is None for requirement in direct.conda.values()),
        "direct environment must pin versions without platform-specific builds",
    )
    _require(
        all(requirement.build is not None for requirement in lock.conda.values()),
        "locked conda dependencies must include exact build strings",
    )
    expected_option = f"--extra-index-url {CUDA_WHEEL_INDEX}"
    _require(
        direct.pip_options == (expected_option,),
        "direct environment must declare the CUDA 12.8 wheel index exactly once",
    )
    _require(
        lock.pip_options == direct.pip_options,
        "direct and locked pip options differ",
    )

    external_imports = discover_external_imports(repository_root)
    unknown_imports = set(external_imports) - set(IMPORT_DISTRIBUTIONS)
    _require(
        not unknown_imports,
        f"external imports lack distribution ownership: {sorted(unknown_imports)}",
    )
    required_pip = {
        normalize_package_name(IMPORT_DISTRIBUTIONS[module])
        for module in external_imports
    }
    _require(
        set(direct.pip) >= required_pip,
        f"direct environment is missing imported distributions: "
        f"{sorted(required_pip - set(direct.pip))}",
    )

    for name, requirement in direct.conda.items():
        _require(name in lock.conda, f"locked environment is missing conda package: {name}")
        _require(
            lock.conda[name].version == requirement.version,
            f"conda version mismatch for {name}: "
            f"{requirement.version} != {lock.conda[name].version}",
        )
    for name, version in direct.pip.items():
        _require(name in lock.pip, f"locked environment is missing pip package: {name}")
        _require(
            lock.pip[name] == version,
            f"pip version mismatch for {name}: {version} != {lock.pip[name]}",
        )

    _require(
        lock.pip.get("torch", "").endswith("+cu128"),
        "locked torch must use the CUDA 12.8 wheel",
    )
    _require(
        lock.pip.get("torchvision", "").endswith("+cu128"),
        "locked torchvision must use the CUDA 12.8 wheel",
    )
    return EnvironmentLockSummary(
        direct_conda_count=len(direct.conda),
        direct_pip_count=len(direct.pip),
        locked_conda_count=len(lock.conda),
        locked_pip_count=len(lock.pip),
        external_import_count=len(external_imports),
    )


def validate_installed_environment(
    lock: EnvironmentSpec,
    records: Iterable[Mapping[str, object]],
) -> InstalledEnvironmentSummary:
    """Require every locked package to match records returned by ``conda list``."""

    conda_records: dict[str, Mapping[str, object]] = {}
    pip_records: dict[str, Mapping[str, object]] = {}
    for record in records:
        raw_name = record.get("name")
        if not isinstance(raw_name, str):
            raise ValueError(f"installed package record has no name: {record!r}")
        name = normalize_package_name(raw_name)
        target = pip_records if record.get("channel") == "pypi" else conda_records
        if name in target:
            raise ValueError(f"duplicate installed package record: {name}")
        target[name] = record

    overlaid_conda = {
        name
        for name in lock.conda
        if name not in conda_records
        and name in lock.pip
        and name in ALLOWED_CROSS_MANAGER_PACKAGES
    }
    for name, requirement in lock.conda.items():
        if name in overlaid_conda:
            continue
        _require(name in conda_records, f"current environment is missing conda package: {name}")
        record = conda_records[name]
        _require(
            record.get("version") == requirement.version,
            f"installed conda version mismatch for {name}: "
            f"{record.get('version')} != {requirement.version}",
        )
        _require(
            record.get("build_string") == requirement.build,
            f"installed conda build mismatch for {name}: "
            f"{record.get('build_string')} != {requirement.build}",
        )
    for name, version in lock.pip.items():
        _require(name in pip_records, f"current environment is missing pip package: {name}")
        _require(
            pip_records[name].get("version") == version,
            f"installed pip version mismatch for {name}: "
            f"{pip_records[name].get('version')} != {version}",
        )

    extra_conda = set(conda_records) - set(lock.conda)
    _require(
        not extra_conda,
        f"current environment has unlocked conda packages: {sorted(extra_conda)}",
    )
    return InstalledEnvironmentSummary(
        matched_conda_count=len(lock.conda) - len(overlaid_conda),
        matched_pip_count=len(lock.pip),
        overlaid_conda_count=len(overlaid_conda),
        extra_pip_count=len(set(pip_records) - set(lock.pip)),
    )


__all__ = [
    "CUDA_WHEEL_INDEX",
    "CondaRequirement",
    "EnvironmentLockSummary",
    "EnvironmentSpec",
    "IMPORT_DISTRIBUTIONS",
    "InstalledEnvironmentSummary",
    "discover_external_imports",
    "normalize_package_name",
    "read_environment_spec",
    "validate_environment_files",
    "validate_installed_environment",
]
