from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from alpha_stalled.environment_lock import (
    IMPORT_DISTRIBUTIONS,
    discover_external_imports,
    read_environment_spec,
    validate_environment_files,
    validate_installed_environment,
)


class EnvironmentLockTests(unittest.TestCase):
    def test_repository_environment_and_lock_are_exact(self) -> None:
        summary = validate_environment_files(ROOT)
        self.assertEqual(summary.direct_conda_count, 3)
        self.assertEqual(summary.direct_pip_count, 16)
        self.assertEqual(summary.locked_conda_count, 148)
        self.assertGreater(summary.locked_pip_count, 80)
        self.assertEqual(summary.external_import_count, 15)

    def test_every_external_import_has_a_declared_distribution(self) -> None:
        imports = discover_external_imports(ROOT)
        self.assertEqual(set(imports), set(IMPORT_DISTRIBUTIONS))

    def test_unpinned_pip_dependency_is_rejected(self) -> None:
        payload = yaml.safe_load((ROOT / "environment.yml").read_text())
        payload["dependencies"][-1]["pip"].append("new-package")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "environment.yml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact name==version pin"):
                read_environment_spec(path)

    def test_direct_and_lock_version_drift_is_rejected(self) -> None:
        payload = yaml.safe_load((ROOT / "environment.yml").read_text())
        payload["dependencies"][-1]["pip"] = [
            "numpy==0" if value == "numpy==2.2.6" else value
            for value in payload["dependencies"][-1]["pip"]
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "environment.yml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pip version mismatch for numpy"):
                validate_environment_files(ROOT, direct_path=path)

    def test_installed_environment_rejects_build_drift_and_allows_pip_extras(self) -> None:
        lock = read_environment_spec(ROOT / "environment.lock.yml")
        records = [
            {
                "name": requirement.name,
                "version": requirement.version,
                "build_string": requirement.build,
                "channel": "conda-forge",
            }
            for requirement in lock.conda.values()
        ]
        records.extend(
            {
                "name": name,
                "version": version,
                "build_string": "pypi_0",
                "channel": "pypi",
            }
            for name, version in lock.pip.items()
        )
        records.append(
            {
                "name": "local-download-helper",
                "version": "1",
                "build_string": "pypi_0",
                "channel": "pypi",
            }
        )
        summary = validate_installed_environment(lock, records)
        self.assertEqual(summary.overlaid_conda_count, 0)
        self.assertEqual(summary.extra_pip_count, 1)

        overlaid = [
            row
            for row in records
            if not (row["name"] == "setuptools" and row["channel"] != "pypi")
        ]
        summary = validate_installed_environment(lock, overlaid)
        self.assertEqual(summary.matched_conda_count, len(lock.conda) - 1)
        self.assertEqual(summary.overlaid_conda_count, 1)

        missing_python = [row for row in records if row["name"] != "python"]
        with self.assertRaisesRegex(ValueError, "missing conda package: python"):
            validate_installed_environment(lock, missing_python)

        drifted = copy.deepcopy(records)
        python_record = next(row for row in drifted if row["name"] == "python")
        python_record["build_string"] = "wrong_build"
        with self.assertRaisesRegex(ValueError, "installed conda build mismatch for python"):
            validate_installed_environment(lock, drifted)


if __name__ == "__main__":
    unittest.main()
