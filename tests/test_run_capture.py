from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
TOOLS_DIR = ROOT / "tools"
for path in (SRC_DIR, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from alpha_stalled.run_capture import (
    captured_provenance,
    execute_captured_run,
    read_run_capture,
    validate_run_capture,
)
from alpha_stalled.run_manifest import SCHEMA_VERSION, build_run_manifest
from verify_run_capture import verify as verify_capture


class RunCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repository = Path(self.temporary.name)
        subprocess.run(["git", "init", "-q"], cwd=self.repository, check=True)
        (self.repository / "input.txt").write_text("input\n", encoding="utf-8")
        subprocess.run(["git", "add", "input.txt"], cwd=self.repository, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Alpha STALLED Tests",
                "-c",
                "user.email=alpha-stalled-tests@example.invalid",
                "commit",
                "-qm",
                "initial",
            ],
            cwd=self.repository,
            check=True,
        )

    def command_writing(self, experiment_id: str) -> list[str]:
        output = f"results/runs/{experiment_id}/result.json"
        code = (
            "from pathlib import Path; "
            f"Path({output!r}).write_text('{{}}\\n', encoding='utf-8')"
        )
        return [sys.executable, "-c", code]

    def test_successful_run_captures_exact_provenance(self) -> None:
        command = self.command_writing("captured_ok")
        summary = execute_captured_run(
            repository_root=self.repository,
            experiment_id="captured_ok",
            protocol_id="capture_test_v1",
            command=command,
            environment="test:unit",
            require_clean=True,
        )
        capture = read_run_capture(summary.capture_path)
        validated = validate_run_capture(
            capture,
            repository_root=self.repository,
            require_completed=True,
        )
        provenance = captured_provenance(capture, repository_root=self.repository)

        self.assertEqual(summary.exit_code, 0)
        self.assertEqual(validated.experiment_id, "captured_ok")
        self.assertEqual(capture["execution"]["command_argv"], command)
        self.assertEqual(capture["execution"]["state"], "completed")
        self.assertEqual(capture["repository"]["git_status_porcelain"], [])
        self.assertEqual(provenance["mode"], "captured")
        self.assertEqual(provenance["commands_kind"], "exact_invocation")
        self.assertEqual(provenance["provenance_gaps"], [])
        self.assertTrue(
            (self.repository / "results/runs/captured_ok/result.json").is_file()
        )
        verified = verify_capture(
            summary.capture_path,
            repository_root=self.repository,
        )
        self.assertTrue(verified["passed"])
        self.assertEqual(verified["exit_code"], 0)

    def test_nonzero_exit_is_persisted(self) -> None:
        summary = execute_captured_run(
            repository_root=self.repository,
            experiment_id="captured_failure",
            protocol_id="capture_test_v1",
            command=[sys.executable, "-c", "raise SystemExit(7)"],
        )
        capture = read_run_capture(summary.capture_path)

        self.assertEqual(summary.exit_code, 7)
        self.assertEqual(capture["execution"]["exit_code"], 7)
        self.assertEqual(capture["execution"]["state"], "completed")

    def test_existing_run_directory_is_never_reused(self) -> None:
        execute_captured_run(
            repository_root=self.repository,
            experiment_id="captured_once",
            protocol_id="capture_test_v1",
            command=[sys.executable, "-c", "pass"],
        )

        with self.assertRaisesRegex(FileExistsError, "refusing to reuse"):
            execute_captured_run(
                repository_root=self.repository,
                experiment_id="captured_once",
                protocol_id="capture_test_v1",
                command=[sys.executable, "-c", "pass"],
            )

    def test_require_clean_rejects_dirty_worktree_before_creating_output(self) -> None:
        (self.repository / "dirty.txt").write_text("dirty\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "worktree is dirty"):
            execute_captured_run(
                repository_root=self.repository,
                experiment_id="captured_dirty",
                protocol_id="capture_test_v1",
                command=[sys.executable, "-c", "pass"],
                require_clean=True,
            )
        self.assertFalse((self.repository / "results/runs/captured_dirty").exists())

    def test_manifest_builder_expands_capture_and_hashes_it(self) -> None:
        summary = execute_captured_run(
            repository_root=self.repository,
            experiment_id="captured_manifest",
            protocol_id="capture_test_v1",
            command=self.command_writing("captured_manifest"),
        )
        spec = {
            "schema_version": SCHEMA_VERSION,
            "identity": {
                "experiment_id": "captured_manifest",
                "protocol_id": "capture_test_v1",
                "parent_experiment_id": "",
                "status": "diagnostic_only",
            },
            "purpose": "Exercise capture-backed manifest construction.",
            "provenance": {
                "capture_path": summary.capture_path.relative_to(
                    self.repository
                ).as_posix()
            },
            "factors": {
                "changed": ["Test output"],
                "frozen": ["Temporary repository"],
            },
            "selection": {
                "uses_generated_for_fit": False,
                "uses_generated_for_selection": False,
                "selection_scope": "No generated data in this unit test.",
            },
            "data": {
                "calibration_real_count": 0,
                "generated_calibration_count": 0,
                "evaluation_video_count": 1,
                "calibration_evaluation_overlap_count": 0,
                "score_direction": "higher_is_real",
                "ap_positive_class": "real",
                "macro_definition": "Single synthetic unit-test result.",
            },
            "metrics": {"macro_auc": 0.5, "macro_ap": 0.5},
            "artifacts": {
                "inputs": [{"role": "test_input", "path": "input.txt"}],
                "intermediates": [],
                "outputs": [
                    {
                        "role": "test_result",
                        "path": "results/runs/captured_manifest/result.json",
                    }
                ],
            },
            "outcome": {"state": "completed", "failures": []},
        }

        manifest = build_run_manifest(spec, self.repository)

        self.assertEqual(manifest["provenance"]["mode"], "captured")
        self.assertEqual(manifest["artifacts"]["inputs"][0]["role"], "run_capture")
        self.assertEqual(manifest["artifacts"]["inputs"][0]["bytes"], summary.capture_path.stat().st_size)
        self.assertEqual(len(manifest["artifacts"]["inputs"][0]["sha256"]), 64)
        json.dumps(manifest)


if __name__ == "__main__":
    unittest.main()
