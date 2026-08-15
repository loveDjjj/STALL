from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOOLS = ROOT / "tools"
for path in (SRC, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from alpha_stalled.run_manifest import (
    SCHEMA_VERSION,
    build_run_manifest,
    validate_run_manifest,
)
from verify_run_manifest import verify


def manifest_spec() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "experiment_id": "test_run",
            "protocol_id": "test_protocol_v1",
            "parent_experiment_id": "",
            "status": "main_method_locked",
        },
        "purpose": "Exercise the run manifest contract.",
        "provenance": {
            "mode": "reconstructed",
            "code_git_commit": "deadbeef",
            "worktree_dirty": False,
            "environment": "conda:test",
            "commands_kind": "canonical_reproduction",
            "commands": ["python test.py"],
            "started_at_utc": None,
            "completed_at_utc": "2026-08-14T00:01:00+00:00",
            "provenance_gaps": ["Exact original invocation was not captured."],
        },
        "factors": {"changed": ["test factor"], "frozen": ["all other factors"]},
        "selection": {
            "uses_generated_for_fit": False,
            "uses_generated_for_selection": False,
            "selection_scope": "No generated data used.",
        },
        "data": {
            "calibration_real_count": 2,
            "generated_calibration_count": 0,
            "evaluation_video_count": 4,
            "calibration_evaluation_overlap_count": 0,
            "score_direction": "higher_is_real",
            "ap_positive_class": "real",
            "macro_definition": "One test dataset.",
        },
        "metrics": {"macro_auc": 0.8, "macro_ap": 0.81},
        "artifacts": {
            "inputs": [{"role": "input", "path": "input.txt"}],
            "intermediates": [],
            "outputs": [{"role": "output", "path": "output.txt"}],
        },
        "outcome": {"state": "completed", "failures": []},
    }


class RunManifestTests(unittest.TestCase):
    def test_builds_and_validates_hashed_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input\n", encoding="utf-8")
            (root / "output.txt").write_text("output\n", encoding="utf-8")
            manifest = build_run_manifest(manifest_spec(), root)
            summary = validate_run_manifest(manifest, repository_root=root)
            self.assertEqual(summary.artifact_count, 2)
            self.assertEqual(summary.verified_hash_count, 2)
            self.assertEqual(manifest["artifacts"]["inputs"][0]["bytes"], 6)

    def test_detects_artifact_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input\n", encoding="utf-8")
            (root / "output.txt").write_text("output\n", encoding="utf-8")
            manifest = build_run_manifest(manifest_spec(), root)
            (root / "output.txt").write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "byte size mismatch|SHA-256 mismatch"):
                validate_run_manifest(manifest, repository_root=root)

    def test_reconstructed_provenance_requires_explicit_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input\n", encoding="utf-8")
            (root / "output.txt").write_text("output\n", encoding="utf-8")
            spec = manifest_spec()
            provenance = spec["provenance"]
            provenance["provenance_gaps"] = []
            with self.assertRaisesRegex(ValueError, "explicit gaps"):
                build_run_manifest(spec, root)

    def test_captured_provenance_requires_capture_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input\n", encoding="utf-8")
            (root / "output.txt").write_text("output\n", encoding="utf-8")
            spec = manifest_spec()
            spec["provenance"] = {
                "mode": "captured",
                "code_git_commit": "deadbeef",
                "worktree_dirty": False,
                "environment": "conda:test",
                "commands_kind": "exact_invocation",
                "commands": ["python test.py"],
                "started_at_utc": "2026-08-14T00:00:00+00:00",
                "completed_at_utc": "2026-08-14T00:01:00+00:00",
                "provenance_gaps": [],
            }
            with self.assertRaisesRegex(ValueError, "run_capture input artifact"):
                build_run_manifest(spec, root)

    def test_rejects_duplicate_artifact_roles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input\n", encoding="utf-8")
            (root / "output.txt").write_text("output\n", encoding="utf-8")
            manifest = build_run_manifest(manifest_spec(), root)
            duplicate = copy.deepcopy(manifest["artifacts"]["inputs"][0])
            duplicate["path"] = "output.txt"
            manifest["artifacts"]["outputs"] = [duplicate]
            with self.assertRaisesRegex(ValueError, "duplicate artifact role"):
                validate_run_manifest(manifest, repository_root=root, verify_hashes=False)

    def test_spec_cannot_predeclare_integrity_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input\n", encoding="utf-8")
            (root / "output.txt").write_text("output\n", encoding="utf-8")
            spec = manifest_spec()
            spec["artifacts"]["inputs"][0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "must not predeclare"):
                build_run_manifest(spec, root)

    def test_locked_run_manifest_matches_registry_and_config(self) -> None:
        payload = verify(verify_hashes=False)
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["experiment_id"], "alpha_stalled_u0_locked")
        self.assertEqual(payload["artifact_count"], 28)
        self.assertEqual(payload["verified_hash_count"], 0)


if __name__ == "__main__":
    unittest.main()
