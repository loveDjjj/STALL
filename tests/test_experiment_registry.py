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

from alpha_stalled.experiment_registry import REQUIRED_COLUMNS, validate_registry
from verify_experiment_registry import verify


def make_row(**overrides: str) -> dict[str, str]:
    row = {
        "experiment_id": "alpha_main",
        "protocol_id": "protocol_v1",
        "parent_experiment_id": "baseline",
        "experiment_name": "Alpha main",
        "git_commit": "deadbeef",
        "datasets": "dataset-a",
        "evaluation_videos": "10",
        "calibration_real_per_dataset": "2",
        "evaluation_overlap": "0",
        "uses_fake_for_selection": "false",
        "region": "1",
        "aggregation": "mean",
        "layer": "23",
        "K": "3",
        "alpha": "0.6",
        "beta": "0.1",
        "score_direction": "higher_is_real",
        "macro_auc": "0.8",
        "macro_ap": "0.81",
        "status": "main_method_locked",
        "paper_status": "main_method",
        "result_path": "result.csv",
        "report_path": "report.md",
        "reason": "locked result",
    }
    row.update(overrides)
    return row


class ExperimentRegistryTests(unittest.TestCase):
    def test_valid_registry_checks_lineage_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("result.csv", "report.md", "baseline.csv", "baseline.md"):
                (root / name).write_text("evidence\n", encoding="utf-8")
            baseline = make_row(
                experiment_id="baseline",
                parent_experiment_id="",
                experiment_name="Baseline",
                status="main_baseline",
                paper_status="main_baseline",
                result_path="baseline.csv",
                report_path="baseline.md",
            )
            summary = validate_registry(
                [baseline, make_row()], REQUIRED_COLUMNS, repository_root=root
            )
            self.assertEqual(summary.row_count, 2)
            self.assertEqual(summary.main_experiment_id, "alpha_main")
            self.assertEqual(summary.status_counts["main_baseline"], 1)

    def test_rejects_unknown_status_and_parent_cycle(self) -> None:
        baseline = make_row(
            experiment_id="baseline",
            parent_experiment_id="alpha_main",
            experiment_name="Baseline",
            status="main_baseline",
        )
        with self.assertRaisesRegex(ValueError, "cycle"):
            validate_registry([baseline, make_row()], REQUIRED_COLUMNS)

        invalid = copy.deepcopy(make_row(parent_experiment_id=""))
        invalid["status"] = "best_result"
        with self.assertRaisesRegex(ValueError, "unsupported status"):
            validate_registry([invalid], REQUIRED_COLUMNS)

    def test_rejects_missing_evidence_for_claim_status(self) -> None:
        row = make_row(parent_experiment_id="", result_path="")
        with self.assertRaisesRegex(ValueError, "result_path is required"):
            validate_registry([row], REQUIRED_COLUMNS)

    def test_leakage_status_requires_nonzero_overlap(self) -> None:
        row = make_row(
            parent_experiment_id="",
            status="invalid_leakage",
            evaluation_overlap="0",
        )
        with self.assertRaisesRegex(ValueError, "must declare positive/unknown overlap"):
            validate_registry([row], REQUIRED_COLUMNS)

    def test_current_registry_matches_locked_config(self) -> None:
        payload = verify()
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["main_experiment_id"], "alpha_stalled_u0_locked")
        self.assertEqual(payload["row_count"], 9)
        self.assertEqual(payload["claim_evidence_checks"], 9)


if __name__ == "__main__":
    unittest.main()
