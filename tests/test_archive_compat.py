from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from alpha_stalled.archive_compat import ARCHIVED_TOOL_ROOT, archived_tool_path


class ArchiveCompatibilityTests(unittest.TestCase):
    ARCHIVED_PATHS = (
        "journal_experiments/analyze_cross_dataset_frozen_hyperparams.py",
        "journal_experiments/analyze_duration_window_feasibility.py",
        "journal_experiments/analyze_journal_experiments.py",
        "journal_experiments/audit_failure_cases.py",
        "journal_experiments/audit_patch_likelihood_assumptions.py",
        "journal_experiments/audit_reference_experiment_alignment.py",
        "journal_experiments/benchmark_csv_stage_runtime.py",
        "journal_experiments/benchmark_video_stage_runtime.py",
        "journal_experiments/bootstrap_macro_average_delta.py",
        "journal_experiments/inspect_dinov3_tokens.py",
        "journal_experiments/summarize_duration_window_representative.py",
        "journal_experiments/summarize_journal_experiments.py",
        "journal_experiments/summarize_metrics_average_rows.py",
        "pre_release_assets/verify_alpha_stalled_release.py",
    )

    def test_archived_implementations_are_resolved_inside_archive_root(self) -> None:
        for relative in self.ARCHIVED_PATHS:
            path = archived_tool_path(relative)
            self.assertTrue(path.is_file())
            self.assertTrue(path.is_relative_to(ARCHIVED_TOOL_ROOT.resolve()))

    def test_archived_tool_path_rejects_escape_and_missing_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be relative"):
            archived_tool_path("../tools/unsafe.py")
        with self.assertRaises(FileNotFoundError):
            archived_tool_path("journal_experiments/missing.py")


if __name__ == "__main__":
    unittest.main()
