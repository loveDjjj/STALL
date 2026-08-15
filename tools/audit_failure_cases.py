#!/usr/bin/env python3
"""Compatibility entrypoint for the archived failure-case audit."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from alpha_stalled.archive_compat import run_archived_tool


if __name__ == "__main__":
    run_archived_tool("journal_experiments/audit_failure_cases.py")
