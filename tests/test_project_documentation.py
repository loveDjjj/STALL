from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from verify_project_documentation import verify


class ProjectDocumentationTests(unittest.TestCase):
    def test_entry_documents_match_locked_release(self) -> None:
        checks = verify()
        self.assertGreaterEqual(len(checks), 20)


if __name__ == "__main__":
    unittest.main()
