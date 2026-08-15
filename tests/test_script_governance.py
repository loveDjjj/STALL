from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_TARGET = re.compile(
    r"(?P<path>(?:tools|src|scripts)/[A-Za-z0-9_./-]+\.(?:py|sh))"
)
RESULT_LOG_TARGET = re.compile(
    r"results/[^\s\"']*(?:\.log|/logs(?:/|(?=[\"'])))"
)


class ScriptGovernanceTests(unittest.TestCase):
    def test_all_shell_entrypoints_parse(self) -> None:
        scripts = sorted((ROOT / "scripts").rglob("*.sh"))
        self.assertGreater(len(scripts), 0)
        for script in scripts:
            with self.subTest(script=script.relative_to(ROOT)):
                subprocess.run(
                    ["bash", "-n", str(script)],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )

    def test_current_launchers_keep_logs_out_of_results(self) -> None:
        offenders = []
        for script in sorted((ROOT / "scripts").rglob("*.sh")):
            text = script.read_text(encoding="utf-8")
            for match in RESULT_LOG_TARGET.finditer(text):
                offenders.append(
                    (str(script.relative_to(ROOT)), match.group(0))
                )
        self.assertEqual(offenders, [])

    def test_literal_repository_targets_exist(self) -> None:
        missing = []
        checked = set()
        for script in sorted((ROOT / "scripts").rglob("*.sh")):
            text = script.read_text(encoding="utf-8")
            for match in REPO_TARGET.finditer(text):
                target = match.group("path")
                checked.add(target)
                if not (ROOT / target).is_file():
                    missing.append((str(script.relative_to(ROOT)), target))
        self.assertGreater(len(checked), 0)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
