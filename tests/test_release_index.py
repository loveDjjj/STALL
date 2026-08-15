from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.release_index import validate_release_indexes


class ReleaseIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.release_root = Path(self.temporary.name) / "release"
        protocol = self.release_root / "u0" / "params"
        protocol.mkdir(parents=True)
        (self.release_root / "README.md").write_text("`u0/`\n", encoding="utf-8")
        (self.release_root / "u0/README.md").write_text(
            "`manifest.json`\n`params/local.npz`\n", encoding="utf-8"
        )
        (self.release_root / "u0/manifest.json").write_text("{}\n", encoding="utf-8")
        (protocol / "local.npz").write_bytes(b"params")

    def test_complete_release_indexes_are_valid(self) -> None:
        summary = validate_release_indexes(self.release_root)
        self.assertEqual(summary.directory_count, 1)
        self.assertEqual(summary.asset_count, 2)

    def test_unindexed_release_directory_is_rejected(self) -> None:
        (self.release_root / "external").mkdir()
        with self.assertRaisesRegex(ValueError, "omits directories"):
            validate_release_indexes(self.release_root)

    def test_root_asset_is_rejected(self) -> None:
        (self.release_root / "scores.csv").write_text("score\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "protocol directories"):
            validate_release_indexes(self.release_root)

    def test_unindexed_nested_asset_is_rejected(self) -> None:
        (self.release_root / "u0/params/extra.npz").write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "omits assets"):
            validate_release_indexes(self.release_root)

    def test_repository_release_indexes_are_valid(self) -> None:
        summary = validate_release_indexes(ROOT / "release")
        self.assertEqual(summary.directory_count, 2)
        self.assertEqual(summary.asset_count, 25)


if __name__ == "__main__":
    unittest.main()
