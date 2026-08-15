from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.cache_inventory import (
    SCHEMA_VERSION,
    build_cache_inventory,
    render_cache_inventory,
    validate_cache_inventory,
)


CURRENT_INVENTORY = ROOT / "results/research_summary/cache_inventory.json"


def _group(cache_id: str, path: str, *, hash_contents: bool = False) -> dict[str, object]:
    return {
        "cache_id": cache_id,
        "path": path,
        "family": "test",
        "dataset_scope": "test",
        "purpose": "Exercise cache inventory tests.",
        "lifecycle": "feature_cache",
        "retention": "keep_active",
        "cleanup_priority": "P1_keep_active",
        "release_dependency": False,
        "rebuildable": True,
        "producer": "test producer",
        "rebuild_command": "pytest tests/test_cache_inventory.py",
        "current_consumers": ["tests/test_cache_inventory.py"],
        "metadata_status": "partial",
        "payload_contract": "Opaque test files.",
        "cache_key": {"present": ["filename"], "missing": ["content identity"]},
        "hash_contents": hash_contents,
        "notes": "Test fixture.",
    }


def _spec(groups: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": "test_snapshot",
        "policy": {"cache_root": "cache", "deletion_mode": "manual_approval_only"},
        "groups": groups,
    }


class CacheInventoryTests(unittest.TestCase):
    def test_build_and_validate_complete_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cache/a").mkdir(parents=True)
            (root / "cache/b").mkdir(parents=True)
            (root / "cache/a/item_2s.pt").write_bytes(b"alpha")
            (root / "cache/b/index.csv").write_text("x\n1\n", encoding="utf-8")
            spec = _spec(
                [
                    _group("group_a", "cache/a"),
                    _group("group_b", "cache/b", hash_contents=True),
                ]
            )

            inventory = build_cache_inventory(spec, root)
            summary = validate_cache_inventory(inventory, repository_root=root)

            self.assertEqual(summary.group_count, 2)
            self.assertEqual(summary.file_count, 2)
            self.assertEqual(summary.logical_bytes, 9)
            self.assertEqual(
                inventory["groups"][0]["stats"]["pt_naming_counts"],
                {"compact_2s": 1},
            )
            self.assertIsNone(inventory["groups"][0]["stats"]["content_sha256"])
            self.assertIsNotNone(inventory["groups"][1]["stats"]["content_sha256"])
            self.assertIn("manual approval", render_cache_inventory(inventory))

    def test_layout_drift_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cache/a").mkdir(parents=True)
            target = root / "cache/a/item.bin"
            target.write_bytes(b"before")
            inventory = build_cache_inventory(_spec([_group("group_a", "cache/a")]), root)
            target.write_bytes(b"after and larger")

            with self.assertRaisesRegex(ValueError, "stale"):
                validate_cache_inventory(inventory, repository_root=root)

    def test_content_drift_same_size_is_detected_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cache/a").mkdir(parents=True)
            target = root / "cache/a/item.bin"
            target.write_bytes(b"before")
            inventory = build_cache_inventory(
                _spec([_group("group_a", "cache/a", hash_contents=True)]), root
            )
            target.write_bytes(b"AFTER!")

            with self.assertRaisesRegex(ValueError, "group stats are stale"):
                validate_cache_inventory(inventory, repository_root=root)

    def test_unregistered_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cache/a").mkdir(parents=True)
            (root / "cache/unregistered").mkdir(parents=True)
            (root / "cache/a/item.bin").write_bytes(b"ok")
            (root / "cache/unregistered/item.bin").write_bytes(b"missed")

            with self.assertRaisesRegex(ValueError, "unregistered cache files"):
                build_cache_inventory(_spec([_group("group_a", "cache/a")]), root)

    def test_overlapping_group_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cache/a/child").mkdir(parents=True)
            spec = _spec(
                [_group("group_a", "cache/a"), _group("group_child", "cache/a/child")]
            )

            with self.assertRaisesRegex(ValueError, "paths overlap"):
                build_cache_inventory(spec, root)

    def test_p3_requires_non_release_safe_delete_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cache/a").mkdir(parents=True)
            group = _group("group_a", "cache/a")
            group["cleanup_priority"] = "P3_safe_delete_candidate"

            with self.assertRaisesRegex(ValueError, "P3 candidate"):
                build_cache_inventory(_spec([group]), root)

    @unittest.skipUnless(CURRENT_INVENTORY.exists(), "inventory not generated")
    def test_current_inventory_declarations_are_valid(self) -> None:
        inventory = json.loads(CURRENT_INVENTORY.read_text(encoding="utf-8"))
        summary = validate_cache_inventory(
            inventory,
            repository_root=ROOT,
            verify_layout=False,
        )
        spec = yaml.safe_load(
            (ROOT / "configs/cache_inventory.yaml").read_text(encoding="utf-8")
        )

        self.assertEqual(summary.snapshot_id, spec["snapshot_id"])
        self.assertEqual(summary.group_count, len(spec["groups"]))
        self.assertEqual(
            inventory["policy"]["deletion_mode"], "manual_approval_only"
        )


if __name__ == "__main__":
    unittest.main()
