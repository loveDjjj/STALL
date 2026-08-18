from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.config_registry import (
    ALLOWED_LIFECYCLES,
    SCHEMA_VERSION,
    read_config_registry,
    validate_config_registry,
)


class ConfigRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        config_root = self.root / "configs"
        config_root.mkdir()
        (config_root / "current.yaml").write_text(
            yaml.safe_dump({"release": {"protocol_version": "current_v1"}}),
            encoding="utf-8",
        )
        (config_root / "historical.yaml").write_text(
            yaml.safe_dump(
                {
                    "config_identity": {
                        "protocol_id": "historical_v1",
                        "lifecycle": "historical_frozen",
                        "current_authority": False,
                    }
                }
            ),
            encoding="utf-8",
        )
        self.registry = {
            "schema_version": SCHEMA_VERSION,
            "lifecycle_classes": sorted(ALLOWED_LIFECYCLES),
            "entries": [
                {
                    "path": "configs/current.yaml",
                    "kind": "protocol_config",
                    "lifecycle": "current_locked",
                    "authority": "current",
                    "protocol_id": "current_v1",
                    "mutable": False,
                },
                {
                    "path": "configs/historical.yaml",
                    "kind": "protocol_config",
                    "lifecycle": "historical_frozen",
                    "authority": "historical",
                    "protocol_id": "historical_v1",
                    "mutable": False,
                },
                {
                    "path": "configs/config_registry.yaml",
                    "kind": "governance_registry",
                    "lifecycle": "governance",
                    "authority": "supporting",
                    "protocol_id": None,
                    "mutable": True,
                },
            ],
        }
        # The validator intentionally fixes the current authority to the repository path.
        self.registry["entries"][0]["path"] = "configs/alpha_stalled_u0_locked.yaml"
        (config_root / "current.yaml").rename(
            config_root / "alpha_stalled_u0_locked.yaml"
        )
        (config_root / "config_registry.yaml").write_text(
            yaml.safe_dump(self.registry, sort_keys=False), encoding="utf-8"
        )

    def test_valid_registry_covers_every_config_asset(self) -> None:
        summary = validate_config_registry(self.registry, self.root)
        self.assertEqual(summary.asset_count, 3)
        self.assertEqual(summary.protocol_config_count, 2)
        self.assertEqual(summary.historical_config_count, 1)

    def test_unregistered_asset_is_rejected(self) -> None:
        (self.root / "configs/stray.yaml").write_text("value: 1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            validate_config_registry(self.registry, self.root)

    def test_multiple_current_configs_are_rejected(self) -> None:
        registry = copy.deepcopy(self.registry)
        registry["entries"][1].update(
            {"authority": "current", "lifecycle": "current_locked"}
        )
        (self.root / "configs/historical.yaml").write_text(
            yaml.safe_dump({"release": {"protocol_version": "historical_v1"}}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "exactly one current config"):
            validate_config_registry(registry, self.root)

    def test_protocol_identity_mismatch_is_rejected(self) -> None:
        registry = copy.deepcopy(self.registry)
        registry["entries"][1]["protocol_id"] = "wrong_v1"
        with self.assertRaisesRegex(ValueError, "protocol identity mismatch"):
            validate_config_registry(registry, self.root)

    def test_current_repository_registry_is_valid(self) -> None:
        registry = read_config_registry(ROOT / "configs/config_registry.yaml")
        summary = validate_config_registry(registry, ROOT)
        self.assertEqual(summary.current_path, "configs/alpha_stalled_u0_locked.yaml")
        self.assertEqual(summary.asset_count, 9)


if __name__ == "__main__":
    unittest.main()
