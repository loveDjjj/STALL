from __future__ import annotations

import copy
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.parameter_assets import (
    SCHEMA_VERSION,
    read_parameter_assets,
    validate_parameter_assets,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ParameterAssetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "precomputed").mkdir()
        (self.root / "release").mkdir()
        self.asset = self.root / "precomputed/global.npz"
        self._write_global(self.asset, temporal_rank=1023)
        self.catalog = {
            "schema_version": SCHEMA_VERSION,
            "local_precomputed_policy": {
                "root": "precomputed",
                "allowed_unregistered_globs": [
                    "patch_params_*.npz",
                    "debug_patch_params_*.npz",
                ],
            },
            "entries": [self._global_entry(self.asset)],
        }

    @staticmethod
    def _write_global(path: Path, temporal_rank: int) -> None:
        np.savez(
            path,
            mu_spat=np.zeros(1024, dtype=np.float32),
            W_spat=np.zeros((1024, 1024), dtype=np.float32),
            calib_ll_spat=np.zeros((2, 16), dtype=np.float64),
            mu_temp=np.zeros(1024, dtype=np.float32),
            W_temp=np.zeros((1024, temporal_rank), dtype=np.float32),
            calib_ll_temp=np.zeros((2, 15), dtype=np.float64),
        )

    def _global_entry(self, path: Path) -> dict[str, object]:
        return {
            "path": path.relative_to(self.root).as_posix(),
            "family": "global_stall",
            "lifecycle": "historical_frozen",
            "role": "test_global",
            "protocol_id": "test_v1",
            "dataset": "test_real",
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "calibration_count": 2,
            "spatial_rank": 1024,
            "temporal_rank": 1023,
        }

    def validate(self, catalog: dict[str, object]):
        return validate_parameter_assets(
            catalog,
            self.root,
            verify_git_tracking=False,
            verify_locked_u0=False,
        )

    def test_valid_parameter_asset_is_accepted(self) -> None:
        summary = self.validate(self.catalog)
        self.assertEqual(summary.governed_asset_count, 1)
        self.assertEqual(summary.historical_frozen_count, 1)

    def test_sha_mismatch_is_rejected(self) -> None:
        catalog = copy.deepcopy(self.catalog)
        catalog["entries"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.validate(catalog)

    def test_npz_shape_mismatch_is_rejected(self) -> None:
        self._write_global(self.asset, temporal_rank=1022)
        catalog = copy.deepcopy(self.catalog)
        catalog["entries"][0].update(
            {"bytes": self.asset.stat().st_size, "sha256": sha256(self.asset)}
        )
        with self.assertRaisesRegex(ValueError, "array contract mismatch"):
            self.validate(catalog)

    def test_unknown_unregistered_precomputed_asset_is_rejected(self) -> None:
        (self.root / "precomputed/unknown.npz").write_bytes(b"unknown")
        with self.assertRaisesRegex(ValueError, "not allowed"):
            self.validate(self.catalog)

    def test_declared_local_sweep_name_is_inventory_only(self) -> None:
        local = self.root / "precomputed/patch_params_candidate.npz"
        local.write_bytes(b"local")
        summary = self.validate(self.catalog)
        self.assertEqual(summary.local_unregistered_count, 1)
        self.assertEqual(summary.local_unregistered_bytes, len(b"local"))

    def test_repository_parameter_catalog_is_valid(self) -> None:
        catalog = read_parameter_assets(ROOT / "configs/parameter_assets.yaml")
        summary = validate_parameter_assets(catalog, ROOT)
        self.assertEqual(summary.governed_asset_count, 8)
        self.assertEqual(summary.current_release_count, 4)
        self.assertEqual(summary.external_confirmation_count, 1)
        self.assertEqual(summary.historical_frozen_count, 3)


if __name__ == "__main__":
    unittest.main()
