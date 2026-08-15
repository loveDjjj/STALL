from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from alpha_stalled.calibration import (
    U0WindowReferences,
    calibrate_u0_window_components,
    cdf_with_positive_infinity,
    fuse_global_components,
    fuse_global_local,
    fuse_local_components,
)
from alpha_stalled.parameters import global_references, load_raw_params
import analyze_u0_locked
import score_u0_locked_windows


class CalibrationParameterTests(unittest.TestCase):
    def test_legacy_entries_reexport_shared_functions(self) -> None:
        self.assertIs(
            analyze_u0_locked.cdf_with_positive_infinity,
            cdf_with_positive_infinity,
        )
        self.assertIs(analyze_u0_locked.global_references, global_references)
        self.assertIs(score_u0_locked_windows.load_raw_params, load_raw_params)

    def test_positive_infinity_policy_preserves_right_inclusive_cdf(self) -> None:
        reference = np.array([-3.0, -1.0, 1.0], dtype=np.float64)
        actual = cdf_with_positive_infinity(
            np.array([-2.0, 0.0, np.inf]), reference
        )
        np.testing.assert_array_equal(actual, [1.0 / 3.0, 2.0 / 3.0, 1.0])
        for invalid in (np.nan, -np.inf):
            with self.assertRaises(ValueError):
                cdf_with_positive_infinity(np.array([invalid]), reference)

    def test_locked_window_calibration_and_fusion_formula(self) -> None:
        reference = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        actual = calibrate_u0_window_components(
            np.array([1.0, 2.0]),
            np.array([0.0, np.inf]),
            np.array([3.0, -1.0]),
            np.array([2.0, 1.0]),
            U0WindowReferences(reference, reference, reference, reference),
        )
        np.testing.assert_array_equal(actual["global_spatial"], [0.5, 0.75])
        np.testing.assert_array_equal(actual["global_t1"], [0.25, 1.0])
        np.testing.assert_array_equal(actual["patch_spatial"], [1.0, 0.0])
        np.testing.assert_array_equal(actual["patch_temporal"], [0.75, 0.5])
        np.testing.assert_array_equal(actual["G_k"], [0.375, 0.875])
        np.testing.assert_array_equal(actual["L_k"], [0.775, 0.45])
        np.testing.assert_allclose(actual["S_k"], [0.535, 0.705], rtol=0.0, atol=1e-15)

    def test_named_fusions_use_locked_defaults_and_reject_shape_drift(self) -> None:
        first = np.array([0.2, 0.8])
        second = np.array([0.6, 0.4])
        np.testing.assert_allclose(fuse_global_components(first, second), [0.4, 0.6])
        np.testing.assert_allclose(fuse_local_components(first, second), [0.56, 0.44])
        np.testing.assert_allclose(fuse_global_local(first, second), [0.36, 0.64])
        with self.assertRaisesRegex(ValueError, "score shapes differ"):
            fuse_global_local(np.array([0.1]), np.array([0.2, 0.3]))

    def test_synthetic_npz_loading_matches_locked_reductions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(
                root / "global.npz",
                mu_spat=np.array([1.0, 2.0], dtype=np.float32),
                W_spat=np.eye(2, dtype=np.float32),
                calib_ll_spat=np.array([[1.0, 3.0], [4.0, 2.0]], dtype=np.float32),
                mu_temp=np.array([5.0, 6.0], dtype=np.float32),
                W_temp=np.eye(2, dtype=np.float32) * 2.0,
                calib_ll_temp=np.array([[-1.0, -3.0], [-4.0, -2.0]], dtype=np.float32),
            )
            np.savez(
                root / "local.npz",
                mu_patch_spat=np.array([7.0, 8.0], dtype=np.float32),
                W_patch_spat=np.eye(2, dtype=np.float32) * 3.0,
                mu_patch_temp=np.array([9.0, 10.0], dtype=np.float32),
                W_patch_temp=np.eye(2, dtype=np.float32) * 4.0,
            )
            config = {
                "global_branch": {"params": "global.npz"},
                "local_branch": {
                    "params_by_dataset": {"demo": {"path": "local.npz"}}
                },
            }

            spatial, temporal = global_references(config, repository_root=root)
            np.testing.assert_array_equal(spatial, [3.0, 4.0])
            np.testing.assert_array_equal(temporal, [-4.0, -3.0])

            params = load_raw_params(config, "demo", repository_root=root)
            self.assertEqual(
                set(params), {"global_spatial", "global_t1", "patch_spatial", "patch_d2"}
            )
            np.testing.assert_array_equal(
                params["global_spatial"].calibration_raw, spatial
            )
            np.testing.assert_array_equal(params["global_t1"].calibration_raw, temporal)
            np.testing.assert_array_equal(params["patch_spatial"].mean, [7.0, 8.0])
            np.testing.assert_array_equal(params["patch_d2"].mean, [9.0, 10.0])
            self.assertEqual(params["patch_spatial"].mean.dtype, np.float64)
            np.testing.assert_array_equal(
                params["patch_spatial"].calibration_raw, [0.0]
            )

    def test_release_io_import_does_not_load_video_or_model_stacks(self) -> None:
        code = (
            "import sys; import alpha_stalled.release_io; "
            "assert 'cv2' not in sys.modules; assert 'torch' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env={"PYTHONPATH": str(ROOT / "src")},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
