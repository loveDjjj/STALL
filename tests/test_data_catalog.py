from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.data_catalog import (
    SPEC_SCHEMA_VERSION,
    build_data_catalog,
    render_data_catalog,
    validate_data_catalog,
)
from alpha_stalled.release_io import video_id


CURRENT_CATALOG = ROOT / "results/research_summary/data_catalog.json"


def _manifest(path: Path, split: str, videos: list[dict[str, object]]) -> None:
    payload = {
        "schema_version": "u0_manifest_v1",
        "protocol_split": split,
        "video_count": len(videos),
        "dataset_counts": {"dataset_a": len(videos)},
        "videos": videos,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class DataCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "cache/indexes").mkdir(parents=True)
        (self.root / "datasets/dataset_a/real/RealSource").mkdir(parents=True)
        (self.root / "datasets/dataset_a/fake/FakeSource").mkdir(parents=True)
        real_path = self.root / "datasets/dataset_a/real/RealSource/real.mp4"
        fake_path = self.root / "datasets/dataset_a/fake/FakeSource/fake.mp4"
        outside_path = self.root / "datasets/dataset_a/fake/FakeSource/outside.mp4"
        for path in (real_path, fake_path, outside_path):
            path.write_bytes(b"video")
        self.rows = [
            {
                "video_path": "datasets/dataset_a/real/RealSource/real.mp4",
                "subset": "real",
                "source_model": "RealSource",
                "fps": 8.0,
                "duration_seconds": 3.0,
                "num_frames": 24,
                "downsample_idxs": "[0,1]",
                "1_sec_idxs": "[0]",
                "2_sec_idxs": "[0,1]",
            },
            {
                "video_path": "datasets/dataset_a/fake/FakeSource/fake.mp4",
                "subset": "annotated",
                "source_model": "FakeSource",
                "fps": 8.0,
                "duration_seconds": 2.0,
                "num_frames": 16,
                "downsample_idxs": "[0,1]",
                "1_sec_idxs": "[0]",
                "2_sec_idxs": "[0,1]",
            },
            {
                "video_path": "datasets/dataset_a/fake/FakeSource/outside.mp4",
                "subset": "annotated",
                "source_model": "FakeSource",
                "fps": 8.0,
                "duration_seconds": 1.0,
                "num_frames": 8,
                "downsample_idxs": "[0]",
                "1_sec_idxs": "[0]",
                "2_sec_idxs": None,
            },
        ]
        pd.DataFrame(self.rows).to_csv(self.root / "cache/indexes/dataset_a.csv", index=False)

        videos = []
        for row, split in zip(self.rows[:2], ("calibration", "evaluation")):
            identity = {
                "dataset": "dataset_a",
                "subset": row["subset"],
                "source_model": row["source_model"],
                "filename": Path(str(row["video_path"])).name,
            }
            videos.append(
                {
                    **identity,
                    "video_id": video_id(identity),
                    "protocol_split": split,
                }
            )
        _manifest(self.root / "calibration.json", "calibration", [videos[0]])
        _manifest(self.root / "evaluation.json", "evaluation", [videos[1]])
        self.spec = {
            "schema_version": SPEC_SCHEMA_VERSION,
            "snapshot_id": "test_data_catalog",
            "release": {
                "protocol_id": "test_protocol",
                "calibration_manifest": "calibration.json",
                "evaluation_manifest": "evaluation.json",
            },
            "datasets": [
                {
                    "dataset_id": "dataset_a",
                    "canonical_index": "cache/indexes/dataset_a.csv",
                    "dataset_root": "datasets/dataset_a",
                    "role": "Test fixture.",
                    "notes": "Test fixture notes.",
                }
            ],
        }

    def test_build_validate_and_render(self) -> None:
        catalog = build_data_catalog(self.spec, self.root)
        summary = validate_data_catalog(catalog, repository_root=self.root)
        stats = catalog["datasets"][0]["stats"]

        self.assertEqual(summary.canonical_video_count, 3)
        self.assertEqual(summary.release_video_count, 2)
        self.assertEqual(summary.missing_file_count, 0)
        self.assertEqual(stats["real_video_count"], 1)
        self.assertEqual(stats["generated_video_count"], 2)
        self.assertEqual(stats["duration_ge_2s_count"], 2)
        self.assertEqual(stats["not_in_locked_release_count"], 1)
        report = render_data_catalog(catalog)
        self.assertIn("Canonical video identities: 3", report)
        self.assertIn("not_in_locked_release", report)

    def test_stale_index_is_detected(self) -> None:
        catalog = build_data_catalog(self.spec, self.root)
        frame = pd.read_csv(self.root / "cache/indexes/dataset_a.csv")
        frame.loc[2, "duration_seconds"] = 2.0
        frame.to_csv(self.root / "cache/indexes/dataset_a.csv", index=False)
        with self.assertRaisesRegex(ValueError, "stale"):
            validate_data_catalog(catalog, repository_root=self.root)

    def test_release_identity_absent_from_index_is_rejected(self) -> None:
        payload = json.loads((self.root / "evaluation.json").read_text())
        payload["videos"][0]["video_id"] = "f" * 64
        (self.root / "evaluation.json").write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "video_id does not match identity fields"):
            build_data_catalog(self.spec, self.root)

    def test_release_manifest_identity_fields_are_bound_to_video_id(self) -> None:
        payload = json.loads((self.root / "evaluation.json").read_text())
        payload["videos"][0]["source_model"] = "WrongSource"
        (self.root / "evaluation.json").write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "video_id does not match identity fields"):
            build_data_catalog(self.spec, self.root)

    def test_duplicate_index_identity_is_rejected(self) -> None:
        frame = pd.DataFrame(self.rows + [self.rows[0]])
        frame.to_csv(self.root / "cache/indexes/dataset_a.csv", index=False)
        with self.assertRaisesRegex(ValueError, "duplicate canonical identities"):
            build_data_catalog(self.spec, self.root)

    @unittest.skipUnless(CURRENT_CATALOG.exists(), "data catalog not generated")
    def test_current_catalog_declarations_are_valid(self) -> None:
        catalog = json.loads(CURRENT_CATALOG.read_text(encoding="utf-8"))
        summary = validate_data_catalog(
            catalog,
            repository_root=ROOT,
            verify_inputs=False,
        )
        spec = yaml.safe_load((ROOT / "configs/data_catalog.yaml").read_text())
        self.assertEqual(summary.snapshot_id, spec["snapshot_id"])
        self.assertEqual(summary.dataset_count, 3)
        self.assertEqual(summary.canonical_video_count, 60949)
        self.assertEqual(summary.release_video_count, 22021)


if __name__ == "__main__":
    unittest.main()
