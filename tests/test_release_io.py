from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from alpha_stalled.release_io import (
    repository_relative,
    resolve_required_video,
    resolve_video,
    sha256_file,
    video_id,
    video_id_shard,
    write_json,
)
from build_u0_release_manifests import (
    repository_relative as legacy_repository_relative,
    resolve_video as legacy_resolve_video,
    sha256_file as legacy_sha256_file,
    video_id as legacy_video_id,
    write_json as legacy_write_json,
)
from score_u0_locked_windows import (
    resolve_video as legacy_required_video,
    stable_shard as legacy_u0_stable_shard,
)


class ReleaseIoTests(unittest.TestCase):
    def test_legacy_manifest_entry_reexports_canonical_helpers(self) -> None:
        self.assertIs(legacy_sha256_file, sha256_file)
        self.assertIs(legacy_video_id, video_id)
        self.assertIs(legacy_repository_relative, repository_relative)
        self.assertIs(legacy_resolve_video, resolve_video)
        self.assertIs(legacy_write_json, write_json)

    def test_video_identity_matches_locked_formula(self) -> None:
        row = {
            "dataset": "d",
            "subset": "annotated",
            "source_model": "g",
            "filename": "v.mp4",
        }
        expected = hashlib.sha256(b"d|annotated|g|v.mp4").hexdigest()
        self.assertEqual(video_id(row), expected)

    def test_video_id_sharding_matches_locked_formula(self) -> None:
        identifier = hashlib.sha256(b"locked-video").hexdigest()
        for num_shards in (1, 2, 3, 8, 17):
            expected = int(identifier[:16], 16) % num_shards
            self.assertEqual(video_id_shard(identifier, num_shards), expected)
            self.assertEqual(legacy_u0_stable_shard(identifier, num_shards), expected)
        self.assertIs(legacy_u0_stable_shard, video_id_shard)

    def test_repository_relative_preserves_release_path_convention(self) -> None:
        self.assertEqual(
            repository_relative(ROOT / "datasets/example.mp4"),
            "STALL/datasets/example.mp4",
        )
        self.assertEqual(repository_relative("datasets/example.mp4"), "datasets/example.mp4")

    def test_resolver_supports_audit_and_strict_modes(self) -> None:
        missing = "STALL/datasets/does-not-exist.mp4"
        self.assertEqual(resolve_video(missing), (ROOT.parent / missing).resolve())
        with self.assertRaises(FileNotFoundError):
            resolve_video(missing, require_file=True)
        with tempfile.NamedTemporaryFile() as handle:
            self.assertEqual(resolve_video(handle.name), Path(handle.name).resolve())
            self.assertEqual(
                resolve_required_video(handle.name), Path(handle.name).resolve()
            )
        self.assertIs(legacy_required_video, resolve_required_video)

    def test_hash_and_atomic_json_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.bin"
            data.write_bytes(b"alpha-stalled")
            self.assertEqual(
                sha256_file(data), hashlib.sha256(b"alpha-stalled").hexdigest()
            )
            output = root / "nested/result.json"
            write_json(output, {"z": 1, "a": [2]})
            self.assertEqual(json.loads(output.read_text()), {"a": [2], "z": 1})
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                '{\n  "a": [\n    2\n  ],\n  "z": 1\n}\n',
            )
            self.assertFalse(output.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
