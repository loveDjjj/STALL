from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_stalled.artifacts import (
    checkpoint_completed_ids,
    checkpoint_completed_keys,
    expected_shard_paths,
    read_checkpoint_parts,
    read_csv_files,
    read_expected_shards,
)


class ArtifactIOTests(unittest.TestCase):
    def test_expected_shards_preserve_declared_order_and_roundtrip_floats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = {
                "b_shard00_of_02.csv": ("b0", 0.12345678901234568),
                "b_shard01_of_02.csv": ("b1", 0.23456789012345678),
                "a_shard00_of_02.csv": ("a0", 0.34567890123456789),
                "a_shard01_of_02.csv": ("a1", 0.45678901234567891),
            }
            for name, (identity, value) in values.items():
                pd.DataFrame({"video_id": [identity], "score": [value]}).to_csv(
                    root / name, index=False
                )
            paths = expected_shard_paths(root, ("b", "a"), 2)
            self.assertEqual([path.name for path in paths], list(values))
            frame = read_expected_shards(root, ("b", "a"), 2)
            self.assertEqual(frame["video_id"].tolist(), ["b0", "b1", "a0", "a1"])
            self.assertEqual(frame["score"].tolist(), [item[1] for item in values.values()])

    def test_missing_or_empty_declared_inputs_fail_fast(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError) as caught:
                expected_shard_paths(root, ("d",), 2)
            self.assertEqual(Path(caught.exception.filename or caught.exception.args[0]).name, "d_shard00_of_02.csv")
            with self.assertRaisesRegex(ValueError, "num_shards must be positive"):
                expected_shard_paths(root, ("d",), 0)
            with self.assertRaisesRegex(ValueError, "at least one CSV"):
                read_csv_files([])
            with self.assertRaisesRegex(FileNotFoundError, "no demo parts"):
                read_checkpoint_parts(root, empty_error="no demo parts")

    def test_checkpoint_parts_and_resume_ids_are_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"video_id": ["b", "c"], "value": [2.0, 3.0]}).to_csv(
                root / "part_00001.csv", index=False
            )
            pd.DataFrame({"video_id": ["a"], "value": [1.0]}).to_csv(
                root / "part_00000.csv", index=False
            )
            frame, paths = read_checkpoint_parts(root)
            self.assertEqual([path.name for path in paths], ["part_00000.csv", "part_00001.csv"])
            self.assertEqual(frame["video_id"].tolist(), ["a", "b", "c"])
            completed, resume_paths = checkpoint_completed_ids(root, cast_str=True)
            self.assertEqual(completed, {"a", "b", "c"})
            self.assertEqual(resume_paths, paths)

    def test_checkpoint_multi_column_keys_preserve_declared_identity_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(
                {
                    "dataset": ["d", "d"],
                    "filename": ["a.mp4", "b.mp4"],
                    "score": [0.1, 0.2],
                }
            ).to_csv(root / "part_000001.csv", index=False)
            completed, paths = checkpoint_completed_keys(
                root, ("dataset", "filename")
            )
            self.assertEqual(completed, {("d", "a.mp4"), ("d", "b.mp4")})
            self.assertEqual([path.name for path in paths], ["part_000001.csv"])
            with self.assertRaisesRegex(ValueError, "at least one checkpoint key"):
                checkpoint_completed_keys(root, ())


if __name__ == "__main__":
    unittest.main()
