from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import analyze_u0_locked
import score_u0_locked_windows
from alpha_stalled import u0_analysis, u0_scoring
from alpha_stalled import u0_protocol
from alpha_stalled.release_io import video_id


def _identity(dataset: str, subset: str, source: str, filename: str, split: str) -> dict:
    row = {
        "dataset": dataset,
        "subset": subset,
        "source_model": source,
        "filename": filename,
        "protocol_split": split,
        "video_path": f"datasets/{dataset}/{subset}/{source}/{filename}",
        "duration_seconds": 2.0,
        "effective_k": 1,
    }
    row["video_id"] = video_id(row)
    return row


class U0ProtocolTests(unittest.TestCase):
    def test_formal_u0_clis_do_not_import_other_tool_modules(self) -> None:
        tool_modules = {path.stem for path in (ROOT / "tools").glob("*.py")}
        for filename in (
            "score_u0_locked_windows.py",
            "analyze_u0_locked.py",
            "verify_u0_locked_release.py",
        ):
            tree = ast.parse((ROOT / "tools" / filename).read_text(encoding="utf-8"))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".", 1)[0])
                elif isinstance(node, ast.Import):
                    imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            self.assertFalse(
                imported & tool_modules,
                f"{filename} imports tool CLIs: {sorted(imported & tool_modules)}",
            )
        forbidden = {"score_u0_locked_windows", "analyze_u0_locked"}
        offenders = []
        for path in (ROOT / "tools").glob("*.py"):
            if path.name in {
                "score_u0_locked_windows.py",
                "analyze_u0_locked.py",
            }:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported = {
                node.module.split(".", 1)[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module
            }
            if imported & forbidden:
                offenders.append(path.name)
        self.assertEqual(offenders, [])

    def test_cli_compatibility_exports_are_shared_objects(self) -> None:
        self.assertIs(analyze_u0_locked.load_raw_windows, u0_protocol.load_raw_windows)
        self.assertIs(
            analyze_u0_locked.load_calibration_references,
            u0_protocol.load_calibration_references,
        )
        self.assertIs(
            analyze_u0_locked.selected_calibration_means,
            u0_protocol.selected_calibration_means,
        )
        self.assertIs(
            score_u0_locked_windows.load_release_rows,
            u0_protocol.load_release_rows,
        )
        self.assertIs(score_u0_locked_windows.decode_row, u0_scoring.decode_row)
        self.assertIs(score_u0_locked_windows.score_batch, u0_scoring.score_batch)
        self.assertIs(analyze_u0_locked.calibrate_windows, u0_analysis.calibrate_windows)
        self.assertIs(
            analyze_u0_locked.aggregate_and_calibrate_videos,
            u0_analysis.aggregate_and_calibrate_videos,
        )

    def test_release_rows_bind_manifest_identity_and_frame_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = _identity("comgenvid", "real", "Real", "r.mp4", "calibration")
            evaluation = _identity("comgenvid", "annotated", "Gen", "f.mp4", "evaluation")
            for split, row in (("calibration", calibration), ("evaluation", evaluation)):
                (root / f"{split}_manifest.json").write_text(
                    json.dumps(
                        {
                            "protocol_split": split,
                            "video_count": 1,
                            "videos": [row],
                        }
                    ),
                    encoding="utf-8",
                )
            windows = {
                calibration["video_id"]: [list(range(16))],
                evaluation["video_id"]: [list(range(16))],
            }
            (root / "frame_indices.json").write_text(
                json.dumps({"videos": windows}), encoding="utf-8"
            )
            rows, actual = u0_protocol.load_release_rows(root)
            self.assertEqual(len(rows), 2)
            self.assertEqual(actual, windows)

            payload = json.loads((root / "evaluation_manifest.json").read_text())
            payload["videos"][0]["filename"] = "wrong.mp4"
            (root / "evaluation_manifest.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "video_id does not match"):
                u0_protocol.load_release_rows(root)

    def test_raw_window_shards_validate_keys_and_finite_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for dataset in u0_protocol.U0_DATASETS:
                row = {
                    "video_id": dataset,
                    "dataset": dataset,
                    "protocol_split": "evaluation",
                    "subset": "annotated",
                    "source_model": "Gen",
                    "filename": f"{dataset}.mp4",
                    "window_id": 0,
                    "global_spatial_raw": -1.0,
                    "global_t1_raw": np.inf,
                    "patch_spatial_raw": -2.0,
                    "patch_d2_raw": -3.0,
                }
                rows.append(row)
                pd.DataFrame([row]).to_csv(
                    root / f"{dataset}_shard00_of_01.csv", index=False
                )
            actual = u0_protocol.load_raw_windows(root, 1)
            self.assertEqual(actual["video_id"].tolist(), sorted(row["video_id"] for row in rows))

            broken = pd.DataFrame([rows[0]])
            broken["patch_d2_raw"] = np.nan
            broken.to_csv(root / "comgenvid_shard00_of_01.csv", index=False)
            with self.assertRaisesRegex(ValueError, "non-finite locked raw score"):
                u0_protocol.load_raw_windows(root, 1)

    def test_calibration_reference_membership_and_locked_indices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            all_rows = []
            expected = {}
            for dataset in u0_protocol.U0_DATASETS:
                rows = []
                for index in range(200):
                    identity = f"{dataset}-{index}"
                    frame_indices = list(range(index, index + 16))
                    row = {
                        "video_id": identity,
                        "dataset": dataset,
                        "protocol_split": "calibration",
                        "subset": "real",
                        "source_model": "Real",
                        "filename": f"{identity}.mp4",
                        "effective_k": 1,
                        "frame_indices": json.dumps(frame_indices, separators=(",", ":")),
                    }
                    rows.append(row)
                    all_rows.append(row)
                    expected[identity] = frame_indices
                pd.DataFrame(rows).to_csv(
                    root / f"{dataset}_shard00_of_01.csv", index=False
                )
            references = u0_protocol.load_calibration_references(root, 1)
            self.assertEqual(len(references), 600)
            release = root / "release"
            release.mkdir()
            (release / "frame_indices.json").write_text(
                json.dumps({"calibration_reference_windows": expected}),
                encoding="utf-8",
            )
            u0_protocol.verify_calibration_reference_windows(references, release)
            references.loc[0, "frame_indices"] = "[999]"
            with self.assertRaisesRegex(ValueError, "differ from locked K1 windows"):
                u0_protocol.verify_calibration_reference_windows(references, release)


if __name__ == "__main__":
    unittest.main()
