import json
import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import build_duration_aware_23source_protocol as protocol_cli
from alpha_stalled.duration_aware_protocol import (
    CALIBRATION_SIZES,
    DATASETS,
    N200_INDEX,
    add_identity,
    build_dataset,
    calibration_size_splits,
    validate,
)


class DurationAwareProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        frames = []
        cls.summaries = {}
        for dataset, spec in DATASETS.items():
            frame, summary = build_dataset(dataset, spec)
            frames.append(frame)
            cls.summaries[dataset] = summary
        cls.tasks = pd.concat(frames, ignore_index=True)

    def test_expected_partition_counts(self) -> None:
        self.assertEqual(self.summaries["comgenvid"]["calibration_real"], 800)
        self.assertEqual(self.summaries["comgenvid"]["evaluation_real"], 898)
        self.assertEqual(self.summaries["comgenvid"]["excluded_real_too_short"], 2)
        self.assertEqual(self.summaries["videofeedback"]["calibration_real"], 500)
        self.assertEqual(self.summaries["videofeedback"]["evaluation_real"], 3580)
        self.assertEqual(self.summaries["genvideo"]["calibration_real"], 2000)
        self.assertEqual(self.summaries["genvideo"]["evaluation_real"], 7984)

    def test_cli_reexports_shared_protocol_objects(self) -> None:
        self.assertIs(protocol_cli.DATASETS, DATASETS)
        self.assertIs(protocol_cli.CALIBRATION_SIZES, CALIBRATION_SIZES)
        self.assertIs(protocol_cli.add_identity, add_identity)
        self.assertIs(protocol_cli.build_dataset, build_dataset)
        self.assertIs(protocol_cli.validate, validate)

    def test_all_23_sources_are_present(self) -> None:
        fake = self.tasks[
            self.tasks["protocol_split"].eq("evaluation")
            & self.tasks["subset"].eq("annotated")
        ]
        self.assertEqual(fake.groupby(["dataset", "source_model"]).ngroups, 23)
        self.assertEqual(len(fake), 45185)

    def test_short_sources_use_one_second(self) -> None:
        fake = self.tasks[
            self.tasks["protocol_split"].eq("evaluation")
            & self.tasks["subset"].eq("annotated")
        ]
        expected = {
            ("videofeedback", "Hotshot-XL"),
            ("genvideo", "HotShot"),
            ("genvideo", "MoonValley"),
        }
        actual = set(
            fake[fake["protocol_duration_sec"].eq(1)]
            .groupby(["dataset", "source_model"])
            .groups
        )
        self.assertTrue(expected.issubset(actual))
        for key in expected:
            values = fake[
                fake["dataset"].eq(key[0]) & fake["source_model"].eq(key[1])
            ]["protocol_duration_sec"]
            self.assertEqual(set(values), {1})

    def test_frame_counts_and_no_overlap(self) -> None:
        validate(self.tasks)
        calibration = set(
            self.tasks[self.tasks["protocol_split"].eq("calibration")]["video_id"]
        )
        evaluation = set(
            self.tasks[self.tasks["protocol_split"].eq("evaluation")]["video_id"]
        )
        self.assertFalse(calibration & evaluation)
        for row in self.tasks.itertuples(index=False):
            windows = json.loads(row.frame_indices)
            self.assertTrue(all(len(window) == 8 * row.protocol_duration_sec for window in windows))

    def test_n200_is_nested_in_maximum_calibration(self) -> None:
        for dataset, spec in DATASETS.items():
            maximum = add_identity(pd.read_csv(spec["calibration"]), dataset)
            n200 = add_identity(pd.read_csv(N200_INDEX[dataset]), dataset)
            self.assertEqual(len(set(n200["video_id"])), 200)
            self.assertTrue(set(n200["video_id"]) <= set(maximum["video_id"]))

    def test_all_calibration_curve_sizes_are_nested(self) -> None:
        for dataset, spec in DATASETS.items():
            maximum = add_identity(pd.read_csv(spec["calibration"]), dataset)
            splits = calibration_size_splits(dataset, maximum)
            self.assertEqual(tuple(splits), CALIBRATION_SIZES[dataset])
            previous = set()
            for size, selected in splits.items():
                self.assertEqual(len(selected), size)
                self.assertTrue(previous <= selected)
                previous = selected

    def test_known_d325_bad_tail_window_is_excluded(self) -> None:
        task = self.tasks[
            self.tasks["dataset"].eq("genvideo")
            & self.tasks["filename"].eq("D325.mp4")
            & self.tasks["protocol_split"].eq("evaluation")
        ]
        self.assertEqual(len(task), 1)
        self.assertEqual(int(task.iloc[0]["effective_k"]), 2)
        windows = json.loads(task.iloc[0]["frame_indices"])
        self.assertNotIn(99, {value for window in windows for value in window})


if __name__ == "__main__":
    unittest.main()
