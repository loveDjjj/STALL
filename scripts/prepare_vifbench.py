#!/usr/bin/env python3
"""审计 ViF-Bench 官方 MP4，并建立无语义配对泄漏的确认集 manifest。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_manifest import compute_windows, downsample_frames, get_video_metadata


EXPECTED_ARCHIVE_SHA256 = (
    "41e79dff9f8ff16f7bcddd99eb18e4019b52f5ed9fe26a6a0fa0739b085907a8"
)
SOURCE_FILE_COMMIT = "67e11331e26dbf732e0b8c7d4dc52f4442d174dd"
CALIBRATION_REAL_COUNT = 80
SPLIT_SEED = 17


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_key(name: str) -> str:
    return hashlib.sha256(f"{SPLIT_SEED}\0{name}".encode("utf-8")).hexdigest()


def _probe(paths: list[Path], workers: int) -> dict[Path, dict]:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        values = list(executor.map(lambda path: get_video_metadata(str(path)), paths))
    return dict(zip(paths, values))


def _manifest_row(path: Path, subset: str, source_model: str, metadata: dict) -> dict:
    fps = float(metadata["fps"])
    duration = float(metadata["duration_seconds"])
    num_frames = int(metadata["num_frames"])
    downsample = downsample_frames(num_frames, fps, 8.0)
    windows = compute_windows(downsample, target_fps=8.0, seed=42)
    relative = path.relative_to(ROOT).as_posix()
    return {
        "video_path": relative,
        "subset": subset,
        "source_model": source_model,
        "fps": fps,
        "duration_seconds": duration,
        "num_frames": num_frames,
        "downsample_idxs": json.dumps(downsample),
        **{
            key: json.dumps(value) if value is not None else None
            for key, value in windows.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "datasets/vifbench_download/source_videos",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT / "datasets/vifbench_download/source_videos.zip",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/manifests/confirmation",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    outputs = (
        args.output_dir / "vifbench_calibration.csv",
        args.output_dir / "vifbench_evaluation.csv",
        args.output_dir / "vifbench_protocol.json",
    )
    if any(path.exists() for path in outputs) and not args.overwrite:
        raise FileExistsError("ViF-Bench manifest已存在；重建请传--overwrite")
    if _sha256(args.archive) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("ViF-Bench source_videos.zip SHA256与官方LFS对象不一致")

    real_paths = sorted((args.dataset_root / "Real").glob("*.mp4"))
    fake_paths = sorted((args.dataset_root / "Fake").glob("*/*.mp4"))
    if len(real_paths) != 165 or len(fake_paths) != 2995:
        raise ValueError(
            f"ViF-Bench文件数异常：real={len(real_paths)}, fake={len(fake_paths)}"
        )
    all_paths = [*real_paths, *fake_paths]
    metadata = _probe(all_paths, args.workers)

    def usable(path: Path) -> bool:
        value = metadata[path]
        return (
            value.get("fps") is not None
            and float(value["fps"]) >= 8.0
            and value.get("duration_seconds") is not None
            and float(value["duration_seconds"]) >= 2.0
            and int(value.get("num_frames") or 0) >= 16
        )

    usable_real = sorted((path for path in real_paths if usable(path)), key=lambda p: _split_key(p.name))
    if len(usable_real) < CALIBRATION_REAL_COUNT + 2:
        raise ValueError(f"ViF-Bench可用real不足：{len(usable_real)}")
    calibration_names = {path.name for path in usable_real[:CALIBRATION_REAL_COUNT]}
    evaluation_real = sorted(usable_real[CALIBRATION_REAL_COUNT:], key=lambda p: p.name)
    evaluation_names = {path.name for path in evaluation_real}
    calibration_real = sorted(usable_real[:CALIBRATION_REAL_COUNT], key=lambda p: p.name)

    evaluation_fake = [
        path for path in fake_paths
        if path.name in evaluation_names and usable(path)
    ]
    calibration_rows = [
        _manifest_row(path, "real", "ViF-Bench-Real", metadata[path])
        for path in calibration_real
    ]
    evaluation_rows = [
        *[
            _manifest_row(path, "real", "ViF-Bench-Real", metadata[path])
            for path in evaluation_real
        ],
        *[
            _manifest_row(path, "annotated", path.parent.name, metadata[path])
            for path in evaluation_fake
        ],
    ]
    calibration = pd.DataFrame(calibration_rows)
    evaluation = pd.DataFrame(evaluation_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(outputs[0], index=False)
    evaluation.to_csv(outputs[1], index=False)

    fake_all_names = {path.name for path in fake_paths}
    protocol = {
        "schema_version": "vifbench_confirmation_protocol_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": "JoeLeelyf/ViF-Bench",
            "source_file": "source_videos.zip",
            "source_file_commit": SOURCE_FILE_COMMIT,
            "archive_sha256": EXPECTED_ARCHIVE_SHA256,
        },
        "split": {
            "seed": SPLIT_SEED,
            "rule": "sha256(seed\\0filename), first 80 usable real for calibration",
            "semantic_pair_guard": (
                "evaluation fake filename must belong to evaluation real filenames"
            ),
            "calibration_real": len(calibration),
            "evaluation_real": int(evaluation["subset"].eq("real").sum()),
            "evaluation_fake": int(evaluation["subset"].eq("annotated").sum()),
            "calibration_real_names_sha256": hashlib.sha256(
                "\n".join(sorted(calibration_names)).encode("utf-8")
            ).hexdigest(),
            "evaluation_real_names_sha256": hashlib.sha256(
                "\n".join(sorted(evaluation_names)).encode("utf-8")
            ).hexdigest(),
        },
        "exclusions": {
            "unusable_real": len(real_paths) - len(usable_real),
            "unusable_fake": sum(not usable(path) for path in fake_paths),
            "fake_without_matching_real_filename": len(
                fake_all_names.difference({path.name for path in real_paths})
            ),
            "fake_matching_calibration_real_removed": sum(
                path.name in calibration_names for path in fake_paths
            ),
        },
        "evaluation_per_generator": {
            str(name): int(count)
            for name, count in evaluation[evaluation["subset"].eq("annotated")]
            .groupby("source_model", sort=True).size().items()
        },
        "manifest_sha256": {
            outputs[0].name: _sha256(outputs[0]),
            outputs[1].name: _sha256(outputs[1]),
        },
    }
    outputs[2].write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(protocol, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
