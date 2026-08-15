"""Canonical dataset inventory and locked-release membership auditing."""

from __future__ import annotations

import copy
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .experiment_registry import IDENTIFIER
from .release_io import sha256_file, video_id


SCHEMA_VERSION = "alpha_stalled_data_catalog_v1"
SPEC_SCHEMA_VERSION = "alpha_stalled_data_catalog_spec_v1"
REQUIRED_INDEX_COLUMNS = (
    "video_path",
    "subset",
    "source_model",
    "fps",
    "duration_seconds",
    "num_frames",
    "downsample_idxs",
    "1_sec_idxs",
    "2_sec_idxs",
)


@dataclass(frozen=True)
class DataCatalogSummary:
    snapshot_id: str
    dataset_count: int
    canonical_video_count: int
    release_video_count: int
    missing_file_count: int


def _required(mapping: Mapping[str, Any], fields: tuple[str, ...], context: str) -> None:
    missing = [field for field in fields if field not in mapping]
    if missing:
        raise ValueError(f"{context} missing fields: {missing}")


def _relative_file(value: Any, repository_root: Path, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a nonempty repository-relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{context} must be repository-relative")
    path = repository_root / relative
    if not path.is_file():
        raise ValueError(f"{context} is not a file: {relative}")
    return path


def _relative_directory(value: Any, repository_root: Path, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a nonempty repository-relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{context} must be repository-relative")
    path = repository_root / relative
    if not path.is_dir():
        raise ValueError(f"{context} is not a directory: {relative}")
    return path


def _load_release_manifest(path: Path, split: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _required(
        payload,
        ("schema_version", "protocol_split", "video_count", "dataset_counts", "videos"),
        f"release manifest {path.name}",
    )
    if payload["protocol_split"] != split:
        raise ValueError(f"{path.name} protocol_split is not {split!r}")
    videos = payload["videos"]
    if not isinstance(videos, list) or len(videos) != payload["video_count"]:
        raise ValueError(f"{path.name} video_count does not match videos")
    for index, row in enumerate(videos):
        _required(
            row,
            ("video_id", "dataset", "subset", "source_model", "filename"),
            f"{path.name}.videos[{index}]",
        )
        if row.get("protocol_split") != split:
            raise ValueError(f"{path.name}.videos[{index}] has wrong protocol_split")
        if row["video_id"] != video_id(row):
            raise ValueError(f"{path.name}.videos[{index}] video_id does not match identity fields")
    actual_counts = Counter(str(row["dataset"]) for row in videos)
    if dict(sorted(actual_counts.items())) != payload["dataset_counts"]:
        raise ValueError(f"{path.name} dataset_counts do not match videos")
    return videos


def _source_file_exists(value: Any, repository_root: Path) -> bool:
    path = Path(str(value))
    if path.is_absolute():
        return path.is_file()
    return any(
        candidate.is_file()
        for candidate in (repository_root / path, repository_root.parent / path)
    )


def _validate_spec(spec: Mapping[str, Any], repository_root: Path) -> None:
    _required(spec, ("schema_version", "snapshot_id", "release", "datasets"), "catalog")
    if spec["schema_version"] != SPEC_SCHEMA_VERSION:
        raise ValueError(f"unsupported data catalog spec schema: {spec['schema_version']!r}")
    snapshot_id = spec["snapshot_id"]
    if not isinstance(snapshot_id, str) or not IDENTIFIER.fullmatch(snapshot_id):
        raise ValueError("snapshot_id must be a stable lowercase identifier")
    release = spec["release"]
    if not isinstance(release, Mapping):
        raise ValueError("release must be an object")
    _required(
        release,
        ("protocol_id", "calibration_manifest", "evaluation_manifest"),
        "release",
    )
    if not isinstance(release["protocol_id"], str) or not release["protocol_id"]:
        raise ValueError("release.protocol_id must be nonempty")
    _relative_file(release["calibration_manifest"], repository_root, "release.calibration_manifest")
    _relative_file(release["evaluation_manifest"], repository_root, "release.evaluation_manifest")

    datasets = spec["datasets"]
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("datasets must be a nonempty list")
    identifiers: set[str] = set()
    paths: set[str] = set()
    for index, declaration in enumerate(datasets):
        context = f"datasets[{index}]"
        if not isinstance(declaration, Mapping):
            raise ValueError(f"{context} must be an object")
        _required(
            declaration,
            ("dataset_id", "canonical_index", "dataset_root", "role", "notes"),
            context,
        )
        dataset_id = declaration["dataset_id"]
        if not isinstance(dataset_id, str) or not IDENTIFIER.fullmatch(dataset_id):
            raise ValueError(f"{context}.dataset_id is invalid")
        if dataset_id in identifiers:
            raise ValueError(f"duplicate dataset_id: {dataset_id}")
        identifiers.add(dataset_id)
        index_path = declaration["canonical_index"]
        if index_path in paths:
            raise ValueError(f"duplicate canonical_index: {index_path}")
        paths.add(index_path)
        _relative_file(index_path, repository_root, f"{context}.canonical_index")
        _relative_directory(declaration["dataset_root"], repository_root, f"{context}.dataset_root")
        for field in ("role", "notes"):
            if not isinstance(declaration[field], str) or not declaration[field].strip():
                raise ValueError(f"{context}.{field} must be nonempty")


def _index_frame(
    declaration: Mapping[str, Any], repository_root: Path
) -> tuple[pd.DataFrame, Path]:
    index_path = repository_root / declaration["canonical_index"]
    frame = pd.read_csv(index_path, float_precision="round_trip")
    missing = sorted(set(REQUIRED_INDEX_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{index_path.name} missing columns: {missing}")
    frame = frame.copy()
    frame["dataset"] = declaration["dataset_id"]
    frame["filename"] = frame["video_path"].map(lambda value: Path(str(value)).name)
    frame["video_id"] = frame.apply(video_id, axis=1)
    duplicates = frame[frame["video_id"].duplicated(keep=False)]
    if not duplicates.empty:
        raise ValueError(
            f"{index_path.name} has duplicate canonical identities: {len(duplicates)} rows"
        )
    return frame, index_path


def _source_rows(frame: pd.DataFrame, release_split: Mapping[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(["subset", "source_model"], sort=True, dropna=False)
    for (subset, source_model), group in grouped:
        split_counts = Counter(
            release_split.get(video_id_value, "not_in_locked_release")
            for video_id_value in group["video_id"]
        )
        rows.append(
            {
                "subset": str(subset),
                "source_model": str(source_model),
                "video_count": int(len(group)),
                "duration_ge_1s_count": int((group["duration_seconds"] >= 1.0).sum()),
                "duration_ge_2s_count": int((group["duration_seconds"] >= 2.0).sum()),
                "native_fps_ge_8_count": int((group["fps"] >= 8.0).sum()),
                "locked_calibration_count": int(split_counts["calibration"]),
                "locked_evaluation_count": int(split_counts["evaluation"]),
                "not_in_locked_release_count": int(split_counts["not_in_locked_release"]),
            }
        )
    return rows


def build_data_catalog(spec: Mapping[str, Any], repository_root: Path) -> dict[str, Any]:
    """Build a deterministic catalog from canonical indexes and release manifests."""

    _validate_spec(spec, repository_root)
    release = spec["release"]
    calibration_path = repository_root / release["calibration_manifest"]
    evaluation_path = repository_root / release["evaluation_manifest"]
    calibration = _load_release_manifest(calibration_path, "calibration")
    evaluation = _load_release_manifest(evaluation_path, "evaluation")
    all_release = calibration + evaluation
    release_ids = [str(row["video_id"]) for row in all_release]
    if len(release_ids) != len(set(release_ids)):
        raise ValueError("locked calibration/evaluation manifests overlap or contain duplicates")
    release_split = {
        str(row["video_id"]): str(row["protocol_split"]) for row in all_release
    }
    release_dataset = {
        str(row["video_id"]): str(row["dataset"]) for row in all_release
    }

    catalog = copy.deepcopy(dict(spec))
    catalog["schema_version"] = SCHEMA_VERSION
    catalog["release"]["calibration_manifest_sha256"] = sha256_file(calibration_path)
    catalog["release"]["evaluation_manifest_sha256"] = sha256_file(evaluation_path)
    catalog_datasets: list[dict[str, Any]] = []
    all_catalog_ids: set[str] = set()
    total_missing = 0
    total_rows = 0
    for declaration in spec["datasets"]:
        frame, index_path = _index_frame(declaration, repository_root)
        dataset_id = declaration["dataset_id"]
        ids = set(frame["video_id"])
        cross_dataset_overlap = all_catalog_ids & ids
        if cross_dataset_overlap:
            raise ValueError(
                f"canonical identities overlap across datasets: {len(cross_dataset_overlap)}"
            )
        all_catalog_ids.update(ids)
        wrong_dataset = [
            value for value in ids if value in release_dataset and release_dataset[value] != dataset_id
        ]
        if wrong_dataset:
            raise ValueError(f"release identities assigned to wrong dataset: {len(wrong_dataset)}")

        paths_present = [
            _source_file_exists(value, repository_root) for value in frame["video_path"]
        ]
        missing_count = int(len(paths_present) - sum(paths_present))
        total_missing += missing_count
        total_rows += len(frame)
        split_counts = Counter(release_split.get(value, "not_in_locked_release") for value in ids)
        generated_count = int((frame["subset"] != "real").sum())
        stats = {
            "index_sha256": sha256_file(index_path),
            "canonical_video_count": int(len(frame)),
            "unique_video_id_count": int(frame["video_id"].nunique()),
            "real_video_count": int((frame["subset"] == "real").sum()),
            "generated_video_count": generated_count,
            "source_count": int(frame.groupby(["subset", "source_model"]).ngroups),
            "duration_ge_1s_count": int((frame["duration_seconds"] >= 1.0).sum()),
            "duration_ge_2s_count": int((frame["duration_seconds"] >= 2.0).sum()),
            "native_fps_ge_8_count": int((frame["fps"] >= 8.0).sum()),
            "one_second_index_count": int(frame["1_sec_idxs"].notna().sum()),
            "two_second_index_count": int(frame["2_sec_idxs"].notna().sum()),
            "source_file_present_count": int(sum(paths_present)),
            "source_file_missing_count": missing_count,
            "locked_calibration_count": int(split_counts["calibration"]),
            "locked_evaluation_count": int(split_counts["evaluation"]),
            "not_in_locked_release_count": int(split_counts["not_in_locked_release"]),
        }
        item = copy.deepcopy(dict(declaration))
        item["stats"] = stats
        item["sources"] = _source_rows(frame, release_split)
        catalog_datasets.append(item)

    absent_release = sorted(set(release_ids).difference(all_catalog_ids))
    if absent_release:
        raise ValueError(
            f"locked release identities absent from canonical indexes: {len(absent_release)}"
        )
    catalog["datasets"] = catalog_datasets
    catalog["summary"] = {
        "dataset_count": len(catalog_datasets),
        "canonical_video_count": int(total_rows),
        "unique_video_id_count": len(all_catalog_ids),
        "release_video_count": len(release_ids),
        "release_calibration_count": len(calibration),
        "release_evaluation_count": len(evaluation),
        "not_in_locked_release_count": int(total_rows - len(release_ids)),
        "source_file_missing_count": total_missing,
        "release_id_absent_from_catalog_count": 0,
    }
    validate_data_catalog(catalog, repository_root=repository_root, verify_inputs=False)
    return catalog


def validate_data_catalog(
    catalog: Mapping[str, Any],
    *,
    repository_root: Path,
    verify_inputs: bool = True,
) -> DataCatalogSummary:
    """Validate catalog structure and optionally rebuild it from source indexes."""

    _required(catalog, ("schema_version", "snapshot_id", "release", "datasets", "summary"), "catalog")
    if catalog["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported data catalog schema: {catalog['schema_version']!r}")
    summary = catalog["summary"]
    _required(
        summary,
        (
            "dataset_count",
            "canonical_video_count",
            "unique_video_id_count",
            "release_video_count",
            "release_calibration_count",
            "release_evaluation_count",
            "not_in_locked_release_count",
            "source_file_missing_count",
            "release_id_absent_from_catalog_count",
        ),
        "summary",
    )
    if summary["dataset_count"] != len(catalog["datasets"]):
        raise ValueError("summary.dataset_count does not match datasets")
    if summary["canonical_video_count"] != summary["unique_video_id_count"]:
        raise ValueError("canonical catalog contains duplicate video IDs")
    if summary["release_video_count"] != (
        summary["release_calibration_count"] + summary["release_evaluation_count"]
    ):
        raise ValueError("release split counts do not sum to release_video_count")
    if summary["canonical_video_count"] != (
        summary["release_video_count"] + summary["not_in_locked_release_count"]
    ):
        raise ValueError("release and non-release counts do not cover canonical catalog")
    if summary["release_id_absent_from_catalog_count"] != 0:
        raise ValueError("catalog does not cover every locked release identity")

    if verify_inputs:
        spec = copy.deepcopy(dict(catalog))
        spec["schema_version"] = SPEC_SCHEMA_VERSION
        spec.pop("summary", None)
        spec["release"].pop("calibration_manifest_sha256", None)
        spec["release"].pop("evaluation_manifest_sha256", None)
        for dataset in spec["datasets"]:
            dataset.pop("stats", None)
            dataset.pop("sources", None)
        rebuilt = build_data_catalog(spec, repository_root)
        if rebuilt != catalog:
            raise ValueError("data catalog is stale relative to canonical indexes or manifests")

    return DataCatalogSummary(
        snapshot_id=str(catalog["snapshot_id"]),
        dataset_count=int(summary["dataset_count"]),
        canonical_video_count=int(summary["canonical_video_count"]),
        release_video_count=int(summary["release_video_count"]),
        missing_file_count=int(summary["source_file_missing_count"]),
    )


def render_data_catalog(catalog: Mapping[str, Any]) -> str:
    """Render a deterministic human-readable catalog report."""

    summary = catalog["summary"]
    lines = [
        "# Dataset catalog",
        "",
        "This report is generated from `configs/data_catalog.yaml`, the three canonical index",
        "CSVs, and the locked U0 calibration/evaluation manifests. It inventories identities",
        "and protocol membership; it is not a content-hash manifest for 115 GiB of source video.",
        "",
        "## Summary",
        "",
        f"- Snapshot: `{catalog['snapshot_id']}`",
        f"- Canonical video identities: {summary['canonical_video_count']:,}",
        f"- Locked U0 identities: {summary['release_video_count']:,} "
        f"({summary['release_calibration_count']:,} calibration + "
        f"{summary['release_evaluation_count']:,} evaluation)",
        f"- Canonical identities outside locked U0: {summary['not_in_locked_release_count']:,}",
        f"- Missing source files on this workspace: {summary['source_file_missing_count']:,}",
        "",
        "## Canonical datasets",
        "",
        "| Dataset | Index rows | Real | Generated | Sources | >=2 s | U0 calib | U0 eval | Outside U0 | Missing |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in catalog["datasets"]:
        stats = dataset["stats"]
        lines.append(
            f"| {dataset['dataset_id']} | {stats['canonical_video_count']:,} | "
            f"{stats['real_video_count']:,} | {stats['generated_video_count']:,} | "
            f"{stats['source_count']:,} | {stats['duration_ge_2s_count']:,} | "
            f"{stats['locked_calibration_count']:,} | {stats['locked_evaluation_count']:,} | "
            f"{stats['not_in_locked_release_count']:,} | {stats['source_file_missing_count']:,} |"
        )
    lines.extend(
        [
            "",
            "## Source coverage",
            "",
            "| Dataset | Subset | Source | Videos | >=2 s | U0 calib | U0 eval | Outside U0 |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in catalog["datasets"]:
        for source in dataset["sources"]:
            lines.append(
                f"| {dataset['dataset_id']} | {source['subset']} | {source['source_model']} | "
                f"{source['video_count']:,} | {source['duration_ge_2s_count']:,} | "
                f"{source['locked_calibration_count']:,} | "
                f"{source['locked_evaluation_count']:,} | "
                f"{source['not_in_locked_release_count']:,} |"
            )
    lines.extend(
        [
            "",
            "## Identity and evidence boundary",
            "",
            "- A canonical `video_id` is SHA-256 of `dataset|subset|source_model|filename`.",
            "- Every locked calibration/evaluation identity must occur exactly once in a canonical index.",
            "- Index and locked-manifest SHA-256 values are stored in the machine-readable catalog.",
            "- Source-file presence is audited, but source video bytes are not hashed by this catalog.",
            "- Derived calibration/holdout CSVs remain experiment artifacts, not additional canonical datasets.",
            "- `not_in_locked_release` means available in a canonical index but unused by locked U0; it does not mean rejected or invalid.",
            "",
            "Rebuild/check with:",
            "",
            "```bash",
            "conda run --no-capture-output -n stall python tools/build_data_catalog.py --check",
            "conda run --no-capture-output -n stall python tools/verify_data_catalog.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "DataCatalogSummary",
    "SCHEMA_VERSION",
    "SPEC_SCHEMA_VERSION",
    "build_data_catalog",
    "render_data_catalog",
    "validate_data_catalog",
]
