#!/usr/bin/env python3
"""Audit one feature-cache root and validate every strict entry sidecar."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.cache_contract import (
    CONTRACT_FILENAME,
    entry_metadata_path,
    prepare_feature_cache,
    validate_cache_entry,
)


def repository_path(path: Path, repository_root: Path) -> Path:
    return path if path.is_absolute() else repository_root / path


def verify(
    cache_root: Path,
    *,
    repository_root: Path = ROOT,
    allow_legacy: bool = False,
    verify_cache_hashes: bool = False,
    limit: int | None = None,
) -> dict[str, object]:
    repository_root = repository_root.resolve()
    cache_root = repository_path(cache_root, repository_root).resolve()
    if not cache_root.is_dir():
        raise FileNotFoundError(f"feature cache root does not exist: {cache_root}")
    contract_path = cache_root / CONTRACT_FILENAME
    cache_files = sorted(cache_root.rglob("*.pt")) if cache_root.is_dir() else []
    if not contract_path.is_file():
        if not allow_legacy:
            raise ValueError(
                "legacy cache has no root contract; rerun with --allow-legacy only for "
                "inventory diagnostics, not provenance evidence"
            )
        return {
            "passed": True,
            "mode": "legacy",
            "cache_root": str(cache_root),
            "cache_file_count": len(cache_files),
            "validated_entry_count": 0,
            "strict_evidence": False,
            "full_entry_coverage": False,
            "verify_cache_hashes": False,
        }

    context = prepare_feature_cache(
        cache_root,
        expected_contract=None,
        policy="strict",
        create=False,
    )
    missing_metadata = [
        path for path in cache_files if not entry_metadata_path(path).is_file()
    ]
    if missing_metadata:
        examples = [str(path.relative_to(cache_root)) for path in missing_metadata[:5]]
        raise ValueError(
            f"strict cache contains {len(missing_metadata)} entries without metadata: {examples}"
        )
    expected_metadata = {entry_metadata_path(path).resolve() for path in cache_files}
    actual_metadata = {
        path.resolve() for path in cache_root.rglob(f"*.pt.meta.json")
    }
    orphaned = sorted(actual_metadata - expected_metadata)
    if orphaned:
        examples = [str(path.relative_to(cache_root)) for path in orphaned[:5]]
        raise ValueError(f"strict cache contains orphaned entry metadata: {examples}")

    selected = cache_files if limit is None else cache_files[:limit]
    for cache_path in selected:
        metadata = json.loads(entry_metadata_path(cache_path).read_text(encoding="utf-8"))
        source_path = Path(metadata["source_video"]["path"])
        validate_cache_entry(
            context,
            cache_path=cache_path,
            source_video_path=source_path,
            frame_indices=metadata["frame_indices"],
            verify_cache_sha256=verify_cache_hashes,
        )

    return {
        "passed": True,
        "mode": "strict",
        "cache_root": str(cache_root),
        "cache_kind": context.contract["identity"]["cache_kind"],
        "contract_sha256": context.contract_sha256,
        "cache_file_count": len(cache_files),
        "validated_entry_count": len(selected),
        "strict_evidence": True,
        "full_entry_coverage": limit is None,
        "verify_cache_hashes": verify_cache_hashes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--allow-legacy",
        action="store_true",
        help="Report an uncontracted legacy root without treating it as strict evidence.",
    )
    parser.add_argument("--verify-cache-hashes", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        help="Diagnostic prefix only; limited validation is not complete evidence.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def main() -> None:
    args = parse_args()
    payload = verify(
        args.cache_root,
        allow_legacy=args.allow_legacy,
        verify_cache_hashes=args.verify_cache_hashes,
        limit=args.limit,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            "feature cache verification passed: "
            f"mode={payload['mode']} files={payload['cache_file_count']} "
            f"validated={payload['validated_entry_count']} "
            f"strict_evidence={payload['strict_evidence']} "
            f"full_entry_coverage={payload['full_entry_coverage']}"
        )


if __name__ == "__main__":
    main()
