"""Versioned identity and per-file provenance for future feature caches."""

from __future__ import annotations

import hashlib
import json
import subprocess
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from .integrity import sha256_file, write_json


CONTRACT_SCHEMA_VERSION = "stall_feature_cache_contract_v1"
ENTRY_SCHEMA_VERSION = "stall_feature_cache_entry_v1"
CONTRACT_FILENAME = ".stall_cache_contract.json"
ENTRY_SUFFIX = ".meta.json"
CACHE_KINDS = frozenset({"global_embeddings", "patch_embeddings"})
CACHE_POLICIES = frozenset({"auto", "strict", "legacy"})


class LegacyFeatureCacheWarning(UserWarning):
    """Raised when an existing cache has no identity contract."""


@dataclass(frozen=True)
class CacheContractContext:
    root: Path
    policy: str
    strict: bool
    contract: dict[str, Any] | None
    contract_sha256: str | None


@dataclass(frozen=True)
class CacheEntrySummary:
    cache_path: Path
    metadata_path: Path
    frame_count: int
    source_sha256: str
    cache_sha256: str


_WARNED_LEGACY_ROOTS: set[str] = set()


def _required(mapping: Mapping[str, Any], fields: Sequence[str], context: str) -> None:
    missing = [field for field in fields if field not in mapping]
    if missing:
        raise ValueError(f"{context} missing fields: {missing}")


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a nonempty string")
    return value.strip()


def _nonnegative_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a nonnegative integer")
    return value


def _sha256(value: Any, context: str) -> str:
    digest = _string(value, context)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return digest


def canonical_sha256(payload: Mapping[str, Any] | Sequence[Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=16)
def _sha256_at_stat(path: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    return sha256_file(path)


def _fingerprint_file(path: Path) -> tuple[int, int, str]:
    stat = path.stat()
    return (
        stat.st_size,
        stat.st_mtime_ns,
        _sha256_at_stat(str(path.resolve()), stat.st_size, stat.st_mtime_ns),
    )


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.rstrip("\n")


def _tracked_git_identity(repository: Path) -> dict[str, Any]:
    commit = _git(repository, "rev-parse", "HEAD")
    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=no")
    if status:
        raise ValueError(
            f"strict feature caches require clean tracked DINOv3 code: {repository}"
        )
    return {"commit": commit, "tracked_worktree_dirty": False}


def _extractor_source_hashes(cache_kind: str) -> dict[str, str]:
    """返回决定当前缓存内容的实现文件哈希。

    缓存契约必须绑定现在仍在主干中实际执行的代码。这里不能继续引用
    重构前的 ``alpha_stalled`` 模块，否则首次严格建缓存就会因找不到历史
    文件而失败。
    """

    source_root = Path(__file__).resolve().parents[1]
    package_root = Path(__file__).resolve().parent
    paths = [
        source_root / "features.py",
        package_root / "video.py",
        package_root / "patch_cache.py",
    ]
    return {path.name: sha256_file(path) for path in paths}


def build_feature_cache_contract(
    model: Any,
    *,
    cache_kind: str,
    frame_batch_size: int,
    video_batch_size: int,
) -> dict[str, Any]:
    """Build the immutable root contract for a model-backed feature cache."""

    if cache_kind not in CACHE_KINDS:
        raise ValueError(f"unsupported cache_kind: {cache_kind!r}")
    frame_batch_size = _nonnegative_integer(frame_batch_size, "frame_batch_size")
    video_batch_size = _nonnegative_integer(video_batch_size, "video_batch_size")
    if frame_batch_size == 0 or video_batch_size == 0:
        raise ValueError("frame_batch_size and video_batch_size must be positive")
    if getattr(model, "model", None) is None:
        raise ValueError("strict cache contract requires a loaded DINOv3 model")
    repo_path = Path(_string(getattr(model, "dino_repo_path", None), "model.dino_repo_path"))
    weights_path = Path(
        _string(getattr(model, "dino_weights_path", None), "model.dino_weights_path")
    )
    if not repo_path.is_dir() or not weights_path.is_file():
        raise ValueError("DINOv3 repository or checkpoint path no longer exists")
    checkpoint_size, _, checkpoint_sha = _fingerprint_file(weights_path)
    dino_identity = _tracked_git_identity(repo_path)

    core_model = model.model.module if hasattr(model.model, "module") else model.model
    blocks = getattr(core_model, "blocks", None)
    block_count = len(blocks) if blocks is not None else 24
    feature_dimension = int(getattr(core_model, "embed_dim", 1024))
    output_tokens = ["normalized_final_cls"]
    if cache_kind == "patch_embeddings":
        output_tokens.append("normalized_final_patch_grid")

    import torch
    import torchvision

    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "identity": {
            "cache_kind": cache_kind,
            "encoder": {
                "implementation": "dinov3_vitl16",
                "repository_commit": dino_identity["commit"],
                "repository_tracked_worktree_dirty": False,
                "checkpoint_filename": weights_path.name,
                "checkpoint_bytes": checkpoint_size,
                "checkpoint_sha256": checkpoint_sha,
                "output_layer": block_count - 1,
                "output_tokens": output_tokens,
                "feature_dimension": feature_dimension,
                "feature_dtype": "float32",
                "cls_register_policy": "cls_separate_register_tokens_excluded",
            },
            "preprocessing": {
                "input_channel_order": "BGR_from_OpenCV_then_RGB",
                "tensor_conversion": "torchvision.transforms.ToTensor",
                "resize": [224, 224],
                "resize_antialias": True,
                "normalization_mean": [0.485, 0.456, 0.406],
                "normalization_std": [0.229, 0.224, 0.225],
            },
            "extraction": {
                "frame_selection": "external_native_frame_indices",
                "frame_grouping": "cross_video_flatten_then_split",
                "frame_batch_size": frame_batch_size,
                "video_batch_size": video_batch_size,
                "extractor_source_sha256": _extractor_source_hashes(cache_kind),
                "torch_version": torch.__version__,
                "torchvision_version": torchvision.__version__,
            },
            "layout": {
                "path_key": "subset/source_model/video_stem[_duration_s].pt",
                "duration_and_frame_indices": "per_entry_metadata",
                "source_video_identity": "per_entry_sha256",
            },
        },
    }


def validate_feature_cache_contract(contract: Mapping[str, Any]) -> str:
    _required(contract, ("schema_version", "identity"), "contract")
    if contract["schema_version"] != CONTRACT_SCHEMA_VERSION:
        raise ValueError(f"unsupported cache contract schema: {contract['schema_version']!r}")
    identity = contract["identity"]
    if not isinstance(identity, Mapping):
        raise ValueError("contract.identity must be an object")
    _required(identity, ("cache_kind", "encoder", "preprocessing", "extraction", "layout"), "identity")
    if identity["cache_kind"] not in CACHE_KINDS:
        raise ValueError("contract identity has unsupported cache kind")
    encoder = identity["encoder"]
    if not isinstance(encoder, Mapping):
        raise ValueError("identity.encoder must be an object")
    _required(
        encoder,
        (
            "implementation",
            "repository_commit",
            "repository_tracked_worktree_dirty",
            "checkpoint_filename",
            "checkpoint_bytes",
            "checkpoint_sha256",
            "output_layer",
            "output_tokens",
            "feature_dimension",
            "feature_dtype",
            "cls_register_policy",
        ),
        "identity.encoder",
    )
    _sha256(encoder["checkpoint_sha256"], "identity.encoder.checkpoint_sha256")
    if encoder["repository_tracked_worktree_dirty"] is not False:
        raise ValueError("strict cache contract cannot use dirty tracked DINOv3 code")
    for field in ("checkpoint_bytes", "output_layer", "feature_dimension"):
        _nonnegative_integer(encoder[field], f"identity.encoder.{field}")
    extraction = identity["extraction"]
    if not isinstance(extraction, Mapping):
        raise ValueError("identity.extraction must be an object")
    _required(
        extraction,
        (
            "frame_selection",
            "frame_grouping",
            "frame_batch_size",
            "video_batch_size",
            "extractor_source_sha256",
            "torch_version",
            "torchvision_version",
        ),
        "identity.extraction",
    )
    for field in ("frame_batch_size", "video_batch_size"):
        if _nonnegative_integer(extraction[field], f"identity.extraction.{field}") == 0:
            raise ValueError(f"identity.extraction.{field} must be positive")
    source_hashes = extraction["extractor_source_sha256"]
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ValueError("identity.extraction.extractor_source_sha256 must be an object")
    for name, digest in source_hashes.items():
        _string(name, "extractor source name")
        _sha256(digest, f"extractor source {name}")
    return canonical_sha256(contract)


def _read_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("feature cache contract must be a JSON object")
    return payload


def _has_pt_files(root: Path) -> bool:
    return root.is_dir() and next(root.rglob("*.pt"), None) is not None


def cache_policy_uses_strict_entries(root: Path | str, policy: str) -> bool:
    """Resolve whether hit detection must require per-entry metadata."""
    if policy not in CACHE_POLICIES:
        raise ValueError(f"unsupported cache policy: {policy!r}")
    root = Path(root)
    if policy == "strict":
        return True
    if policy == "legacy":
        return False
    return (root / CONTRACT_FILENAME).is_file() or not _has_pt_files(root)


def _warn_legacy(root: Path) -> None:
    key = str(root.resolve())
    if key not in _WARNED_LEGACY_ROOTS:
        warnings.warn(
            f"legacy feature cache has no identity contract: {root}; "
            "reuse is not provenance-safe",
            LegacyFeatureCacheWarning,
            stacklevel=3,
        )
        _WARNED_LEGACY_ROOTS.add(key)


def prepare_feature_cache(
    root: Path | str,
    *,
    expected_contract: Mapping[str, Any] | None,
    policy: str = "auto",
    create: bool = False,
    required_cache_kind: str | None = None,
) -> CacheContractContext:
    """Resolve strict/legacy mode and create or compare an immutable contract."""

    if policy not in CACHE_POLICIES:
        raise ValueError(f"unsupported cache policy: {policy!r}")
    if required_cache_kind is not None and required_cache_kind not in CACHE_KINDS:
        raise ValueError(f"unsupported required_cache_kind: {required_cache_kind!r}")
    root = Path(root)
    contract_path = root / CONTRACT_FILENAME
    has_contract = contract_path.is_file()
    has_payloads = _has_pt_files(root)

    if policy == "legacy":
        if has_contract:
            raise ValueError("contract-backed cache cannot be opened with legacy policy")
        return CacheContractContext(root, policy, False, None, None)
    if policy == "auto" and not has_contract and has_payloads:
        _warn_legacy(root)
        return CacheContractContext(root, "legacy", False, None, None)

    if expected_contract is None:
        if not has_contract:
            raise ValueError("strict feature cache requires an expected contract")
        stored = _read_contract(contract_path)
        digest = validate_feature_cache_contract(stored)
        if (
            required_cache_kind is not None
            and stored["identity"]["cache_kind"] != required_cache_kind
        ):
            raise ValueError("feature cache kind does not match the requested reader")
        return CacheContractContext(root, "strict", True, stored, digest)

    expected = dict(expected_contract)
    expected_digest = validate_feature_cache_contract(expected)
    if has_contract:
        stored = _read_contract(contract_path)
        stored_digest = validate_feature_cache_contract(stored)
        if (
            required_cache_kind is not None
            and stored["identity"]["cache_kind"] != required_cache_kind
        ):
            raise ValueError("feature cache kind does not match the requested reader")
        if stored != expected:
            raise ValueError(
                "feature cache contract mismatch; use a new cache root for a different "
                "checkpoint, layer, preprocessing, or batching protocol"
            )
        return CacheContractContext(root, "strict", True, stored, stored_digest)
    if has_payloads:
        raise ValueError(
            "strict policy refuses to bless existing .pt files without a cache contract"
        )
    if not create:
        raise ValueError("feature cache contract is missing; run the producer first")
    root.mkdir(parents=True, exist_ok=True)
    write_json(contract_path, expected)
    return CacheContractContext(root, "strict", True, expected, expected_digest)


def prepare_model_feature_cache(
    root: Path | str,
    *,
    model: Any,
    cache_kind: str,
    frame_batch_size: int,
    video_batch_size: int,
    policy: str = "auto",
    create: bool = False,
) -> CacheContractContext:
    """Lazily build model identity only when a root resolves to strict mode."""
    root = Path(root)
    if not cache_policy_uses_strict_entries(root, policy):
        return prepare_feature_cache(
            root,
            expected_contract=None,
            policy="legacy" if policy == "legacy" else "auto",
            create=create,
            required_cache_kind=cache_kind,
        )
    expected = build_feature_cache_contract(
        model,
        cache_kind=cache_kind,
        frame_batch_size=frame_batch_size,
        video_batch_size=video_batch_size,
    )
    return prepare_feature_cache(
        root,
        expected_contract=expected,
        policy="strict" if policy == "strict" else "auto",
        create=create,
        required_cache_kind=cache_kind,
    )


def entry_metadata_path(cache_path: Path | str) -> Path:
    return Path(f"{Path(cache_path)}{ENTRY_SUFFIX}")


def cache_entry_is_complete(cache_path: Path | str, *, strict: bool) -> bool:
    cache_path = Path(cache_path)
    return cache_path.is_file() and (
        not strict or entry_metadata_path(cache_path).is_file()
    )


def tensor_descriptor(value: Any) -> dict[str, Any]:
    return {
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
    }


def write_cache_entry_metadata(
    context: CacheContractContext,
    *,
    cache_path: Path,
    source_video_path: Path | str,
    frame_indices: Sequence[int],
    payload: Mapping[str, Any],
) -> CacheEntrySummary:
    if not context.strict or context.contract_sha256 is None:
        raise ValueError("entry metadata is only written for strict caches")
    cache_path = cache_path.resolve()
    root = context.root.resolve()
    try:
        relative_cache = cache_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("cache entry is outside its contract root") from exc
    source = Path(source_video_path)
    if not cache_path.is_file() or not source.is_file():
        raise FileNotFoundError("cache entry or source video is missing")
    indices = [int(index) for index in frame_indices]
    if len(indices) != len(set(indices)):
        raise ValueError("strict cache entry frame indices must be distinct")
    source_size, source_mtime_ns, source_sha = _fingerprint_file(source)
    cache_size, _, cache_sha = _fingerprint_file(cache_path)
    metadata = {
        "schema_version": ENTRY_SCHEMA_VERSION,
        "contract_sha256": context.contract_sha256,
        "cache_file": {
            "relative_path": relative_cache,
            "bytes": cache_size,
            "sha256": cache_sha,
        },
        "source_video": {
            "path": str(source.resolve()),
            "index_path": str(source_video_path),
            "bytes": source_size,
            "mtime_ns": source_mtime_ns,
            "sha256": source_sha,
        },
        "frame_indices": indices,
        "frame_indices_sha256": canonical_sha256(indices),
        "payload": dict(payload),
    }
    metadata_path = entry_metadata_path(cache_path)
    write_json(metadata_path, metadata)
    return CacheEntrySummary(
        cache_path=cache_path,
        metadata_path=metadata_path,
        frame_count=len(indices),
        source_sha256=source_sha,
        cache_sha256=cache_sha,
    )


def validate_cache_entry(
    context: CacheContractContext,
    *,
    cache_path: Path,
    source_video_path: Path | str,
    frame_indices: Sequence[int],
    payload: Mapping[str, Any] | None = None,
    verify_cache_sha256: bool = False,
) -> CacheEntrySummary:
    if not context.strict or context.contract_sha256 is None:
        raise ValueError("entry validation is only available for strict caches")
    cache_path = cache_path.resolve()
    metadata_path = entry_metadata_path(cache_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise ValueError("cache entry metadata must be an object")
    _required(
        metadata,
        (
            "schema_version",
            "contract_sha256",
            "cache_file",
            "source_video",
            "frame_indices",
            "frame_indices_sha256",
            "payload",
        ),
        "entry",
    )
    if metadata["schema_version"] != ENTRY_SCHEMA_VERSION:
        raise ValueError("unsupported cache entry metadata schema")
    _sha256(metadata["contract_sha256"], "entry.contract_sha256")
    if metadata["contract_sha256"] != context.contract_sha256:
        raise ValueError("cache entry belongs to a different root contract")
    expected_relative = cache_path.relative_to(context.root.resolve()).as_posix()
    cache_info = metadata["cache_file"]
    source_info = metadata["source_video"]
    if not isinstance(cache_info, Mapping) or not isinstance(source_info, Mapping):
        raise ValueError("cache_file and source_video metadata must be objects")
    _required(cache_info, ("relative_path", "bytes", "sha256"), "entry.cache_file")
    _required(
        source_info,
        ("path", "bytes", "mtime_ns", "sha256"),
        "entry.source_video",
    )
    _string(cache_info["relative_path"], "entry.cache_file.relative_path")
    for field in ("bytes",):
        _nonnegative_integer(cache_info[field], f"entry.cache_file.{field}")
    for field in ("bytes", "mtime_ns"):
        _nonnegative_integer(source_info[field], f"entry.source_video.{field}")
    _string(source_info["path"], "entry.source_video.path")
    if cache_info["relative_path"] != expected_relative:
        raise ValueError("cache entry relative path mismatch")
    cache_stat = cache_path.stat()
    if cache_stat.st_size != cache_info["bytes"]:
        raise ValueError("cache entry byte size changed")
    cache_sha = _sha256(cache_info["sha256"], "entry.cache_file.sha256")
    if verify_cache_sha256 and sha256_file(cache_path) != cache_sha:
        raise ValueError("cache entry SHA-256 mismatch")

    source = Path(source_video_path)
    source_stat = source.stat()
    source_sha = _sha256(source_info["sha256"], "entry.source_video.sha256")
    if (
        source_stat.st_size != source_info["bytes"]
        or source_stat.st_mtime_ns != source_info["mtime_ns"]
    ) and sha256_file(source) != source_sha:
        raise ValueError("source video differs from the cached fingerprint")
    indices = [int(index) for index in frame_indices]
    if not isinstance(metadata["frame_indices"], list) or any(
        isinstance(index, bool) or not isinstance(index, int)
        for index in metadata["frame_indices"]
    ):
        raise ValueError("entry.frame_indices must be an integer list")
    if metadata["frame_indices"] != indices:
        raise ValueError("requested frame indices differ from cache entry metadata")
    _sha256(metadata["frame_indices_sha256"], "entry.frame_indices_sha256")
    if metadata["frame_indices_sha256"] != canonical_sha256(indices):
        raise ValueError("cache entry frame-index digest is invalid")
    if not isinstance(metadata["payload"], Mapping):
        raise ValueError("entry.payload must be an object")
    if payload is not None and metadata["payload"] != dict(payload):
        raise ValueError("loaded tensor payload disagrees with cache entry metadata")
    return CacheEntrySummary(
        cache_path=cache_path,
        metadata_path=metadata_path,
        frame_count=len(indices),
        source_sha256=source_sha,
        cache_sha256=cache_sha,
    )


__all__ = [
    "CACHE_KINDS",
    "CACHE_POLICIES",
    "CONTRACT_FILENAME",
    "CONTRACT_SCHEMA_VERSION",
    "ENTRY_SCHEMA_VERSION",
    "ENTRY_SUFFIX",
    "CacheContractContext",
    "CacheEntrySummary",
    "LegacyFeatureCacheWarning",
    "build_feature_cache_contract",
    "cache_entry_is_complete",
    "cache_policy_uses_strict_entries",
    "canonical_sha256",
    "entry_metadata_path",
    "prepare_feature_cache",
    "prepare_model_feature_cache",
    "tensor_descriptor",
    "validate_cache_entry",
    "validate_feature_cache_contract",
    "write_cache_entry_metadata",
]
