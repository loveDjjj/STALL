#!/usr/bin/env python3
"""运行 Feature-change K3 的严格跨真实域校准与 Universal Real Bank 矩阵。"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from branches.local_branch import local_d2_features
from config import config_digest, load_config
from cross_domain import evaluate_cross_domain_cell
from data.cache_contract import prepare_feature_cache
from data.manifest import load_manifest
from data.packed_cache import PackedCacheReader
from data.video import decode_all_frames, decode_indexed_frames
from features import AlphaStallFeatureExtractor
from math_utils import (
    GaussianMeanCandidateScorerFloat64,
    StableGaussianParams,
    l2_normalized_first_order,
    score_gaussian_aggregate_float64,
)
from pipeline import (
    _fit_parameter,
    _load_cache_payload,
    _load_global_parameters,
    _window_features,
)
from temporal_selection.calibration import load_frozen_local_reference
from temporal_selection.dense_scoring import (
    request_feature_arrays,
    selected_window_requests,
    union_frame_indices,
)
from temporal_selection.manifest import read_window_manifests


@dataclass(frozen=True)
class DomainSpec:
    name: str
    calibration: Path
    evaluation: Path
    window_dir: Path
    reference: Path
    cache_root: Path


DOMAINS = {
    "comgenvid": DomainSpec(
        "comgenvid",
        ROOT / "data/manifests/development/comgenvid_calibration.csv",
        ROOT / "data/manifests/development/comgenvid_evaluation.csv",
        ROOT / "results/caes/window_selection_seed17",
        ROOT / "precomputed/caes_detector/comgenvid_local_d2.npz",
        ROOT / "cache/patch_embeddings_k3_2s_8fps",
    ),
    "videofeedback": DomainSpec(
        "videofeedback",
        ROOT / "data/manifests/development/videofeedback_calibration.csv",
        ROOT / "data/manifests/development/videofeedback_evaluation.csv",
        ROOT / "results/caes/window_selection_seed17",
        ROOT / "precomputed/caes_detector/videofeedback_local_d2.npz",
        ROOT / "cache/patch_embeddings_k3_2s_8fps",
    ),
    "genvideo": DomainSpec(
        "genvideo",
        ROOT / "data/manifests/development/genvideo_calibration.csv",
        ROOT / "data/manifests/development/genvideo_evaluation.csv",
        ROOT / "results/caes/window_selection_seed17",
        ROOT / "precomputed/caes_detector/genvideo_local_d2.npz",
        ROOT / "cache/patch_embeddings_k3_2s_8fps",
    ),
    "genvidbench": DomainSpec(
        "genvidbench",
        ROOT / "data/manifests/external/calibration.csv",
        ROOT / "data/manifests/external/evaluation.csv",
        ROOT / "results/caes/external_genvidbench_feature_change_seed17",
        ROOT / "precomputed/caes_detector_external/genvidbench_local_d2.npz",
        ROOT / "cache/patch_embeddings_k3_2s_8fps",
    ),
}
DEVELOPMENT_DOMAINS = ("comgenvid", "videofeedback", "genvideo")
ALL_DOMAINS = (*DEVELOPMENT_DOMAINS, "genvidbench")
UNIVERSAL_BANKS = {
    "universal3": DEVELOPMENT_DOMAINS,
    "universal4": ALL_DOMAINS,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _decode(path: Path, indices: list[int]):
    try:
        return decode_indexed_frames(path, indices, require_all=True)
    except ValueError as error:
        frames = decode_all_frames(path, require_open=True)
        if not indices or max(indices) >= len(frames):
            raise ValueError(f"跨域评分顺序解码仍缺帧：{path}") from error
        return frames[indices]


def _load_rows_and_manifests(
    spec: DomainSpec, split: str, limit: int | None = None
) -> tuple[pd.DataFrame, dict]:
    path = spec.calibration if split == "calibration" else spec.evaluation
    rows = load_manifest(str(path))
    manifest_path = (
        spec.window_dir / "manifests" / spec.name
        / f"feature_change_{split}.jsonl"
    )
    manifests = {
        item.video_id: item for item in read_window_manifests(manifest_path)
    }
    rows = rows[
        rows["video_path"].map(
            lambda value: f"{spec.name}:{value}" in manifests
        )
    ].reset_index(drop=True)
    if limit is not None:
        rows = (
            rows.groupby(["subset", "source_model"], sort=True, group_keys=False)
            .head(limit).reset_index(drop=True)
        )
        ids = {f"{spec.name}:{value}" for value in rows["video_path"]}
        manifests = {key: value for key, value in manifests.items() if key in ids}
    if len(rows) != len(manifests):
        raise ValueError(f"{spec.name}/{split} rows与Feature-change manifest不一致")
    return rows, manifests


def _domain_uniform_sample(
    spec: DomainSpec,
    config: dict,
    *,
    vectors_per_video: int,
    device: str,
) -> np.ndarray:
    """从每条real均匀抽取相同数量D2位置，避免长视频或域规模支配。"""

    rows, _ = _load_rows_and_manifests(spec, "calibration")
    context = prepare_feature_cache(
        spec.cache_root,
        expected_contract=None,
        policy="strict",
        create=False,
        required_cache_kind="patch_embeddings",
    )
    reader = PackedCacheReader(spec.cache_root)
    samples = []
    for row_index, row in rows.iterrows():
        payload = _load_cache_payload(
            ROOT, spec.cache_root, row, context, reader
        )
        windows = _window_features(payload, row, requested_k=3)
        patch = torch.from_numpy(np.stack([item[1] for item in windows]))
        features = local_d2_features(patch).reshape(-1, 1024).numpy()
        count = min(vectors_per_video, len(features))
        seed = int.from_bytes(
            hashlib.sha256(
                f"17\0{spec.name}\0{row['video_path']}".encode("utf-8")
            ).digest()[:8],
            "little",
        ) % (2**32)
        indices = np.random.default_rng(seed).choice(
            len(features), size=count, replace=False
        )
        samples.append(features[indices].astype(np.float32, copy=False))
        if (row_index + 1) % 25 == 0 or row_index + 1 == len(rows):
            print(
                f"[{spec.name}] Universal采样 {row_index + 1}/{len(rows)}",
                flush=True,
            )
    return np.concatenate(samples)


def _load_or_fit_references(
    config: dict, output_dir: Path, device: str, overwrite: bool
) -> tuple[dict[str, StableGaussianParams], dict[str, str]]:
    references = {
        name: load_frozen_local_reference(spec.reference).params
        for name, spec in DOMAINS.items()
    }
    reference_hashes = {
        name: _sha256(spec.reference) for name, spec in DOMAINS.items()
    }
    universal_dir = output_dir / "universal_references"
    universal_dir.mkdir(parents=True, exist_ok=True)
    required = {
        name: universal_dir / f"{name}_local_d2.npz"
        for name in UNIVERSAL_BANKS
    }
    if overwrite:
        for path in required.values():
            path.unlink(missing_ok=True)
    if all(path.is_file() for path in required.values()):
        for name, path in required.items():
            with np.load(path, allow_pickle=False) as data:
                references[name] = StableGaussianParams(
                    mean=np.asarray(data["mean"], dtype=np.float64),
                    whitening=np.asarray(data["whitening"], dtype=np.float64),
                    calibration_raw=np.array([0.0, 1.0], dtype=np.float64),
                )
            reference_hashes[name] = _sha256(path)
        return references, reference_hashes

    per_domain = {
        name: _domain_uniform_sample(
            DOMAINS[name], config, vectors_per_video=500, device=device
        )
        for name in ALL_DOMAINS
    }
    for name, domains in UNIVERSAL_BANKS.items():
        per_domain_count = 100_000 if len(domains) == 3 else 75_000
        values = np.concatenate([
            per_domain[domain][:per_domain_count] for domain in domains
        ])
        params = _fit_parameter(values, device, "empirical")
        path = required[name]
        np.savez_compressed(
            path,
            mean=params.mean,
            whitening=params.whitening,
            domains=np.asarray(domains),
            vectors_per_domain=np.asarray(per_domain_count),
        )
        references[name] = StableGaussianParams(
            mean=params.mean,
            whitening=params.whitening,
            calibration_raw=np.array([0.0, 1.0], dtype=np.float64),
        )
        reference_hashes[name] = _sha256(path)
        print(
            f"[{name}] Universal参考冻结：{len(values)}个D2位置",
            flush=True,
        )
    return references, reference_hashes


def _score_split(
    spec: DomainSpec,
    split: str,
    rows: pd.DataFrame,
    manifests: dict,
    *,
    references: dict[str, StableGaussianParams],
    global_parameters: dict[str, StableGaussianParams],
    model: AlphaStallFeatureExtractor,
    device: str,
    output_dir: Path,
    identity_sha: str,
    chunk_videos: int,
    decode_workers: int,
) -> pd.DataFrame:
    shard_dir = output_dir / "raw_shards" / spec.name / split
    shard_dir.mkdir(parents=True, exist_ok=True)
    ordered = rows.assign(
        video_id=[f"{spec.name}:{value}" for value in rows["video_path"]]
    ).reset_index(drop=True)
    names = list(references)
    center = np.mean(
        np.stack([references[name].mean for name in names]), axis=0
    )
    local_scorer = GaussianMeanCandidateScorerFloat64(
        [references[name] for name in names], center, device=device
    )
    all_records = []
    started = time.monotonic()
    for start in range(0, len(ordered), chunk_videos):
        stop = min(start + chunk_videos, len(ordered))
        shard_path = shard_dir / f"shard-{start // chunk_videos:05d}.json"
        video_ids = ordered.iloc[start:stop]["video_id"].tolist()
        expected = sum(manifests[video_id].effective_k for video_id in video_ids)
        if shard_path.is_file():
            payload = json.loads(shard_path.read_text(encoding="utf-8"))
            if (
                payload.get("identity_sha256") != identity_sha
                or payload.get("video_ids") != video_ids
                or len(payload.get("records", [])) != expected
            ):
                raise ValueError(f"跨域raw shard身份不匹配：{shard_path}")
            all_records.extend(payload["records"])
            print(f"[{spec.name}/{split}] 恢复 {stop}/{len(ordered)}", flush=True)
            continue

        batch_rows = [row for _, row in ordered.iloc[start:stop].iterrows()]

        def decode(row):
            video_id = str(row["video_id"])
            requests = selected_window_requests(
                {"feature_change": manifests[video_id]}
            )
            union = union_frame_indices(requests)
            path = Path(str(row["video_path"]))
            if not path.is_absolute():
                path = ROOT / path
            return row, video_id, requests, union, _decode(path, union)

        with ThreadPoolExecutor(
            max_workers=min(decode_workers, len(batch_rows))
        ) as executor:
            decoded = list(executor.map(decode, batch_rows))
        outputs = model.frames_to_global_patch_embeddings(
            [item[4] for item in decoded], batch_size=8
        )
        global_windows, patch_windows, metadata, uses = [], [], [], []
        for item, output in zip(decoded, outputs):
            row, video_id, requests, union, _ = item
            current_global, current_patch = request_feature_arrays(
                requests,
                extracted_frame_indices=union,
                global_features=output["global"],
                patch_features=output["patch"],
            )
            global_windows.extend(current_global)
            patch_windows.extend(current_patch)
            metadata.extend([(row, video_id)] * len(requests))
            uses.extend([request.uses[0] for request in requests])

        records = []
        for offset in range(0, len(global_windows), 48):
            stop_window = min(offset + 48, len(global_windows))
            current_global = np.stack(global_windows[offset:stop_window])
            current_patch = np.stack(patch_windows[offset:stop_window])
            spatial_raw, spatial = score_gaussian_aggregate_float64(
                current_global,
                global_parameters["global_spatial"],
                "max",
                device=device,
            )
            temporal_features, zero_mask = l2_normalized_first_order(
                torch.from_numpy(current_global)
            )
            temporal_raw, temporal = score_gaussian_aggregate_float64(
                temporal_features,
                global_parameters["global_t1"],
                "min",
                device=device,
                invalid_mask=zero_mask,
                allow_positive_infinity_percentile=True,
            )
            local = local_scorer.score(
                local_d2_features(torch.from_numpy(current_patch))
            )
            for local_index in range(stop_window - offset):
                index = offset + local_index
                row, video_id = metadata[index]
                use = uses[index]
                record = {
                    "video_id": video_id,
                    "dataset": spec.name,
                    "subset": str(row["subset"]),
                    "source_model": str(row["source_model"]),
                    "video_path": str(row["video_path"]),
                    "window_id": int(use.selection.rank) - 1,
                    "candidate_id": int(use.candidate.candidate_id),
                    "selection_rank": int(use.selection.rank),
                    "global_spatial_raw": float(spatial_raw[local_index]),
                    "global_spatial": float(spatial[local_index]),
                    "global_t1_raw": float(temporal_raw[local_index]),
                    "global_t1": float(temporal[local_index]),
                }
                for reference_index, name in enumerate(names):
                    record[f"patch_temporal_raw__{name}"] = float(
                        local[local_index, reference_index]
                    )
                records.append(record)
        if len(records) != expected:
            raise ValueError(f"{spec.name}/{split}窗口数与manifest不一致")
        payload = {
            "identity_sha256": identity_sha,
            "video_ids": video_ids,
            "records": records,
        }
        temporary = shard_path.with_suffix(".tmp.json")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(shard_path)
        all_records.extend(records)
        rate = stop / max(time.monotonic() - started, 1e-9)
        print(
            f"[{spec.name}/{split}] 评分 {stop}/{len(ordered)}，{rate:.2f}视频/秒",
            flush=True,
        )
    return pd.DataFrame(all_records)


def _audit_diagonal(raw: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    source_runs = {
        "comgenvid": ROOT / "results/runs/caes_stage_fs/window_scores.csv",
        "videofeedback": ROOT / "results/runs/caes_stage_fs/window_scores.csv",
        "genvideo": ROOT / "results/runs/caes_stage_fs/window_scores.csv",
        "genvidbench": ROOT / "results/runs/caes_external_genvidbench_feature_change/window_scores.csv",
    }
    rows = []
    for domain in ALL_DOMAINS:
        expected = pd.read_csv(
            source_runs[domain], float_precision="round_trip", low_memory=False
        )
        expected = expected[
            expected["selector"].eq("feature_change")
            & expected["dataset"].eq(domain)
        ]
        actual = pd.concat([
            raw[(domain, "calibration")], raw[(domain, "evaluation")]
        ])
        merged = actual.merge(
            expected[[
                "video_id", "window_id", "global_spatial_raw", "global_t1_raw",
                "patch_temporal_raw",
            ]],
            on=["video_id", "window_id"],
            suffixes=("_actual", "_expected"),
            validate="one_to_one",
        )
        if len(merged) != len(actual) or len(merged) != len(expected):
            raise ValueError(f"{domain} diagonal审计身份不完整")
        for component, actual_column, expected_column in (
            ("global_spatial", "global_spatial_raw_actual", "global_spatial_raw_expected"),
            ("global_t1", "global_t1_raw_actual", "global_t1_raw_expected"),
            ("local_d2", f"patch_temporal_raw__{domain}", "patch_temporal_raw"),
        ):
            actual_values = merged[actual_column].to_numpy(dtype=np.float64)
            expected_values = merged[expected_column].to_numpy(dtype=np.float64)
            matching_positive_infinity = np.isposinf(actual_values) & np.isposinf(
                expected_values
            )
            finite = np.isfinite(actual_values) & np.isfinite(expected_values)
            valid = finite | matching_positive_infinity
            if not valid.all():
                raise ValueError(
                    f"{domain}/{component} diagonal存在有限值与无穷值身份漂移"
                )
            difference = np.zeros(len(merged), dtype=np.float64)
            difference[finite] = np.abs(
                actual_values[finite] - expected_values[finite]
            )
            rows.append({
                "domain": domain,
                "component": component,
                "windows": len(merged),
                "max_abs_difference": float(difference.max()),
                "mean_abs_difference": float(difference.mean()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/runs/cross_domain_feature_k3_equal_fusion",
    )
    parser.add_argument("--chunk-videos", type=int, default=64)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume与--overwrite不能同时使用")
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"跨域输出已存在：{args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(ROOT / "configs/benchmark.yaml")
    config["method"]["fusion"]["global_weight"] = 0.5
    config["method"]["fusion"]["local_weight"] = 0.5
    references, reference_hashes = _load_or_fit_references(
        config, args.output_dir, args.device, args.overwrite
    )
    window_hashes = {
        f"{domain}/{split}": _sha256(
            DOMAINS[domain].window_dir / "manifests" / domain
            / f"feature_change_{split}.jsonl"
        )
        for domain in ALL_DOMAINS
        for split in ("calibration", "evaluation")
    }
    identity = {
        "schema_version": "cross_domain_feature_k3_equal_fusion_v1",
        "config_hash": config_digest(config),
        "domains": list(ALL_DOMAINS),
        "banks": list(references),
        "reference_sha256": reference_hashes,
        "window_manifest_sha256": window_hashes,
        "fusion": {"global": 0.5, "local": 0.5},
        "limit": args.limit,
        "implementation_sha256": _sha256(Path(__file__)),
    }
    identity_sha = _canonical_digest(identity)
    (args.output_dir / "run_identity.json").write_text(
        json.dumps({**identity, "identity_sha256": identity_sha}, indent=2) + "\n",
        encoding="utf-8",
    )

    global_parameters = _load_global_parameters(ROOT, config)
    model = AlphaStallFeatureExtractor(args.device)
    raw: dict[tuple[str, str], pd.DataFrame] = {}
    for domain in ALL_DOMAINS:
        spec = DOMAINS[domain]
        for split in ("calibration", "evaluation"):
            rows, manifests = _load_rows_and_manifests(spec, split, args.limit)
            raw[(domain, split)] = _score_split(
                spec,
                split,
                rows,
                manifests,
                references=references,
                global_parameters=global_parameters,
                model=model,
                device=args.device,
                output_dir=args.output_dir,
                identity_sha=identity_sha,
                chunk_videos=args.chunk_videos,
                decode_workers=args.decode_workers,
            )

    audit = _audit_diagonal(raw) if args.limit is None else pd.DataFrame()
    summaries, operating, generators, video_outputs = [], [], [], []
    for bank in references:
        source_domains = (
            UNIVERSAL_BANKS[bank] if bank in UNIVERSAL_BANKS else (bank,)
        )
        calibration = pd.concat(
            [raw[(domain, "calibration")] for domain in source_domains],
            ignore_index=True,
        )
        calibration["patch_temporal_raw"] = calibration[
            f"patch_temporal_raw__{bank}"
        ]
        for target in ALL_DOMAINS:
            evaluation = raw[(target, "evaluation")].copy()
            evaluation["patch_temporal_raw"] = evaluation[
                f"patch_temporal_raw__{bank}"
            ]
            summary, points, per_generator = evaluate_cross_domain_cell(
                calibration,
                evaluation,
                config,
                calibration_bank=bank,
                evaluation_domain=target,
            )
            summary["calibration_source_domains"] = "+".join(source_domains)
            points["calibration_source_domains"] = "+".join(source_domains)
            per_generator["calibration_source_domains"] = "+".join(source_domains)
            summaries.append(summary)
            operating.append(points)
            generators.append(per_generator)
    outputs = {
        "matrix_metrics.csv": pd.concat(summaries, ignore_index=True),
        "operating_points.csv": pd.concat(operating, ignore_index=True),
        "generator_metrics.csv": pd.concat(generators, ignore_index=True),
        "diagonal_raw_audit.csv": audit,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False)
    manifest = {
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "identity_sha256": identity_sha,
        "artifacts": {
            name: _sha256(args.output_dir / name) for name in outputs
        },
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[完成] 跨真实域校准矩阵：{args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
