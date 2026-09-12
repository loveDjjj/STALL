"""真实拟合资产到Gaussian、独立视频CDF及可导出参考；不引用历史脚本。"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from artifacts import checkpoint_read, checkpoint_write, paper_json
from math_utils import StableGaussianParams, l2_normalized_first_order
from reference import (
    SCHEMA,
    ReferenceBundle,
    file_digest,
    fit_gaussian,
    local_video_cdfs,
    load_bundle,
    publish_arrays,
    save_bundle,
)
from selection import uniform_windows, validate_indices


FIT_CONTRACT = dict(
    input_size=224,
    batch_size=8,
    pad_tail=True,
    dimensions=1024,
    fit_selector="uniform",
    fit_k=3,
    local_vectors_per_clip=256,
    weighting="equal total weight per clip",
    covariance_ddof=1,
    ridge=1e-5,
    difference_dtype="float32",
    scoring_dtype="float64",
    global_cdf="uniform first window",
    local_cdf="ranked effective-K linspace",
)


def sample_d2_positions(values, sampling_identity):
    """兼容原17:path种子；不能用归档后的物理路径重新定义256位置。"""
    if len(values) < 256:
        raise ValueError("Local拟合向量不足256位置")
    seed = int.from_bytes(
        hashlib.sha256(("17:" + sampling_identity).encode()).digest()[:8], "little"
    )
    indices = np.random.default_rng(seed).choice(len(values), size=256, replace=False)
    return values[indices]


def prepare_fit_assets(
    root,
    config,
    frame,
    states,
    directory,
    identity,
    progress=None,
    *,
    row_indices=None,
    write_manifest=True,
    require_cached=False,
):
    """仅保存Uniform Global及抽样D2；大Patch特征在每视频消费完成后释放。"""
    from features import AlphaStallFeatureExtractor
    from data.video import decode_bounded
    from data.prefetch import bounded_map, frame_reservation
    from workflow import video_file_states
    from execution import row_numbers

    root = Path(root)
    directory = Path(directory)
    rows = frame.to_dict("records")
    pending = []
    for index, row in zip(row_numbers(frame, row_indices), rows):
        path = directory / "fit_features" / f"{index:06d}.npz"
        marker = directory / "fit_features" / f"{index:06d}.json"
        if marker.exists():
            saved = checkpoint_read(marker, identity, row["video_id"])
            if file_digest(path) != saved["sha256"]:
                raise ValueError("拟合特征checkpoint损坏")
            row.update(feature_asset=str(path), feature_asset_sha256=saved["sha256"])
        else:
            if require_cached:
                raise ValueError("worker返回后仍有缺失拟合特征")
            windows = uniform_windows(json.loads(row["downsample_idxs"]), 3)
            union = sorted({i for window in windows for i in window.frame_indices})
            pending.append((index, row, windows, union, path, marker))
    if pending:
        if file_digest(root / config["encoder"]["weights"]) != config["encoder"]["weights_sha256"]:
            raise ValueError("DINO权重不匹配")
        model = AlphaStallFeatureExtractor(
            config["runtime"]["device"],
            dino_repo=str(root / config["encoder"]["repo"]),
            dino_weights=str(root / config["encoder"]["weights"]),
            pad_tail_batch=True,
        )

        def prepare(item):
            _, row, _, union, _, _ = item
            return model.prepare_frames(decode_bounded(root / row["video_path"], union))

        runtime = config["runtime"]
        stream = bounded_map(
            pending,
            prepare,
            lambda item: frame_reservation(root / item[1]["video_path"], len(item[3])),
            workers=runtime.get("decode_workers", 4),
            depth=runtime.get("prefetch_depth", 4),
            budget=runtime.get("prefetch_memory_mb", 4096) * 2**20,
        )
        try:
            for item, frames in stream:
                index, row, windows, union, path, marker = item
                feature = model.frames_to_global_patch_embeddings([frames], batch_size=8)[0]
                where = {value: position for position, value in enumerate(union)}
                picks = [[where[i] for i in window.frame_indices] for window in windows]
                patches = torch.from_numpy(
                    np.stack([feature["patch"][positions] for positions in picks])
                )
                raw = patches[:, 2:] - 2.0 * patches[:, 1:-1] + patches[:, :-2]
                sampling_identity = row.get("legacy_path") or row["video_path"]
                vectors = sample_d2_positions(raw.reshape(-1, 1024), sampling_identity).numpy()
                stat = (root / row["video_path"]).stat()
                expected = states[row["video_id"]]
                if stat.st_size != expected["bytes"] or stat.st_mtime_ns != expected["mtime_ns"]:
                    raise ValueError("拟合原视频在提取期间改变")
                publish_arrays(
                    path,
                    dict(
                        identity=np.asarray(identity),
                        video_id=np.asarray(row["video_id"]),
                        sampling_identity=np.asarray(sampling_identity),
                        raw_d2=vectors,
                        global_windows=np.stack(
                            [feature["global"][positions] for positions in picks]
                        ),
                        frame_indices=np.asarray([window.frame_indices for window in windows]),
                    ),
                )
                digest = file_digest(path)
                checkpoint_write(marker, identity, dict(video_id=row["video_id"], sha256=digest))
                row.update(feature_asset=str(path), feature_asset_sha256=digest)
                if progress:
                    progress("features", index, len(frame), row["video_id"])
                del feature, patches, raw, vectors, frames
        finally:
            stream.close()
        del model
        if config["runtime"]["device"].startswith("cuda"):
            torch.cuda.empty_cache()
    if video_file_states(root, frame) != {vid: states[vid] for vid in frame.video_id}:
        raise ValueError("拟合视频集合状态改变")
    result = pd.DataFrame(rows)
    # 当前run生成的清单独立于源清单；路径变化不回写源video_id或抽样身份。
    if write_manifest:
        path = directory / "prepared_fit.csv"
        temporary = directory / "prepared_fit.csv.tmp"
        result.to_csv(temporary, index=False)
        temporary.replace(path)
    return result


def _read_role(path, split, *, real_only=True):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {
        "video_id",
        "video_path",
        "source_group",
        "dataset",
        "subset",
        "split",
        "downsample_idxs",
    }
    if required - set(frame) or frame.empty:
        raise ValueError(f"{split}清单缺少身份字段或为空")
    if not frame["split"].eq(split).all():
        raise ValueError(f"清单角色不是{split}")
    if real_only and not frame.subset.eq("real").all():
        raise ValueError("Gaussian/CDF/threshold只能读取真实视频")
    if frame.video_id.duplicated().any() or frame.video_path.duplicated().any():
        raise ValueError("真实参考身份重复")
    if frame[["video_id", "video_path", "source_group", "dataset"]].eq("").any().any():
        raise ValueError("身份或源组为空")
    for indices in frame.downsample_idxs:
        if len(validate_indices(json.loads(indices))) < 16:
            raise ValueError("参考视频不足16互异帧")
    return frame


def prepare_fit_inputs(root, config, dataset, fit_path, cdf_path, evaluation_path, threshold_path):
    """运行前审计四种数据角色；评测标签只用于排除泄漏，不进入统计拟合。"""
    from workflow import video_file_states

    root = Path(root)
    if config["selection"]["k"] != 3:
        raise ValueError("当前fit构建K1/2/3参考，requested-K须为3")
    paths = dict(fit=fit_path, cdf=cdf_path, evaluation=evaluation_path, threshold=threshold_path)
    frames = {
        role: _read_role(path, role, real_only=role != "evaluation") for role, path in paths.items()
    }
    fit, cdf = frames["fit"], frames["cdf"]
    if not fit.dataset.eq(dataset).all() or not frames["evaluation"].dataset.eq(dataset).all():
        raise ValueError("目标拟合或评测清单的数据域不符")
    if len(fit) < 2 or len(cdf) < 2:
        raise ValueError("真实拟合/CDF视频不足")
    # 源组是已审计的逻辑分组；内容hash可用时增加跨角色检查，不冒充语义去重。
    for i, role in enumerate(paths):
        for other in list(paths)[i + 1 :]:
            for column in ("video_id", "source_group", "content_sha256"):
                if column not in frames[role] or column not in frames[other]:
                    continue
                left = set(frames[role][column]) - {""}
                right = set(frames[other][column]) - {""}
                if left & right:
                    raise ValueError(f"{role}/{other}存在相同{column}")
            left = {str((root / p).resolve()) for p in frames[role].video_path}
            right = {str((root / p).resolve()) for p in frames[other].video_path}
            if left & right:
                raise ValueError(f"{role}/{other}指向相同物理视频")
    source = config.get("fit", {}).get("feature_source", "cache")
    assets = {}
    fit_states = {}
    if source == "cache":
        if {"feature_asset", "feature_asset_sha256"} - set(fit):
            raise ValueError("cache拟合需要有内容hash的轻量特征资产")
        for row in fit.itertuples():
            path = root / row.feature_asset
            digest = file_digest(path)
            if digest != row.feature_asset_sha256:
                raise ValueError("拟合特征资产hash不符")
            assets[row.video_id] = dict(path=str(path.resolve()), sha256=digest)
    elif source == "video":
        fit_states = video_file_states(root, fit)
    else:
        raise ValueError("未知拟合特征源")
    inputs = dict(
        dataset=dataset,
        contract=FIT_CONTRACT,
        manifests={
            role: dict(
                path=str(Path(path).resolve()), sha256=file_digest(path), rows=len(frames[role])
            )
            for role, path in paths.items()
        },
        fit_assets=assets,
        fit_video_files=fit_states,
        feature_source=source,
        cdf_video_files=video_file_states(root, cdf),
        independence="source_group / known content SHA256 / resolved path; not semantic deduplication",
    )
    return fit, cdf, inputs


def fit_from_assets(root, frame, *, progress=None):
    """复用原索引及256位置；均值/协方差不接触fake或CDF视频。"""
    values = {name: [] for name in ("gs", "gt", "lt")}
    for index, row in enumerate(frame.itertuples(), 1):
        path = Path(root) / row.feature_asset
        if file_digest(path) != row.feature_asset_sha256:
            raise ValueError("读取时拟合特征资产改变")
        with np.load(path, allow_pickle=False) as z:
            if str(z["video_id"]) != row.video_id:
                raise ValueError("拟合特征视频身份错误")
            g = z["global_windows"]
            raw = z["raw_d2"]
            indices = z["frame_indices"]
            expected = np.asarray(
                [w.frame_indices for w in uniform_windows(json.loads(row.downsample_idxs), 3)]
            )
            if not np.array_equal(indices, expected):
                raise ValueError("拟合特征不是对应Uniform K3索引")
            if (
                g.dtype != np.float32
                or g.shape != (len(expected), 16, 1024)
                or raw.dtype != np.float32
                or raw.shape != (256, 1024)
            ):
                raise ValueError("拟合特征形状/dtype不匹配固定合同")
            if not np.isfinite(g).all() or not np.isfinite(raw).all():
                raise ValueError("拟合特征非有限")
            gt, zero = l2_normalized_first_order(torch.from_numpy(g))
            values["gs"].append(g.reshape(-1, 1024))
            values["gt"].append(gt[~zero].reshape(-1, 1024).numpy())
            values["lt"].append(
                torch.nn.functional.normalize(torch.from_numpy(raw), dim=-1, eps=1e-12).numpy()
            )
        if progress:
            progress("assets", index, len(frame), row.video_id)
    params = {}
    statistics = {}
    for name, observations in values.items():
        params[name] = fit_gaussian(observations, ridge=FIT_CONTRACT["ridge"])
        statistics[name] = dict(
            clips=len(frame),
            usable_clips=sum(bool(len(x)) for x in observations),
            observations=sum(len(x) for x in observations),
            dimensions=1024,
        )
        if progress:
            progress("gaussian", len(params), 3, name)
    return params, statistics


def _save_gaussians(path, params, metadata):
    arrays = {"metadata": np.asarray(json.dumps(metadata, sort_keys=True, ensure_ascii=False))}
    for name, p in params.items():
        arrays.update({name + "_mean": p.mean, name + "_whitening": p.whitening})
    publish_arrays(path, arrays)


def _load_gaussians(path, identity):
    with np.load(path, allow_pickle=False) as z:
        metadata = json.loads(str(z["metadata"]))
        if metadata.get("identity") != identity:
            raise ValueError("Gaussian恢复身份不符")
        params = {
            name: StableGaussianParams(
                z[name + "_mean"].copy(), z[name + "_whitening"].copy(), np.empty(0)
            )
            for name in ("gs", "gt", "lt")
        }
    for p in params.values():
        if (
            p.mean.shape != (1024,)
            or p.whitening.shape != (1024, 1024)
            or not np.isfinite(p.mean).all()
            or not np.isfinite(p.whitening).all()
        ):
            raise ValueError("Gaussian资产损坏")
    return params, metadata


def run_reference_fit(
    root, config, dataset, fit, cdf, inputs, directory, identity, *, stop_after="cdf", progress=None
):
    """先发布可恢复Gaussian，再逐视频重算CDF；只在后者完成后发布参考包。"""
    from execution import devices_for, build_jobs, run_workers, gpu_job
    import copy

    config = copy.deepcopy(config)
    config["runtime"]["device"] = devices_for(config)[0]
    distributed = len(devices_for(config)) > 1
    root = Path(root)
    directory = Path(directory)
    if inputs.get("feature_source", "cache") == "video":
        if distributed:
            jobs = build_jobs(
                fit,
                config,
                dict(
                    kind="fit_features",
                    root=root,
                    directory=directory,
                    identity=identity,
                    states=inputs["fit_video_files"],
                ),
            )
            run_workers(gpu_job, jobs, directory, "fit_features", progress)
        fit = prepare_fit_assets(
            root,
            config,
            fit,
            inputs["fit_video_files"],
            directory,
            identity,
            progress,
            require_cached=distributed,
        )
    gaussian_path = directory / "gaussians.npz"
    state_path = directory / "gaussian_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["identity"] != identity or state["sha256"] != file_digest(gaussian_path):
            raise ValueError("Gaussian checkpoint已改变")
        params, metadata = _load_gaussians(gaussian_path, identity)
    else:
        params, statistics = fit_from_assets(root, fit, progress=progress)
        metadata = dict(
            identity=identity,
            dataset=dataset,
            contract=FIT_CONTRACT,
            statistics=statistics,
            source_groups=int(fit.source_group.nunique()),
            fit_manifest=inputs["manifests"]["fit"],
        )
        _save_gaussians(gaussian_path, params, metadata)
        paper_json(state_path, dict(identity=identity, sha256=file_digest(gaussian_path)))
    if stop_after == "gaussian":
        return None
    if stop_after != "cdf":
        raise ValueError("未知拟合停止阶段")
    if distributed:
        jobs = build_jobs(
            cdf,
            config,
            dict(
                kind="cdf",
                root=root,
                directory=directory,
                identity=identity,
                states=inputs["cdf_video_files"],
            ),
        )
        run_workers(gpu_job, jobs, directory, "cdf", progress)
    records = score_cdf_checkpointed(
        root,
        config,
        cdf,
        params,
        inputs["cdf_video_files"],
        directory,
        identity,
        progress=None if distributed else progress,
        require_cached=distributed,
    )
    if file_digest(gaussian_path) != json.loads(state_path.read_text())["sha256"]:
        raise ValueError("CDF评分期间Gaussian文件改变")
    gs = np.asarray([float(r["uniform"]["windows"][0]["global_spatial_raw"]) for r in records])
    gt = np.asarray([float(r["uniform"]["windows"][0]["global_temporal_raw"]) for r in records])
    lt = local_video_cdfs(
        [[float(w["local_raw"]) for w in r["selected"]["windows"]] for r in records]
    )
    bundle = ReferenceBundle(
        StableGaussianParams(params["gs"].mean, params["gs"].whitening, gs),
        StableGaussianParams(params["gt"].mean, params["gt"].whitening, gt),
        params["lt"],
        lt,
        dict(
            schema=SCHEMA,
            dataset=dataset,
            protocol_id=config["protocol"]["id"],
            selector=config["selection"]["name"],
            requested_k=3,
            fit_contract=FIT_CONTRACT,
            fit_identity=identity,
            gaussian_sha256=file_digest(gaussian_path),
            manifests=inputs["manifests"],
            statistics=metadata["statistics"],
            source_groups=metadata["source_groups"],
            independent_threshold="not scored; no operating threshold exported",
        ),
    )
    save_bundle(directory / "references" / f"{dataset}.npz", bundle)
    paper_json(
        directory / "references/manifest.json",
        dict(
            status="completed",
            protocol_id=config["protocol"]["id"],
            references=[
                dict(
                    dataset=dataset, sha256=file_digest(directory / "references" / f"{dataset}.npz")
                )
            ],
        ),
    )
    return bundle


def score_cdf_checkpointed(
    root,
    config,
    cdf,
    params,
    states,
    directory,
    identity,
    progress=None,
    *,
    row_indices=None,
    require_cached=False,
):
    from workflow import RawVideoScorer, video_file_states, prefetched_raw_scores
    from execution import row_numbers

    root = Path(root)
    directory = Path(directory)
    scorer = None
    records = []
    items = list(zip(row_numbers(cdf, row_indices), cdf.itertuples()))

    def make_scorer():
        if file_digest(root / config["encoder"]["weights"]) != config["encoder"]["weights_sha256"]:
            raise ValueError("DINO权重不匹配")
        return RawVideoScorer(
            params["gs"],
            params["gt"],
            params["lt"],
            config["runtime"]["device"],
            dino_repo=str(root / config["encoder"]["repo"]),
            dino_weights=str(root / config["encoder"]["weights"]),
        )

    def publish(index, row, record):
        stat = (root / row.video_path).stat()
        expected = states[row.video_id]
        if stat.st_size != expected["bytes"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError("CDF视频评分期间改变")
        checkpoint_write(directory / "cdf_raw" / f"{index:06d}.json", identity, record)

    prefetch = config["runtime"].get("prefetch_enabled", False)
    if prefetch and not require_cached:
        pending = []
        for item in items:
            path = directory / "cdf_raw" / f"{item[0]:06d}.json"
            if path.exists():
                checkpoint_read(path, identity, item[1].video_id)
            else:
                pending.append(item)
        if pending:
            scorer = make_scorer()
            order = {row.video_id: i + 1 for i, (_, row) in enumerate(items)}
            stream = prefetched_raw_scores(root, config, scorer, pending, include_uniform=True)
            try:
                for (index, row), result in stream:
                    publish(index, row, dict(video_id=row.video_id, **result))
                    if progress:
                        progress("cdf", order[row.video_id], len(cdf), row.video_id)
            finally:
                stream.close()
        require_cached = True
    for completed, (index, row) in enumerate(items, 1):
        path = directory / "cdf_raw" / f"{index:06d}.json"
        if path.exists():
            record = checkpoint_read(path, identity, row.video_id)
        else:
            if require_cached:
                raise ValueError("worker返回后仍有缺失CDF检查点")
            if scorer is None:
                scorer = make_scorer()
            indices = validate_indices(json.loads(row.downsample_idxs))
            # 保留两次前向的原批次上下文，不能为了复用帧而悄悄改变尾批/并列分数。
            uniform = scorer.score_global_first_window(root / row.video_path, indices)
            selected = scorer.score_raw(
                root / row.video_path, indices, selector=config["selection"]["name"], k=3
            )
            record = dict(video_id=row.video_id, uniform=uniform, selected=selected)
            publish(index, row, record)
        records.append(record)
        if progress and not prefetch:
            progress("cdf", completed, len(cdf), row.video_id)
    if progress and prefetch and items:
        progress("cdf", len(items), len(cdf), items[-1][1].video_id)
    if video_file_states(root, cdf) != {vid: states[vid] for vid in cdf.video_id}:
        raise ValueError("CDF输入文件集合改变")
    return records


def verified_fit_bundle(directory):
    """只有完整fit产物可导出；只拟合Gaussian、半个CDF或被修改的结果均拒绝。"""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "run_manifest.json").read_text())
    if "fit" not in manifest.get("completed_steps", []):
        raise ValueError("fit未完成CDF，不能导出")
    state = json.loads((directory / "artifact_manifest.json").read_text())
    if state.get("status") != "completed":
        raise ValueError("拟合产物未完成")
    for name, digest in state["artifacts"].items():
        if file_digest(directory / name) != digest:
            raise ValueError("拟合产物hash改变，拒绝导出")
    registry = json.loads((directory / "references/manifest.json").read_text())
    if registry["status"] != "completed" or len(registry["references"]) != 1:
        raise ValueError("拟合参考清单错误")
    row = registry["references"][0]
    path = directory / "references" / f"{row['dataset']}.npz"
    if file_digest(path) != row["sha256"]:
        raise ValueError("拟合参考包hash改变")
    bundle = load_bundle(path)
    if bundle.metadata["dataset"] != row["dataset"]:
        raise ValueError("拟合参考包域错误")
    return path, dict(
        source_run=str(directory),
        source_manifest_sha256=file_digest(directory / "run_manifest.json"),
        source_bundle_sha256=file_digest(path),
        dataset=row["dataset"],
        protocol_id=bundle.metadata["protocol_id"],
    )


def export_fit_bundle(source, provenance, directory):
    import shutil

    directory = Path(directory) / "references"
    directory.mkdir()
    target = directory / f"{provenance['dataset']}.npz"
    shutil.copy2(source, target)
    if (
        file_digest(source) != provenance["source_bundle_sha256"]
        or file_digest(target) != provenance["source_bundle_sha256"]
    ):
        raise ValueError("导出期间源参考改变")
    paper_json(
        directory / "manifest.json",
        dict(
            status="completed",
            protocol_id=provenance["protocol_id"],
            provenance=provenance,
            references=[
                dict(dataset=provenance["dataset"], sha256=provenance["source_bundle_sha256"])
            ],
        ),
    )
