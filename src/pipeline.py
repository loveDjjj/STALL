"""Alpha STALL 从严格特征缓存到可发布结果的完整执行链路。"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import traceback
import copy
import gc
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Callable

import numpy as np
import pandas as pd
import torch
import yaml

from branches.global_branch import (
    GLOBAL_SPATIAL_AGGREGATION,
    GLOBAL_TEMPORAL_AGGREGATION,
    load_official_stall_parameters,
)
from branches.local_branch import LOCAL_AGGREGATION
from data.cache_contract import prepare_feature_cache, tensor_descriptor, validate_cache_entry
from data.manifest import load_manifest
from data.patch_cache import _get_patch_cache_path, cache_frame_indices
from data.packed_cache import PackedCacheReader
from data.sampling import parse_indices, uniform_windows
from dynamics.local import LocalDynamicsResult, build_local_dynamics
from likelihood.conditional import (
    ConditionalGaussianParams,
    assign_condition_bins,
    score_conditional_gaussian_mean_float64,
)
from math_utils import (
    StableGaussianParams,
    WhiteningTransform,
    l2_normalized_first_order,
    score_gaussian_aggregate_float64,
    stable_sorted,
)


COMPONENTS = ("global_spatial", "global_t1", "patch_spatial", "patch_temporal")
GLOBAL_WINDOW_COLUMNS = (
    "global_spatial_raw", "global_spatial", "global_t1_raw", "global_t1"
)


def _write_worker_event(path: Path, payload: dict) -> None:
    """顺序追加单个 worker 的诊断事件，避免多进程争写同一文件。"""

    record = {"timestamp_monotonic": monotonic(), "pid": os.getpid(), **payload}
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class DatasetManifests:
    dataset: str
    calibration: Path
    evaluation: Path


def resolve_dataset_manifests(repository_root: Path, config: dict, dataset: str) -> DatasetManifests:
    """解析开发集或外部集的 calibration/evaluation 清单。"""

    if dataset == "genvidbench":
        root = repository_root / config["data"]["external_manifests"]
        return DatasetManifests(dataset, root / "calibration.csv", root / "evaluation.csv")
    root = repository_root / config["data"]["development_manifests"]
    return DatasetManifests(
        dataset,
        root / f"{dataset}_calibration.csv",
        root / f"{dataset}_evaluation.csv",
    )


def _video_id(dataset: str, video_path: str) -> str:
    return f"{dataset}:{video_path}"


def _seed(seed: int, dataset: str) -> int:
    digest = hashlib.sha256(f"{seed}:{dataset}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _choose_calibration(frame: pd.DataFrame, dataset: str, count: int, seed: int) -> pd.DataFrame:
    real = frame[frame["subset"].eq("real")].copy()
    if len(real) < count:
        raise ValueError(f"{dataset} 仅有 {len(real)} 个真实校准视频，少于配置要求的 {count} 个")
    if len(real) == count:
        return real.reset_index(drop=True)
    return real.sample(n=count, random_state=_seed(seed, dataset)).sort_values("video_path").reset_index(drop=True)


def _ensure_disjoint(calibration: pd.DataFrame, evaluation: pd.DataFrame, dataset: str) -> None:
    overlap = set(calibration["video_path"]).intersection(evaluation["video_path"])
    if overlap:
        raise ValueError(f"{dataset} calibration 与 evaluation 视频重叠，示例：{sorted(overlap)[:3]}")


def _apply_short_video_policy(frame: pd.DataFrame, dataset: str, split: str, policy: str) -> tuple[pd.DataFrame, int]:
    """确保当前 2 秒方法不会隐式漏掉不足 16 帧的视频。"""

    usable = frame["downsample_idxs"].map(lambda value: len(parse_indices(value)) >= 16)
    unavailable = int((~usable).sum())
    if unavailable and policy == "error":
        examples = frame.loc[~usable, "video_path"].head(3).tolist()
        raise ValueError(
            f"{dataset} {split} 有 {unavailable} 条视频不足 2 秒（16 帧）；"
            f"当前 short_video_policy=error，示例：{examples}"
        )
    return frame[usable].reset_index(drop=True), unavailable


def _cache_window_count(context) -> int | None:
    """从根级 contract 读取多窗口缓存协议。"""

    contract = context.contract or {}
    selection = contract.get("identity", {}).get("extraction", {}).get("frame_selection")
    if not isinstance(selection, dict):
        return None
    if selection.get("mode") != "uniform_window_union":
        raise ValueError("不支持的严格缓存帧选择协议")
    count = selection.get("window_count")
    if not isinstance(count, int) or count < 1:
        raise ValueError("严格缓存缺少有效的 window_count")
    if selection.get("window_frames") != 16 or selection.get("strategy") != "uniform":
        raise ValueError("严格缓存窗口协议与当前 Alpha STALL 采样不兼容")
    return count


def _load_cache_payload(
    repository_root: Path,
    cache_root: Path,
    row: pd.Series,
    context,
    reader: PackedCacheReader | None = None,
    use_locked_override: bool = False,
) -> dict:
    """读取并逐条验证完整或多窗口严格缓存。"""

    stem = Path(str(row["video_path"])).stem
    cache_path = _get_patch_cache_path(
        cache_root, str(row["subset"]), str(row["source_model"]), stem, 2, False
    )
    cache_key = cache_path.relative_to(cache_root).as_posix()
    # 锁定 U0 的帧索引仅有一条视频超出当前 K=3 缓存并集。覆盖特征只可用于
    # 锁定帧协议；refit 实验必须读取通用缓存，不能混入该条专用帧集合。
    locked_override = repository_root / "precomputed" / "locked_u0" / "cache_overrides" / cache_key
    if use_locked_override and locked_override.is_file():
        return torch.load(locked_override, weights_only=True)
    packed = reader.get(cache_key) if reader is not None else None
    if packed is not None:
        payload = packed["payload"]
    else:
        if not cache_path.is_file():
            raise FileNotFoundError(f"严格缓存缺失：{cache_path}")
        payload = torch.load(cache_path, weights_only=True)
    indices = cache_frame_indices(
        row,
        duration_sec=2,
        compact=False,
        cache_window_count=_cache_window_count(context),
    )
    descriptor = {"format": "torch_dict_global_patch_v1", "global": tensor_descriptor(payload["global"]), "patch": tensor_descriptor(payload["patch"]), "grid_size": [int(value) for value in payload["grid_size"]]}
    if packed is None:
        validate_cache_entry(context, cache_path=cache_path, source_video_path=str(repository_root / row["video_path"]), frame_indices=indices, payload=descriptor)
    else:
        metadata = packed["entry_metadata"]
        if metadata["frame_indices"] != indices or metadata["payload"] != descriptor:
            raise ValueError("packed cache 条目与当前 manifest 或 payload 不一致")
    return payload


def _cache_key(cache_root: Path, row: pd.Series) -> str:
    """计算严格缓存键；集中定义以保证 shard 调度与常规读取完全一致。"""

    stem = Path(str(row["video_path"])).stem
    return _get_patch_cache_path(
        cache_root, str(row["subset"]), str(row["source_model"]), stem, 2, False
    ).relative_to(cache_root).as_posix()


def _window_features(
    payload: dict,
    row: pd.Series,
    requested_k: int,
    locked_windows: list[list[int]] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """按当前协议或锁定 U0 索引选择每个视频的窗口特征。"""

    downsample = parse_indices(row["downsample_idxs"])
    windows = locked_windows if locked_windows is not None else uniform_windows(
        downsample, requested_k, window_frames=16
    )
    position = {frame: index for index, frame in enumerate(payload["frame_indices"])}
    output: list[tuple[np.ndarray, np.ndarray]] = []
    for window in windows:
        try:
            selected = [position[frame] for frame in window]
        except KeyError as error:
            raise ValueError(f"缓存帧索引与 manifest 不一致：{row['video_path']}") from error
        output.append((payload["global"].numpy()[selected], payload["patch"].numpy()[selected]))
    return output


def _temporal_global(values: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    difference = tensor[1:] - tensor[:-1]
    return torch.nn.functional.normalize(difference, p=2, dim=-1, eps=1e-12).numpy()


def _patch_grid_size(values: np.ndarray | torch.Tensor) -> tuple[int, int]:
    """当前缓存使用方形 patch 网格；拒绝静默猜测非方形布局。"""

    patch_count = int(values.shape[-2])
    side = int(round(np.sqrt(patch_count)))
    if side * side != patch_count:
        raise ValueError(f"无法从 {patch_count} 个 patch 推断方形网格")
    return side, side


def _temporal_patch(
    values: np.ndarray | torch.Tensor, local_config: dict, device: str
) -> LocalDynamicsResult:
    """构造 Local evidence；真实拟合与测试评分共用同一实现。"""

    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return build_local_dynamics(
        tensor,
        grid_size=_patch_grid_size(tensor),
        local_config=local_config,
        device=device,
    )


def _reservoir_add(reservoir: np.ndarray | None, values: np.ndarray, limit: int, seen: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1, values.shape[-1])
    if reservoir is None:
        reservoir = np.empty((limit, flat.shape[1]), dtype=np.float32)
    for value in flat:
        seen += 1
        if seen <= limit:
            reservoir[seen - 1] = value
        else:
            target = int(rng.integers(0, seen))
            if target < limit:
                reservoir[target] = value
    return reservoir, seen


def _fit_parameter(
    features: np.ndarray, device: str, covariance_estimator: str
) -> StableGaussianParams:
    transform = WhiteningTransform(
        features, device=device, covariance_estimator=covariance_estimator
    )
    return StableGaussianParams(
        mean=transform.mean_.detach().cpu().numpy().astype(np.float64),
        whitening=transform.whitening_matrix_.detach().cpu().numpy().astype(np.float64),
        calibration_raw=np.array([0.0, 1.0], dtype=np.float64),
        shrinkage=float(transform.shrinkage_),
    )


def _fit_local_parameters(calibration_windows: list[tuple[np.ndarray, np.ndarray]], config: dict, device: str) -> dict[str, StableGaussianParams]:
    """仅用目标域互斥真实视频拟合 Alpha STALL 的 Local 参数。"""

    method = config["method"]
    covariance_estimator = str(method["local"]["covariance_estimator"])
    limit = int(config["runtime"]["max_features_for_fit"])
    if limit < 2:
        raise ValueError("runtime.max_features_for_fit 必须至少为 2")
    needed: set[str] = set()
    if method["local"]["enabled"]:
        if method["local"]["spatial_enabled"]:
            needed.add("patch_spatial")
        if (
            method["local"]["temporal_enabled"]
            and not method["local"].get("conditional", {}).get("enabled", False)
        ):
            needed.add("patch_temporal")
    reservoirs: dict[str, np.ndarray | None] = {key: None for key in needed}
    seen = {key: 0 for key in needed}
    rng = np.random.default_rng(int(config["calibration"]["seed"]))
    for _global_values, patch_values in calibration_windows:
        items: dict[str, np.ndarray] = {}
        if "patch_spatial" in needed:
            items["patch_spatial"] = patch_values
        if "patch_temporal" in needed:
            temporal = _temporal_patch(
                patch_values, method["local"], device
            )
            items["patch_temporal"] = temporal.features.squeeze(0).numpy()
        for name, values in items.items():
            reservoirs[name], seen[name] = _reservoir_add(reservoirs[name], values, limit, seen[name], rng)
    return {
        name: _fit_parameter(
            values[: min(limit, seen[name])], device, covariance_estimator
        )
        for name, values in reservoirs.items()
        if values is not None and seen[name] >= 2
    }


def _fit_conditional_local_parameter(
    calibration_windows: list[tuple[np.ndarray, np.ndarray]],
    config: dict,
    device: str,
) -> ConditionalGaussianParams:
    """仅用 calibration real 的 speed quantile 拟合三个条件 Gaussian。"""

    local = config["method"]["local"]
    conditional = local.get("conditional", {})
    if not conditional.get("enabled", False):
        raise ValueError("条件 Local 拟合要求 conditional.enabled=true")
    bin_count = int(conditional["bins"])
    states = []
    for _, patch_values in calibration_windows:
        result = _temporal_patch(patch_values, local, device)
        if result.conditioning_state is None:
            raise ValueError("当前 Local dynamics 未返回条件运动状态")
        states.append(result.conditioning_state.numpy().reshape(-1))
    all_states = np.concatenate(states).astype(np.float64, copy=False)
    boundaries = np.quantile(
        all_states,
        np.arange(1, bin_count, dtype=np.float64) / float(bin_count),
        method="linear",
    )

    total_limit = int(config["runtime"]["max_features_for_fit"])
    per_bin_limit = max(2, total_limit // bin_count)
    reservoirs: list[np.ndarray | None] = [None] * bin_count
    seen = [0] * bin_count
    rng = np.random.default_rng(int(config["calibration"]["seed"]))
    for _, patch_values in calibration_windows:
        result = _temporal_patch(patch_values, local, device)
        features = result.features.numpy().reshape(-1, result.features.shape[-1])
        assignments = assign_condition_bins(
            result.conditioning_state, boundaries
        ).reshape(-1)
        for bin_index in range(bin_count):
            selected = features[assignments == bin_index]
            if not len(selected):
                continue
            reservoirs[bin_index], seen[bin_index] = _reservoir_add(
                reservoirs[bin_index], selected, per_bin_limit, seen[bin_index], rng
            )
    if any(item is None or count < 2 for item, count in zip(reservoirs, seen)):
        raise ValueError(f"条件运动 bin 样本不足：{seen}")
    estimator = str(local["covariance_estimator"])
    target = torch.device(device)
    gc.collect()
    if target.type == "cuda":
        torch.cuda.empty_cache()
    fitted_bins = []
    for reservoir, count in zip(reservoirs, seen):
        if reservoir is None:
            raise ValueError("条件运动 bin 缺少 reservoir")
        fitted_bins.append(
            _fit_parameter(
                reservoir[: min(per_bin_limit, count)], device, estimator
            )
        )
        # `_fit_parameter` 已把 mean/W 转为 CPU NumPy；立即回收该 bin 的
        # covariance/eigen 临时张量，避免三次 1024-D eigh 的缓存累积到 OOM。
        gc.collect()
        if target.type == "cuda":
            torch.cuda.empty_cache()
    return ConditionalGaussianParams(
        boundaries=np.asarray(boundaries, dtype=np.float64),
        bins=tuple(fitted_bins),
        state_name=str(conditional["state"]),
    )


def _load_locked_local_parameters(
    repository_root: Path, config: dict, dataset: str
) -> dict[str, StableGaussianParams]:
    """加载锁定 U0 的 Local 参数及其独立 K1 窗口 CDF 原始参考。"""

    directory = repository_root / str(config["method"]["local"]["locked_parameters_dir"])
    path = directory / f"{dataset}_region1_mean.npz"
    required = {
        "mu_patch_spat", "W_patch_spat", "calib_patch_spat_scores",
        "mu_patch_temp", "W_patch_temp", "calib_patch_temp_scores",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"锁定 Local 参数缺少字段：{sorted(missing)}")
        return {
            "patch_spatial": StableGaussianParams(
                mean=np.asarray(data["mu_patch_spat"], dtype=np.float64),
                whitening=np.asarray(data["W_patch_spat"], dtype=np.float64),
                calibration_raw=stable_sorted(np.asarray(data["calib_patch_spat_scores"], dtype=np.float64)),
            ),
            "patch_temporal": StableGaussianParams(
                mean=np.asarray(data["mu_patch_temp"], dtype=np.float64),
                whitening=np.asarray(data["W_patch_temp"], dtype=np.float64),
                calibration_raw=stable_sorted(np.asarray(data["calib_patch_temp_scores"], dtype=np.float64)),
            ),
        }


def _locked_video_id(dataset: str, row: pd.Series) -> str:
    """计算锁定帧索引使用的稳定视频身份。"""

    identity = "|".join((
        dataset, str(row["subset"]), str(row["source_model"]),
        Path(str(row["video_path"])).name,
    ))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _locked_windows(
    repository_root: Path, config: dict
) -> dict[str, list[list[int]]] | None:
    """默认 K=3 主实验读取不可变 U0 选帧索引。"""

    local = config["method"]["local"]
    if local.get("parameter_source") != "locked_u0" or int(config["sampling"]["num_windows"]) != 3:
        return None
    path = repository_root / str(local["locked_frame_indices"])
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    videos = payload.get("videos")
    if not isinstance(videos, dict):
        raise ValueError("锁定帧索引缺少 videos 映射")
    return {
        str(video_id): [[int(frame) for frame in window] for window in windows]
        for video_id, windows in videos.items()
    }


def _load_global_parameters(repository_root: Path, config: dict) -> dict[str, StableGaussianParams]:
    """加载并校验严格继承原始 STALL 的独立 VATEX Global 参数。"""

    global_config = config["method"]["global"]
    if not global_config["enabled"]:
        return {}
    parameter_path = repository_root / str(global_config["parameters"])
    if not parameter_path.is_file():
        raise FileNotFoundError(f"官方 STALL VATEX 参数文件不存在：{parameter_path}")
    digest = hashlib.sha256(parameter_path.read_bytes()).hexdigest()
    if digest != str(global_config["parameters_sha256"]):
        raise ValueError(
            "官方 STALL VATEX 参数 SHA256 不匹配："
            f"期望 {global_config['parameters_sha256']}，实际 {digest}"
        )
    return load_official_stall_parameters(parameter_path)


def _load_reused_global_windows(
    repository_root: Path, config: dict, cache_contract_sha256: str | None
) -> tuple[pd.DataFrame | None, dict[str, object] | None]:
    """读取并验证一个已完成 run 的逐窗口 Global 分数。"""

    run_name = config["runtime"].get("reuse_global_run")
    if run_name is None:
        return None, None
    directory = repository_root / "results" / "runs" / str(run_name)
    required_files = (
        directory / "progress.json",
        directory / "run_manifest.json",
        directory / "resolved_config.yaml",
        directory / "window_scores.csv",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"复用 Global 的来源 run 缺少文件：{missing}")
    progress = json.loads(required_files[0].read_text(encoding="utf-8"))
    manifest = json.loads(required_files[1].read_text(encoding="utf-8"))
    if progress.get("status") != "completed" or manifest.get("status") != "completed":
        raise ValueError(f"复用 Global 的来源 run 尚未完成：{run_name}")
    source_config = yaml.safe_load(required_files[2].read_text(encoding="utf-8"))
    checks = {
        "method.global": (source_config["method"]["global"], config["method"]["global"]),
        "sampling": (source_config["sampling"], config["sampling"]),
        "calibration": (source_config["calibration"], config["calibration"]),
        "data": (source_config["data"], config["data"]),
        "runtime.cache_dir": (
            source_config["runtime"]["cache_dir"], config["runtime"]["cache_dir"]
        ),
    }
    mismatched = [name for name, (left, right) in checks.items() if left != right]
    if mismatched:
        raise ValueError(f"复用 Global 的来源配置不兼容：{mismatched}")
    source_contract = manifest.get("pipeline", {}).get("cache_contract_sha256")
    if source_contract != cache_contract_sha256:
        raise ValueError("复用 Global 的来源 cache contract 与当前严格缓存不一致")

    windows = pd.read_csv(required_files[3], float_precision="round_trip")
    keys = ["dataset", "split", "video_id", "window_id"]
    required_columns = set(keys).union(GLOBAL_WINDOW_COLUMNS)
    missing_columns = required_columns.difference(windows.columns)
    if missing_columns:
        raise ValueError(f"复用 Global 的窗口表缺少字段：{sorted(missing_columns)}")
    if windows.duplicated(keys).any():
        raise ValueError("复用 Global 的窗口表含重复窗口身份")
    selected = windows[keys + list(GLOBAL_WINDOW_COLUMNS)].copy()
    identity = {
        "run_name": str(run_name),
        "git_commit": manifest.get("git_commit"),
        "config_hash": manifest.get("config_hash"),
        "window_scores_sha256": hashlib.sha256(required_files[3].read_bytes()).hexdigest(),
    }
    return selected, identity


def _attach_reused_global(
    frame: pd.DataFrame,
    source: pd.DataFrame,
    *,
    dataset: str,
    split: str,
) -> pd.DataFrame:
    """按唯一窗口身份一一附加 Global 分数，拒绝缺失或多余窗口。"""

    keys = ["dataset", "video_id", "window_id"]
    reference = source[
        source["dataset"].eq(dataset) & source["split"].eq(split)
    ].drop(columns="split")
    merged = frame.merge(reference, on=keys, how="left", validate="one_to_one", sort=False)
    if len(merged) != len(frame) or merged[list(GLOBAL_WINDOW_COLUMNS)].isna().any().any():
        raise ValueError(f"{dataset} {split} 无法与复用 Global 窗口一一对齐")
    # merge 保留主表原始顺序，但显式恢复内部 input order 以防 pandas 版本差异。
    if "_input_order" in merged:
        merged = merged.sort_values(["_input_order", "window_id"], kind="mergesort")
    return merged.reset_index(drop=True)


def _save_fitted_local_artifacts(
    output_dir: Path,
    dataset: str,
    selected_calibration: pd.DataFrame,
    calibration_windows: pd.DataFrame,
    parameters: dict[str, StableGaussianParams],
    config: dict,
) -> str | None:
    """保存 real-only Local 参数、窗口参考与明确的校准视频身份。"""

    local_names = [name for name in ("patch_spatial", "patch_temporal") if name in parameters]
    if not local_names:
        return None
    directory = output_dir / "fitted_local"
    directory.mkdir(parents=True, exist_ok=True)
    identifiers = selected_calibration.copy()
    identifiers.insert(0, "dataset", dataset)
    identifiers["video_id"] = identifiers["video_path"].map(
        lambda path: _video_id(dataset, str(path))
    )
    identifier_columns = [
        column for column in ("dataset", "video_id", "subset", "source_model", "video_path")
        if column in identifiers
    ]
    identifiers[identifier_columns].to_csv(
        directory / f"{dataset}_calibration_ids.csv", index=False
    )

    payload: dict[str, np.ndarray] = {}
    for name in local_names:
        payload[f"{name}_mean"] = np.asarray(parameters[name].mean, dtype=np.float64)
        payload[f"{name}_whitening"] = np.asarray(
            parameters[name].whitening, dtype=np.float64
        )
        if parameters[name].shrinkage is not None:
            payload[f"{name}_shrinkage"] = np.asarray(
                [parameters[name].shrinkage], dtype=np.float64
            )
        raw_name = f"{name}_raw"
        if raw_name in calibration_windows:
            payload[f"{name}_window_reference"] = stable_sorted(
                calibration_windows[raw_name].to_numpy(dtype=np.float64)
            )
    target = directory / f"{dataset}_local_params.npz"
    temporary = target.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    metadata = {
        "dataset": dataset,
        "sha256": digest,
        "calibration_videos": len(selected_calibration),
        "local_config": config["method"]["local"],
        "arrays": {name: list(value.shape) for name, value in payload.items()},
    }
    (directory / f"{dataset}_local_params.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return digest


def _save_fitted_conditional_artifacts(
    output_dir: Path,
    dataset: str,
    selected_calibration: pd.DataFrame,
    calibration_windows: pd.DataFrame,
    parameters: ConditionalGaussianParams,
    config: dict,
) -> str:
    """保存条件边界、逐 bin Gaussian、窗口 CDF 参考和校准视频身份。"""

    directory = output_dir / "fitted_local"
    directory.mkdir(parents=True, exist_ok=True)
    identifiers = selected_calibration.copy()
    identifiers.insert(0, "dataset", dataset)
    identifiers["video_id"] = identifiers["video_path"].map(
        lambda path: _video_id(dataset, str(path))
    )
    columns = [
        name for name in ("dataset", "video_id", "subset", "source_model", "video_path")
        if name in identifiers
    ]
    identifiers[columns].to_csv(
        directory / f"{dataset}_calibration_ids.csv", index=False
    )
    payload: dict[str, np.ndarray] = {
        "boundaries": np.asarray(parameters.boundaries, dtype=np.float64),
        "patch_temporal_window_reference": stable_sorted(
            calibration_windows["patch_temporal_raw"].to_numpy(dtype=np.float64)
        ),
    }
    for index, gaussian in enumerate(parameters.bins):
        payload[f"bin_{index}_mean"] = np.asarray(gaussian.mean, dtype=np.float64)
        payload[f"bin_{index}_whitening"] = np.asarray(
            gaussian.whitening, dtype=np.float64
        )
    target = directory / f"{dataset}_conditional_local_params.npz"
    temporary = target.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    metadata = {
        "dataset": dataset,
        "sha256": digest,
        "state_name": parameters.state_name,
        "bin_count": len(parameters.bins),
        "calibration_videos": len(selected_calibration),
        "local_config": config["method"]["local"],
        "arrays": {name: list(value.shape) for name, value in payload.items()},
    }
    (directory / f"{dataset}_conditional_local_params.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return digest


def _score_windows(
    repository_root: Path,
    rows: pd.DataFrame,
    cache_root: Path,
    context,
    dataset: str,
    config: dict,
    parameters: dict[str, StableGaussianParams],
    device: str,
    *,
    score_global: bool = True,
    local_candidates: dict[str, tuple[dict, StableGaussianParams]] | None = None,
    conditional_candidates: dict[
        str, tuple[dict, ConditionalGaussianParams]
    ] | None = None,
    report: Callable[[int, int], None] | None = None,
    diagnostic: Callable[[dict], None] | None = None,
) -> pd.DataFrame:
    """预取严格缓存后逐窗口评分；窗口的浮点计算顺序保持不变。"""

    method = config["method"]
    requested_k = int(config["sampling"]["num_windows"])
    locked_window_map = _locked_windows(repository_root, config)
    records: list[dict] = []
    batch_size = int(config["runtime"].get("score_batch_size", 1))
    io_workers = int(config["runtime"].get("cache_io_workers", 1))
    indexed_rows = list(rows.iterrows())
    packed_reader = PackedCacheReader(cache_root)

    def load_batch(batch: list[tuple[int, pd.Series]]) -> list[tuple[int, pd.Series, dict]]:
        def load(item: tuple[int, pd.Series]) -> tuple[int, pd.Series, dict]:
            position, row = item
            return position, row, _load_cache_payload(
                repository_root, cache_root, row, context, packed_reader,
                use_locked_override=locked_window_map is not None,
            )
        with ThreadPoolExecutor(max_workers=min(io_workers, len(batch))) as executor:
            return list(executor.map(load, batch))

    batches = [indexed_rows[offset: offset + batch_size] for offset in range(0, len(indexed_rows), batch_size)]
    completed = 0
    # 下一批读取和当前批评分并行，减少 GPU 等待缓存 I/O 的空档。
    with ThreadPoolExecutor(max_workers=1) as prefetcher:
        future = prefetcher.submit(load_batch, batches[0]) if batches else None
        for batch_index in range(len(batches)):
            loaded = future.result()
            if batch_index == 0 and diagnostic is not None:
                diagnostic({"event": "first_batch_loaded", "videos": len(loaded)})
            future = prefetcher.submit(load_batch, batches[batch_index + 1]) if batch_index + 1 < len(batches) else None
            batch_windows: list[tuple[dict, np.ndarray, np.ndarray]] = []
            for position, row, payload in loaded:
                locked = (
                    locked_window_map.get(_locked_video_id(dataset, row))
                    if locked_window_map is not None else None
                )
                if locked_window_map is not None and locked is None:
                    raise ValueError(f"锁定帧索引缺少视频：{row['video_path']}")
                for window_id, (global_values, patch_values) in enumerate(
                    _window_features(payload, row, requested_k, locked)
                ):
                    batch_windows.append(({
                        "video_id": _video_id(dataset, str(row["video_path"])),
                        "dataset": dataset,
                        "subset": str(row["subset"]),
                        "source_model": str(row["source_model"]),
                        "video_path": str(row["video_path"]),
                        "window_id": window_id,
                        "_input_order": position,
                    }, global_values, patch_values))
            global_values = np.stack([item[1] for item in batch_windows])
            patch_values = np.stack([item[2] for item in batch_windows])
            scores: dict[str, np.ndarray] = {}
            if method["global"]["enabled"] and score_global:
                scores["global_spatial_raw"], scores["global_spatial"] = score_gaussian_aggregate_float64(
                    global_values, parameters["global_spatial"], GLOBAL_SPATIAL_AGGREGATION, device=device
                )
                global_temporal, global_zero_mask = l2_normalized_first_order(
                    torch.from_numpy(global_values)
                )
                scores["global_t1_raw"], scores["global_t1"] = score_gaussian_aggregate_float64(
                    global_temporal, parameters["global_t1"], GLOBAL_TEMPORAL_AGGREGATION,
                    device=device, invalid_mask=global_zero_mask,
                    # 原始 STALL: 整窗零差分的 min 聚合结果为 +inf，CDF 应为 1。
                    allow_positive_infinity_percentile=True,
                )
            patch_tensor = torch.from_numpy(patch_values)
            locked_local = method["local"].get("parameter_source") == "locked_u0"
            if method["local"]["enabled"] and method["local"]["spatial_enabled"]:
                scores["patch_spatial_raw"], scores["patch_spatial"] = score_gaussian_aggregate_float64(
                    patch_values, parameters["patch_spatial"], LOCAL_AGGREGATION,
                    device=device, compute_percentile=locked_local,
                )
            if local_candidates is not None:
                for candidate_name, (candidate_local, candidate_params) in local_candidates.items():
                    temporal = _temporal_patch(
                        patch_tensor, candidate_local, device
                    )
                    raw, _ = score_gaussian_aggregate_float64(
                        temporal.features,
                        candidate_params,
                        LOCAL_AGGREGATION,
                        device=device,
                        compute_percentile=False,
                        position_weights=temporal.aggregation_weights,
                    )
                    scores[f"patch_temporal_raw__{candidate_name}"] = raw
            if conditional_candidates is not None:
                for candidate_name, (candidate_local, candidate_params) in conditional_candidates.items():
                    temporal = _temporal_patch(
                        patch_tensor, candidate_local, device
                    )
                    if temporal.conditioning_state is None:
                        raise ValueError("条件候选缺少 current_speed 状态")
                    scores[f"patch_temporal_raw__{candidate_name}"] = (
                        score_conditional_gaussian_mean_float64(
                            temporal.features,
                            temporal.conditioning_state,
                            candidate_params,
                            device=device,
                        )
                    )
            if (
                local_candidates is None
                and conditional_candidates is None
                and method["local"]["enabled"]
                and method["local"]["temporal_enabled"]
            ):
                temporal = _temporal_patch(
                    patch_tensor, method["local"], device
                )
                if method["local"].get("conditional", {}).get("enabled", False):
                    if temporal.conditioning_state is None:
                        raise ValueError("条件 Local 缺少 current_speed 状态")
                    scores["patch_temporal_raw"] = score_conditional_gaussian_mean_float64(
                        temporal.features,
                        temporal.conditioning_state,
                        parameters["patch_temporal_conditional"],
                        device=device,
                    )
                    scores["patch_temporal"] = np.full(
                        len(scores["patch_temporal_raw"]), np.nan, dtype=np.float64
                    )
                else:
                    scores["patch_temporal_raw"], scores["patch_temporal"] = score_gaussian_aggregate_float64(
                        temporal.features, parameters["patch_temporal"], LOCAL_AGGREGATION,
                        device=device, compute_percentile=locked_local,
                        position_weights=temporal.aggregation_weights,
                    )
            for index, (record, _, _) in enumerate(batch_windows):
                record.update({name: float(values[index]) for name, values in scores.items()})
                records.append(record)
            completed += len(loaded)
            if batch_index == 0 and diagnostic is not None:
                diagnostic({"event": "first_batch_scored", "videos": completed, "windows": len(batch_windows)})
            if report is not None:
                report(completed, len(indexed_rows))
    if not records:
        raise ValueError(f"{dataset} 没有满足 16 帧窗口条件的视频")
    return pd.DataFrame(records)


def _split_rows_by_packed_shard(
    rows: pd.DataFrame, cache_root: Path, workers: int
) -> list[pd.DataFrame] | None:
    """将 evaluation 分为互不重叠的连续 shard 区间。

    评分子进程各自持有 LRU。若同一 shard 被两个子进程访问，机械盘仍会发生交错读取；
    因此按 index 中的物理 shard 名排序，并且只在 shard 边界切分任务。原始 DataFrame
    行号被保留，最终输出仍按原 manifest 顺序写出。
    """

    reader = PackedCacheReader(cache_root)
    groups: dict[str, list[int]] = {}
    for position, (_, row) in enumerate(rows.iterrows()):
        location = reader.location(_cache_key(cache_root, row))
        if location is None:
            return None
        groups.setdefault(location[0], []).append(position)
    ordered_groups = [(name, groups[name]) for name in sorted(groups)]
    target = max(1, int(np.ceil(len(rows) / workers)))
    chunks: list[list[int]] = [[]]
    for _, positions in ordered_groups:
        if (
            chunks[-1]
            and len(chunks) < workers
            and len(chunks[-1]) + len(positions) > target
        ):
            chunks.append([])
        chunks[-1].extend(positions)
    return [rows.iloc[positions].copy() for positions in chunks if positions]


def _score_rows_worker(
    repository_root: str,
    cache_root: str,
    context,
    dataset: str,
    config: dict,
    parameters: dict[str, StableGaussianParams],
    device: str,
    rows: pd.DataFrame,
    score_global: bool = True,
    progress_queue=None,
    worker_log_path: str | None = None,
) -> pd.DataFrame:
    """独立 CUDA 进程的评测分片入口；必须保持模块级以支持 spawn。"""

    log_path = Path(worker_log_path) if worker_log_path is not None else None

    def diagnostic(payload: dict) -> None:
        if log_path is not None:
            _write_worker_event(log_path, {"device": device, "dataset": dataset, **payload})

    reader = PackedCacheReader(Path(cache_root))
    first_shard = reader.location(_cache_key(Path(cache_root), rows.iloc[0])) if not rows.empty else None
    diagnostic({"event": "worker_started", "videos": len(rows), "first_shard": first_shard[0] if first_shard else None})

    # 每个 worker 最多发约 100 次进度。batch 大小未必整除该步长，必须按“跨过步长”而非
    # “恰好整除”判断，否则运行正常时 progress.json 也可能长期停在 0%。
    report_stride = max(1, len(rows) // 100)
    last_reported = 0

    def worker_report(done: int, total: int) -> None:
        nonlocal last_reported
        if progress_queue is not None and (done == total or done - last_reported >= report_stride):
            progress_queue.put((device, done, total))
            last_reported = done

    try:
        result = _score_windows(
            Path(repository_root), rows, Path(cache_root), context, dataset, config,
            parameters, device, report=worker_report if progress_queue is not None else None,
            diagnostic=diagnostic, score_global=score_global,
        )
    except BaseException as error:
        diagnostic({"event": "worker_failed", "exception_type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        raise
    diagnostic({"event": "worker_completed", "videos": len(rows)})
    return result


def _score_evaluation(
    repository_root: Path,
    rows: pd.DataFrame,
    cache_root: Path,
    context,
    dataset: str,
    config: dict,
    parameters: dict[str, StableGaussianParams],
    devices: list[str],
    report: Callable[[int, int], None],
    worker_log_dir: Path | None = None,
    score_global: bool = True,
) -> pd.DataFrame:
    """单卡使用预取流水线；双卡按互斥连续 shard 区间评分并合并。"""

    if len(devices) == 1 or len(rows) < 2:
        return _score_windows(
            repository_root, rows, cache_root, context, dataset, config,
            parameters, devices[0], report=report, score_global=score_global,
        )
    chunks = _split_rows_by_packed_shard(rows, cache_root, len(devices))
    if chunks is None:
        chunks = [chunk.copy() for chunk in np.array_split(rows, len(devices)) if not chunk.empty]
    report(0, len(rows))
    # Manager 队列可安全传给 spawn 子进程；主进程据此持续写 progress.json。
    with multiprocessing.Manager() as manager, ProcessPoolExecutor(
        max_workers=len(chunks), mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        progress_queue = manager.Queue()
        futures = {
            executor.submit(
                _score_rows_worker, str(repository_root), str(cache_root), context,
                dataset, config, parameters, device, chunk, score_global, progress_queue,
                str(worker_log_dir / f"worker_{dataset}_{device.replace(':', '_')}.jsonl") if worker_log_dir else None,
            ): (device, len(chunk))
            for device, chunk in zip(devices, chunks)
        }
        local_progress = {device: 0 for device, _ in futures.values()}
        outputs = []
        pending = set(futures)
        while pending:
            try:
                device, done, _ = progress_queue.get(timeout=0.5)
                local_progress[device] = max(local_progress[device], done)
                report(sum(local_progress.values()), len(rows))
            except Exception as error:
                # Queue 超时是正常轮询；子进程异常由 future.result() 原样抛出。
                if error.__class__.__name__ != "Empty":
                    raise
            completed, pending = wait(pending, timeout=0, return_when=FIRST_COMPLETED)
            for future in completed:
                device, size = futures[future]
                outputs.append(future.result())
                local_progress[device] = size
                report(sum(local_progress.values()), len(rows))
    return pd.concat(outputs, ignore_index=True).sort_values(["_input_order", "window_id"]).reset_index(drop=True)


def _calibrate_component(target: pd.Series, reference: pd.Series) -> np.ndarray:
    return np.searchsorted(stable_sorted(reference.to_numpy()), target.to_numpy(), side="right") / float(len(reference))


def _weighted(frame: pd.DataFrame, names: list[str], weights: list[float]) -> np.ndarray:
    active = [(name, weight) for name, weight in zip(names, weights) if name in frame]
    if not active:
        raise ValueError("配置关闭了所有可融合分量")
    total = sum(weight for _, weight in active)
    if total <= 0:
        raise ValueError("启用分量的融合权重必须为正")
    return sum(frame[name].to_numpy() * (weight / total) for name, weight in active)


def _calibrate_and_aggregate(calibration: pd.DataFrame, evaluation: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """只以 calibration real 建立窗口与视频级 CDF，绝不读取 evaluation real 作为参考。"""

    method = config["method"]
    calibration = calibration.copy()
    evaluation = evaluation.copy()
    active_raw = []
    if method["local"]["enabled"] and method["local"]["spatial_enabled"]:
        active_raw.append("patch_spatial_raw")
    if method["local"]["enabled"] and method["local"]["temporal_enabled"]:
        active_raw.append("patch_temporal_raw")
    reference = calibration[calibration["subset"].eq("real")]
    if reference.empty:
        raise ValueError("校准窗口中没有真实视频")
    if method["local"].get("parameter_source") == "fit_real_only":
        for raw in active_raw:
            calibrated = raw.removesuffix("_raw")
            calibration[calibrated] = _calibrate_component(calibration[raw], reference[raw])
            evaluation[calibrated] = _calibrate_component(evaluation[raw], reference[raw])
    for frame in (calibration, evaluation):
        if method["global"]["enabled"]:
            frame["global_score_window"] = _weighted(frame, ["global_spatial", "global_t1"], [float(method["global"]["spatial_weight"]), float(method["global"]["temporal_weight"])])
        if method["local"]["enabled"]:
            frame["local_score_window"] = _weighted(frame, ["patch_spatial", "patch_temporal"], [float(method["local"]["spatial_weight"]), float(method["local"]["temporal_weight"])])

    def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
        columns = ["video_id", "dataset", "subset", "source_model", "video_path"]
        aggregate_columns = {"effective_k": ("window_id", "size")}
        if method["global"]["enabled"]:
            aggregate_columns["global_raw"] = ("global_score_window", "mean")
        if method["local"]["enabled"]:
            aggregate_columns["local_raw"] = ("local_score_window", "mean")
        return frame.groupby(columns, as_index=False).agg(**aggregate_columns)

    def calibration_reference_for_k(frame: pd.DataFrame, target_k: int) -> pd.DataFrame:
        """从每个 calibration 视频的均匀子窗口构造给定 effective-K 的参考。"""

        columns = ["video_id", "dataset", "subset", "source_model", "video_path"]
        records: list[dict] = []
        for keys, video_windows in frame.groupby(columns, sort=False):
            ordered = video_windows.sort_values("window_id").reset_index(drop=True)
            if len(ordered) < target_k:
                continue
            positions = np.unique(np.rint(np.linspace(0, len(ordered) - 1, target_k)).astype(int))
            if len(positions) != target_k:
                continue
            selected = ordered.iloc[positions]
            record = dict(zip(columns, keys))
            record["effective_k"] = target_k
            if method["global"]["enabled"]:
                record["global_raw"] = float(selected["global_score_window"].mean())
            if method["local"]["enabled"]:
                record["local_raw"] = float(selected["local_score_window"].mean())
            records.append(record)
        return pd.DataFrame(records)

    evaluation_video = aggregate(evaluation)
    outputs = []
    for (dataset, effective_k), target in evaluation_video.groupby(["dataset", "effective_k"], sort=False):
        reference_video = calibration_reference_for_k(
            calibration[calibration["dataset"].eq(dataset)], int(effective_k)
        )
        if len(reference_video) < 2:
            raise ValueError(f"{dataset} 的 calibration real 在 effective-K={effective_k} 下不足两个视频")
        output = target.copy()
        if method["global"]["enabled"]:
            output["global_score"] = _calibrate_component(output["global_raw"], reference_video["global_raw"])
        if method["local"]["enabled"]:
            output["local_score"] = _calibrate_component(output["local_raw"], reference_video["local_raw"])
        output["final_score"] = _weighted(output, ["global_score", "local_score"], [float(method["fusion"]["global_weight"]), float(method["fusion"]["local_weight"])])
        outputs.append(output)
    return calibration, pd.concat(outputs, ignore_index=True)


def run_from_cache(
    repository_root: Path,
    config: dict,
    *,
    report: Callable[[dict], None] | None = None,
    worker_log_dir: Path | None = None,
    artifact_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """执行配置指定的全部数据集，返回逐窗口、逐视频分数和运行元信息。"""

    runtime = config["runtime"]
    cache_root = repository_root / runtime["cache_dir"]
    context = prepare_feature_cache(cache_root, expected_contract=None, policy="strict", create=False, required_cache_kind="patch_embeddings")
    reused_global, reused_global_identity = _load_reused_global_windows(
        repository_root, config, context.contract_sha256
    )
    cached_window_count = _cache_window_count(context)
    if cached_window_count is not None and int(config["sampling"]["num_windows"]) > cached_window_count:
        raise ValueError(
            f"请求 K={config['sampling']['num_windows']}，但严格缓存只覆盖 K<={cached_window_count}；"
            "请使用兼容配置或重建缓存。"
        )
    devices = [str(item) for item in runtime.get("devices", [])] or [str(runtime["device"])]
    if any(device.startswith("cuda") for device in devices) and not torch.cuda.is_available():
        raise RuntimeError("配置请求 CUDA，但 PyTorch 未检测到 CUDA")
    for device in devices:
        if device.startswith("cuda"):
            index = torch.device(device).index or 0
            free_bytes, _ = torch.cuda.mem_get_info(index)
            required_gib = float(runtime["minimum_free_gib"])
            free_gib = free_bytes / 1024**3
            if free_gib < required_gib:
                raise RuntimeError(
                    f"GPU {index} 仅空闲 {free_gib:.1f} GiB，低于主实验安全阈值 "
                    f"{required_gib:.1f} GiB；拒绝开始评分以避免争抢显存。"
                )
    all_windows, all_videos = [], []
    packed_reader = PackedCacheReader(cache_root)
    metadata: dict[str, object] = {
        "cache_root": str(cache_root),
        "cache_contract_sha256": context.contract_sha256,
        "reused_global": reused_global_identity,
        "datasets": {},
    }
    for dataset in config["data"]["datasets"]:
        dataset = str(dataset)
        primary_device = torch.device(devices[0])
        if primary_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(primary_device)
        manifests = resolve_dataset_manifests(repository_root, config, str(dataset))
        calibration = load_manifest(str(manifests.calibration))
        evaluation = load_manifest(str(manifests.evaluation))
        _ensure_disjoint(calibration, evaluation, str(dataset))
        policy = str(config["data"]["short_video_policy"])
        calibration, excluded_calibration = _apply_short_video_policy(calibration, str(dataset), "calibration", policy)
        evaluation, excluded_evaluation = _apply_short_video_policy(evaluation, str(dataset), "evaluation", policy)
        selected_calibration = _choose_calibration(calibration, str(dataset), int(config["calibration"]["real_videos_per_dataset"]), int(config["calibration"]["seed"]))
        if report:
            report({"current_dataset": dataset, "phase": "calibration_load", "completed": 0, "total": len(selected_calibration), "message": f"[{dataset}] 读取 {len(selected_calibration)} 条真实校准视频"})
        local_config = config["method"]["local"]
        calibration_windows_features = []
        if local_config.get("enabled") and local_config.get("parameter_source") == "fit_real_only":
            for completed, (_, row) in enumerate(selected_calibration.iterrows(), start=1):
                payload = _load_cache_payload(repository_root, cache_root, row, context, packed_reader)
                calibration_windows_features.extend(_window_features(payload, row, int(config["sampling"]["num_windows"])))
                if report and (completed == len(selected_calibration) or completed % max(1, len(selected_calibration) // 20) == 0):
                    report({"current_dataset": dataset, "phase": "calibration_load", "completed": completed, "total": len(selected_calibration), "message": f"[{dataset}] 校准缓存 {completed}/{len(selected_calibration)}"})
        if report:
            if not local_config.get("enabled"):
                local_message = "Local 分支已关闭"
            elif local_config.get("parameter_source") == "locked_u0":
                local_message = "加载锁定 Local 参数与独立 K=1 CDF"
            else:
                local_message = "拟合 Local 高斯参数"
            report({"current_dataset": dataset, "phase": "fit", "message": f"[{dataset}] {local_message}；Global 固定加载官方 VATEX 参数"})
        parameters = _load_global_parameters(repository_root, config)
        if not local_config.get("enabled"):
            pass
        elif local_config.get("parameter_source") == "locked_u0":
            parameters.update(_load_locked_local_parameters(repository_root, config, dataset))
        else:
            parameters.update(_fit_local_parameters(calibration_windows_features, config, devices[0]))
            if local_config.get("conditional", {}).get("enabled", False):
                parameters["patch_temporal_conditional"] = _fit_conditional_local_parameter(
                    calibration_windows_features, config, devices[0]
                )
        if report:
            report({"current_dataset": dataset, "phase": "calibration_score", "completed": 0, "total": len(selected_calibration), "message": f"[{dataset}] 评分校准窗口"})
        calibration_started = monotonic()
        calibration_windows = _score_windows(
            repository_root, selected_calibration, cache_root, context, dataset, config, parameters, devices[0],
            report=(lambda done, total: report({"current_dataset": dataset, "phase": "calibration_score", "completed": done, "total": total, "message": f"[{dataset}] 校准评分 {done}/{total}"}) if report and (done == total or done % max(1, total // 20) == 0) else None),
            score_global=reused_global is None,
        )
        if reused_global is not None:
            calibration_windows = _attach_reused_global(
                calibration_windows, reused_global, dataset=dataset, split="calibration"
            )
        if report:
            report({"current_dataset": dataset, "phase": "evaluation_score", "completed": 0, "total": len(evaluation), "message": f"[{dataset}] 使用 {', '.join(devices)} 评分 evaluation（{len(evaluation)} 条）"})
        evaluation_started = monotonic()
        def evaluation_progress(done: int, total: int) -> None:
            if report and (done == total or done % max(1, total // 100) == 0):
                elapsed = max(monotonic() - evaluation_started, 1e-6)
                rate = done / elapsed
                eta = (total - done) / rate if rate else 0.0
                report({"current_dataset": dataset, "phase": "evaluation_score", "completed": done, "total": total, "rate_videos_per_second": rate, "eta_seconds": eta, "message": f"[{dataset}] evaluation {done}/{total} ({done / total:.1%})，{rate:.2f} 视频/秒，ETA {eta / 60:.1f} 分钟"})
        evaluation_windows = _score_evaluation(
            repository_root, evaluation, cache_root, context, dataset, config,
            parameters, devices, evaluation_progress, worker_log_dir=worker_log_dir,
            score_global=reused_global is None,
        )
        if reused_global is not None:
            evaluation_windows = _attach_reused_global(
                evaluation_windows, reused_global, dataset=dataset, split="evaluation"
            )
        if report:
            report({"current_dataset": dataset, "phase": "aggregate", "message": f"[{dataset}] 仅用 calibration real 执行窗口校准和视频聚合"})
        calibrated_windows, videos = _calibrate_and_aggregate(calibration_windows, evaluation_windows, config)
        local_params_sha256 = None
        if artifact_dir is not None:
            if local_config.get("conditional", {}).get("enabled", False):
                local_params_sha256 = _save_fitted_conditional_artifacts(
                    artifact_dir,
                    dataset,
                    selected_calibration,
                    calibration_windows,
                    parameters["patch_temporal_conditional"],
                    config,
                )
            else:
                local_params_sha256 = _save_fitted_local_artifacts(
                    artifact_dir,
                    dataset,
                    selected_calibration,
                    calibration_windows,
                    parameters,
                    config,
                )
        calibrated_windows["split"] = "calibration"
        evaluation_windows["split"] = "evaluation"
        all_windows.extend([calibrated_windows, evaluation_windows])
        all_videos.append(videos)
        metadata["datasets"][str(dataset)] = {
            "calibration_videos": len(selected_calibration),
            "evaluation_videos": len(videos),
            "excluded_short_calibration_videos": excluded_calibration,
            "excluded_short_evaluation_videos": excluded_evaluation,
            "parameters": sorted(parameters),
            "devices": devices,
            "calibration_score_seconds": monotonic() - calibration_started,
            "evaluation_score_seconds": monotonic() - evaluation_started,
            "local_params_sha256": local_params_sha256,
            "primary_peak_vram_gib": (
                torch.cuda.max_memory_allocated(primary_device) / 1024**3
                if primary_device.type == "cuda"
                else 0.0
            ),
        }
        if report:
            report({"current_dataset": dataset, "phase": "dataset_completed", "message": f"[{dataset}] 完成：evaluation {len(videos)} 条视频"})
    windows = pd.concat(all_windows, ignore_index=True)
    windows = windows.drop(columns=["_input_order"], errors="ignore")
    return windows, pd.concat(all_videos, ignore_index=True), metadata


def run_local_candidate_matrix_from_cache(
    repository_root: Path,
    config: dict,
    candidates: dict[str, str | dict[str, object]],
    *,
    report: Callable[[dict], None] | None = None,
    artifact_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """一次扫描严格缓存，完成多个 Local dynamics/likelihood 对照。"""

    if not candidates:
        raise ValueError("trajectory matrix 至少需要一个候选")
    runtime = config["runtime"]
    cache_root = repository_root / runtime["cache_dir"]
    context = prepare_feature_cache(
        cache_root,
        expected_contract=None,
        policy="strict",
        create=False,
        required_cache_kind="patch_embeddings",
    )
    reused_global, reused_identity = _load_reused_global_windows(
        repository_root, config, context.contract_sha256
    )
    if reused_global is None:
        raise ValueError("trajectory matrix 必须显式配置已验证的 reuse_global_run")
    devices = [str(item) for item in runtime.get("devices", [])] or [str(runtime["device"])]
    if len(devices) != 1:
        raise ValueError("trajectory matrix 当前要求单卡，避免多进程重复持有五组参数")
    device = devices[0]
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("配置请求 CUDA，但 PyTorch 未检测到 CUDA")
        target = torch.device(device)
        free_bytes, _ = torch.cuda.mem_get_info(target)
        if free_bytes / 1024**3 < float(runtime["minimum_free_gib"]):
            raise RuntimeError("trajectory matrix 的目标 GPU 空闲显存不足")
    else:
        target = torch.device(device)

    global_parameters = _load_global_parameters(repository_root, config)
    all_windows: dict[str, list[pd.DataFrame]] = {name: [] for name in candidates}
    all_videos: dict[str, list[pd.DataFrame]] = {name: [] for name in candidates}
    candidate_specs = {
        name: ({"dynamics": spec, "conditional": False} if isinstance(spec, str) else dict(spec))
        for name, spec in candidates.items()
    }
    for name, spec in candidate_specs.items():
        if not isinstance(spec.get("dynamics"), str):
            raise ValueError(f"候选 {name} 缺少 dynamics")
        spec.setdefault("conditional", False)
    metadata: dict[str, object] = {
        "cache_root": str(cache_root),
        "cache_contract_sha256": context.contract_sha256,
        "reused_global": reused_identity,
        "candidates": candidate_specs,
        "datasets": {},
    }
    packed_reader = PackedCacheReader(cache_root)
    identity_columns = [
        "video_id", "dataset", "subset", "source_model", "video_path",
        "window_id", "_input_order",
    ]
    global_columns = list(GLOBAL_WINDOW_COLUMNS)

    for dataset_value in config["data"]["datasets"]:
        dataset = str(dataset_value)
        if target.type == "cuda":
            torch.cuda.reset_peak_memory_stats(target)
        manifests = resolve_dataset_manifests(repository_root, config, dataset)
        calibration = load_manifest(str(manifests.calibration))
        evaluation = load_manifest(str(manifests.evaluation))
        _ensure_disjoint(calibration, evaluation, dataset)
        policy = str(config["data"]["short_video_policy"])
        calibration, excluded_calibration = _apply_short_video_policy(
            calibration, dataset, "calibration", policy
        )
        evaluation, excluded_evaluation = _apply_short_video_policy(
            evaluation, dataset, "evaluation", policy
        )
        selected_calibration = _choose_calibration(
            calibration,
            dataset,
            int(config["calibration"]["real_videos_per_dataset"]),
            int(config["calibration"]["seed"]),
        )
        if report:
            report({"message": f"[{dataset}] 一次读取真实校准窗口，拟合 {len(candidates)} 个轨迹候选"})
        calibration_features = []
        for _, row in selected_calibration.iterrows():
            payload = _load_cache_payload(
                repository_root, cache_root, row, context, packed_reader
            )
            calibration_features.extend(
                _window_features(payload, row, int(config["sampling"]["num_windows"]))
            )

        candidate_configs: dict[str, dict] = {}
        candidate_parameters: dict[str, object] = {}
        scoring_candidates: dict[str, tuple[dict, StableGaussianParams]] = {}
        conditional_scoring_candidates: dict[
            str, tuple[dict, ConditionalGaussianParams]
        ] = {}
        for name, spec in candidate_specs.items():
            candidate_config = copy.deepcopy(config)
            candidate_config["method"]["local"]["dynamics"] = spec["dynamics"]
            candidate_config["method"]["local"]["conditional"]["enabled"] = bool(
                spec["conditional"]
            )
            if "covariance_estimator" in spec:
                candidate_config["method"]["local"]["covariance_estimator"] = str(
                    spec["covariance_estimator"]
                )
            parameters = dict(global_parameters)
            candidate_configs[name] = candidate_config
            if spec["conditional"]:
                fitted = _fit_conditional_local_parameter(
                    calibration_features, candidate_config, device
                )
                candidate_parameters[name] = fitted
                conditional_scoring_candidates[name] = (
                    candidate_config["method"]["local"], fitted
                )
            else:
                parameters.update(
                    _fit_local_parameters(calibration_features, candidate_config, device)
                )
                candidate_parameters[name] = parameters
                scoring_candidates[name] = (
                    candidate_config["method"]["local"], parameters["patch_temporal"]
                )
            gc.collect()
            if target.type == "cuda":
                torch.cuda.empty_cache()

        def progress_callback(done: int, total: int) -> None:
            if report and (done == total or done % max(1, total // 50) == 0):
                report({
                    "current_dataset": dataset,
                    "phase": "trajectory_matrix_score",
                    "completed": done,
                    "total": total,
                    "message": f"[{dataset}] trajectory matrix {done}/{total}",
                })

        started = monotonic()
        calibration_matrix = _score_windows(
            repository_root,
            selected_calibration,
            cache_root,
            context,
            dataset,
            config,
            global_parameters,
            device,
            score_global=False,
            local_candidates=scoring_candidates or None,
            conditional_candidates=conditional_scoring_candidates or None,
        )
        calibration_matrix = _attach_reused_global(
            calibration_matrix, reused_global, dataset=dataset, split="calibration"
        )
        evaluation_matrix = _score_windows(
            repository_root,
            evaluation,
            cache_root,
            context,
            dataset,
            config,
            global_parameters,
            device,
            score_global=False,
            local_candidates=scoring_candidates or None,
            conditional_candidates=conditional_scoring_candidates or None,
            report=progress_callback,
        )
        evaluation_matrix = _attach_reused_global(
            evaluation_matrix, reused_global, dataset=dataset, split="evaluation"
        )
        score_seconds = monotonic() - started

        parameter_hashes = {}
        for name in candidates:
            raw_column = f"patch_temporal_raw__{name}"
            calibration_one = calibration_matrix[
                identity_columns + global_columns + [raw_column]
            ].rename(columns={raw_column: "patch_temporal_raw"})
            evaluation_one = evaluation_matrix[
                identity_columns + global_columns + [raw_column]
            ].rename(columns={raw_column: "patch_temporal_raw"})
            calibrated, videos = _calibrate_and_aggregate(
                calibration_one, evaluation_one, candidate_configs[name]
            )
            calibrated["split"] = "calibration"
            evaluation_one["split"] = "evaluation"
            candidate_windows = pd.concat(
                [calibrated, evaluation_one], ignore_index=True
            ).drop(columns=["_input_order"], errors="ignore")
            candidate_windows.insert(0, "variant", name)
            videos.insert(0, "variant", name)
            all_windows[name].append(candidate_windows)
            all_videos[name].append(videos)
            if artifact_dir is not None:
                candidate_dir = artifact_dir / "candidates" / name
                if candidate_specs[name]["conditional"]:
                    parameter_hashes[name] = _save_fitted_conditional_artifacts(
                        candidate_dir,
                        dataset,
                        selected_calibration,
                        calibration_one,
                        candidate_parameters[name],
                        candidate_configs[name],
                    )
                else:
                    parameter_hashes[name] = _save_fitted_local_artifacts(
                        candidate_dir,
                        dataset,
                        selected_calibration,
                        calibration_one,
                        candidate_parameters[name],
                        candidate_configs[name],
                    )
        metadata["datasets"][dataset] = {
            "calibration_videos": len(selected_calibration),
            "evaluation_videos": len(evaluation),
            "excluded_short_calibration_videos": excluded_calibration,
            "excluded_short_evaluation_videos": excluded_evaluation,
            "score_seconds_all_candidates": score_seconds,
            "local_params_sha256": parameter_hashes,
            "primary_peak_vram_gib": (
                torch.cuda.max_memory_allocated(target) / 1024**3
                if target.type == "cuda"
                else 0.0
            ),
        }
    windows = pd.concat(
        [pd.concat(all_windows[name], ignore_index=True) for name in candidates],
        ignore_index=True,
    )
    videos = pd.concat(
        [pd.concat(all_videos[name], ignore_index=True) for name in candidates],
        ignore_index=True,
    )
    return windows, videos, metadata


def run_trajectory_matrix_from_cache(
    repository_root: Path,
    config: dict,
    candidates: dict[str, str],
    *,
    report: Callable[[dict], None] | None = None,
    artifact_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """兼容 Stage 2 入口；全部候选使用无条件 Gaussian。"""

    return run_local_candidate_matrix_from_cache(
        repository_root,
        config,
        candidates,
        report=report,
        artifact_dir=artifact_dir,
    )


def build_bootstrap(videos: pd.DataFrame, config: dict) -> pd.DataFrame:
    """输出相对 Global-only 与 Local-only 的视频级 AUC bootstrap。"""

    comparisons = []
    if "global_score" in videos:
        comparisons.append(("final_score", "global_score", "final_vs_global"))
    if "local_score" in videos:
        comparisons.append(("final_score", "local_score", "final_vs_local"))
    if not comparisons:
        return pd.DataFrame()
    from evaluation.metrics import paired_bootstrap
    return paired_bootstrap(videos, seed=int(config["calibration"]["seed"]), iterations=int(config["metrics"]["bootstrap_iterations"]), comparisons=tuple(comparisons))
