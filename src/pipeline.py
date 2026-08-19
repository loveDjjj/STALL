"""Alpha STALL 从严格特征缓存到可发布结果的完整执行链路。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from branches.global_branch import GLOBAL_SPATIAL_AGGREGATION, GLOBAL_TEMPORAL_AGGREGATION
from branches.local_branch import LOCAL_AGGREGATION, local_d1_features, local_d2_features
from data.cache_contract import prepare_feature_cache, tensor_descriptor, validate_cache_entry
from data.manifest import load_manifest
from data.patch_cache import _get_patch_cache_path, cache_frame_indices
from data.sampling import parse_indices, uniform_windows
from math_utils import StableGaussianParams, WhiteningTransform, score_gaussian_aggregate_float64, stable_sorted


COMPONENTS = ("global_spatial", "global_t1", "patch_spatial", "patch_temporal")


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


def _load_cache_payload(repository_root: Path, cache_root: Path, row: pd.Series, context) -> dict:
    """读取并逐条验证完整或多窗口严格缓存。"""

    stem = Path(str(row["video_path"])).stem
    cache_path = _get_patch_cache_path(
        cache_root, str(row["subset"]), str(row["source_model"]), stem, 2, False
    )
    if not cache_path.is_file():
        raise FileNotFoundError(f"严格缓存缺失：{cache_path}")
    payload = torch.load(cache_path, weights_only=True)
    indices = cache_frame_indices(
        row,
        duration_sec=2,
        compact=False,
        cache_window_count=_cache_window_count(context),
    )
    validate_cache_entry(
        context,
        cache_path=cache_path,
        source_video_path=str(repository_root / row["video_path"]),
        frame_indices=indices,
        payload={
            "format": "torch_dict_global_patch_v1",
            "global": tensor_descriptor(payload["global"]),
            "patch": tensor_descriptor(payload["patch"]),
            "grid_size": [int(value) for value in payload["grid_size"]],
        },
    )
    return payload


def _window_features(payload: dict, row: pd.Series, requested_k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    downsample = parse_indices(row["downsample_idxs"])
    windows = uniform_windows(downsample, requested_k, window_frames=16)
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


def _temporal_patch(values: np.ndarray, order: int) -> np.ndarray:
    tensor = torch.as_tensor(values, dtype=torch.float32).unsqueeze(0)
    if order == 1:
        return local_d1_features(tensor).squeeze(0).numpy()
    if order == 2:
        return local_d2_features(tensor).squeeze(0).numpy()
    raise ValueError("method.local.temporal_order 只能是 1 或 2")


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


def _fit_parameter(features: np.ndarray, device: str) -> StableGaussianParams:
    transform = WhiteningTransform(features, device=device)
    return StableGaussianParams(
        mean=transform.mean_.detach().cpu().numpy().astype(np.float64),
        whitening=transform.whitening_matrix_.detach().cpu().numpy().astype(np.float64),
        calibration_raw=np.array([0.0, 1.0], dtype=np.float64),
    )


def _fit_parameters(calibration_windows: list[tuple[np.ndarray, np.ndarray]], config: dict, device: str) -> dict[str, StableGaussianParams]:
    method = config["method"]
    limit = int(config["runtime"]["max_features_for_fit"])
    if limit < 2:
        raise ValueError("runtime.max_features_for_fit 必须至少为 2")
    needed = {"global_spatial", "global_t1"} if method["global"]["enabled"] else set()
    if method["local"]["enabled"]:
        if method["local"]["spatial_enabled"]:
            needed.add("patch_spatial")
        if method["local"]["temporal_enabled"]:
            needed.add("patch_temporal")
    reservoirs: dict[str, np.ndarray | None] = {key: None for key in needed}
    seen = {key: 0 for key in needed}
    rng = np.random.default_rng(int(config["calibration"]["seed"]))
    order = int(method["local"]["temporal_order"])
    for global_values, patch_values in calibration_windows:
        items: dict[str, np.ndarray] = {}
        if "global_spatial" in needed:
            items["global_spatial"] = global_values
            items["global_t1"] = _temporal_global(global_values)
        if "patch_spatial" in needed:
            items["patch_spatial"] = patch_values
        if "patch_temporal" in needed:
            items["patch_temporal"] = _temporal_patch(patch_values, order)
        for name, values in items.items():
            reservoirs[name], seen[name] = _reservoir_add(reservoirs[name], values, limit, seen[name], rng)
    return {
        name: _fit_parameter(values[: min(limit, seen[name])], device)
        for name, values in reservoirs.items()
        if values is not None and seen[name] >= 2
    }


def _score_windows(repository_root: Path, rows: pd.DataFrame, cache_root: Path, context, dataset: str, config: dict, parameters: dict[str, StableGaussianParams], device: str) -> pd.DataFrame:
    method = config["method"]
    requested_k = int(config["sampling"]["num_windows"])
    records: list[dict] = []
    for _, row in rows.iterrows():
        payload = _load_cache_payload(repository_root, cache_root, row, context)
        windows = _window_features(payload, row, requested_k)
        for window_id, (global_values, patch_values) in enumerate(windows):
            record = {
                "video_id": _video_id(dataset, str(row["video_path"])),
                "dataset": dataset,
                "subset": str(row["subset"]),
                "source_model": str(row["source_model"]),
                "video_path": str(row["video_path"]),
                "window_id": window_id,
            }
            if method["global"]["enabled"]:
                spatial, _ = score_gaussian_aggregate_float64(global_values[None], parameters["global_spatial"], GLOBAL_SPATIAL_AGGREGATION, device=device, compute_percentile=False)
                temporal, _ = score_gaussian_aggregate_float64(_temporal_global(global_values)[None], parameters["global_t1"], GLOBAL_TEMPORAL_AGGREGATION, device=device, compute_percentile=False)
                record["global_spatial_raw"] = float(spatial[0])
                record["global_t1_raw"] = float(temporal[0])
            if method["local"]["enabled"] and method["local"]["spatial_enabled"]:
                spatial, _ = score_gaussian_aggregate_float64(patch_values[None], parameters["patch_spatial"], LOCAL_AGGREGATION, device=device, compute_percentile=False)
                record["patch_spatial_raw"] = float(spatial[0])
            if method["local"]["enabled"] and method["local"]["temporal_enabled"]:
                temporal_features = _temporal_patch(patch_values, int(method["local"]["temporal_order"]))
                temporal, _ = score_gaussian_aggregate_float64(temporal_features[None], parameters["patch_temporal"], LOCAL_AGGREGATION, device=device, compute_percentile=False)
                record["patch_temporal_raw"] = float(temporal[0])
            records.append(record)
    if not records:
        raise ValueError(f"{dataset} 没有满足 16 帧窗口条件的视频")
    return pd.DataFrame(records)


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
    if method["global"]["enabled"]:
        active_raw += ["global_spatial_raw", "global_t1_raw"]
    if method["local"]["enabled"] and method["local"]["spatial_enabled"]:
        active_raw.append("patch_spatial_raw")
    if method["local"]["enabled"] and method["local"]["temporal_enabled"]:
        active_raw.append("patch_temporal_raw")
    reference = calibration[calibration["subset"].eq("real")]
    if reference.empty:
        raise ValueError("校准窗口中没有真实视频")
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


def run_from_cache(repository_root: Path, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """执行配置指定的全部数据集，返回逐窗口、逐视频分数和运行元信息。"""

    runtime = config["runtime"]
    cache_root = repository_root / runtime["cache_dir"]
    context = prepare_feature_cache(cache_root, expected_contract=None, policy="strict", create=False, required_cache_kind="patch_embeddings")
    cached_window_count = _cache_window_count(context)
    if cached_window_count is not None and int(config["sampling"]["num_windows"]) > cached_window_count:
        raise ValueError(
            f"请求 K={config['sampling']['num_windows']}，但严格缓存只覆盖 K<={cached_window_count}；"
            "请使用兼容配置或重建缓存。"
        )
    device = str(runtime["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("配置请求 CUDA，但 PyTorch 未检测到 CUDA")
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
    metadata: dict[str, object] = {"cache_root": str(cache_root), "cache_contract_sha256": context.contract_sha256, "datasets": {}}
    for dataset in config["data"]["datasets"]:
        manifests = resolve_dataset_manifests(repository_root, config, str(dataset))
        calibration = load_manifest(str(manifests.calibration))
        evaluation = load_manifest(str(manifests.evaluation))
        _ensure_disjoint(calibration, evaluation, str(dataset))
        policy = str(config["data"]["short_video_policy"])
        calibration, excluded_calibration = _apply_short_video_policy(calibration, str(dataset), "calibration", policy)
        evaluation, excluded_evaluation = _apply_short_video_policy(evaluation, str(dataset), "evaluation", policy)
        selected_calibration = _choose_calibration(calibration, str(dataset), int(config["calibration"]["real_videos_per_dataset"]), int(config["calibration"]["seed"]))
        calibration_windows_features = []
        for _, row in selected_calibration.iterrows():
            payload = _load_cache_payload(repository_root, cache_root, row, context)
            calibration_windows_features.extend(_window_features(payload, row, int(config["sampling"]["num_windows"])))
        parameters = _fit_parameters(calibration_windows_features, config, device)
        calibration_windows = _score_windows(repository_root, selected_calibration, cache_root, context, str(dataset), config, parameters, device)
        evaluation_windows = _score_windows(repository_root, evaluation, cache_root, context, str(dataset), config, parameters, device)
        calibrated_windows, videos = _calibrate_and_aggregate(calibration_windows, evaluation_windows, config)
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
        }
    return pd.concat(all_windows, ignore_index=True), pd.concat(all_videos, ignore_index=True), metadata


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
