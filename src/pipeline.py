"""Alpha STALL 从严格特征缓存到可发布结果的完整执行链路。"""

from __future__ import annotations

import hashlib
import multiprocessing
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Callable

import numpy as np
import pandas as pd
import torch

from branches.global_branch import GLOBAL_SPATIAL_AGGREGATION, GLOBAL_TEMPORAL_AGGREGATION
from branches.local_branch import LOCAL_AGGREGATION, local_d1_features, local_d2_features
from data.cache_contract import prepare_feature_cache, tensor_descriptor, validate_cache_entry
from data.manifest import load_manifest
from data.patch_cache import _get_patch_cache_path, cache_frame_indices
from data.packed_cache import PackedCacheReader
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


def _load_cache_payload(repository_root: Path, cache_root: Path, row: pd.Series, context, reader: PackedCacheReader | None = None) -> dict:
    """读取并逐条验证完整或多窗口严格缓存。"""

    stem = Path(str(row["video_path"])).stem
    cache_path = _get_patch_cache_path(
        cache_root, str(row["subset"]), str(row["source_model"]), stem, 2, False
    )
    cache_key = cache_path.relative_to(cache_root).as_posix()
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
    report: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """预取严格缓存后逐窗口评分；窗口的浮点计算顺序保持不变。"""

    method = config["method"]
    requested_k = int(config["sampling"]["num_windows"])
    records: list[dict] = []
    batch_size = int(config["runtime"].get("score_batch_size", 1))
    io_workers = int(config["runtime"].get("cache_io_workers", 1))
    indexed_rows = list(rows.iterrows())
    packed_reader = PackedCacheReader(cache_root)

    def load_batch(batch: list[tuple[int, pd.Series]]) -> list[tuple[int, pd.Series, dict]]:
        def load(item: tuple[int, pd.Series]) -> tuple[int, pd.Series, dict]:
            position, row = item
            return position, row, _load_cache_payload(repository_root, cache_root, row, context, packed_reader)
        with ThreadPoolExecutor(max_workers=min(io_workers, len(batch))) as executor:
            return list(executor.map(load, batch))

    batches = [indexed_rows[offset: offset + batch_size] for offset in range(0, len(indexed_rows), batch_size)]
    completed = 0
    # 下一批读取和当前批评分并行，减少 GPU 等待缓存 I/O 的空档。
    with ThreadPoolExecutor(max_workers=1) as prefetcher:
        future = prefetcher.submit(load_batch, batches[0]) if batches else None
        for batch_index in range(len(batches)):
            loaded = future.result()
            future = prefetcher.submit(load_batch, batches[batch_index + 1]) if batch_index + 1 < len(batches) else None
            batch_windows: list[tuple[dict, np.ndarray, np.ndarray]] = []
            for position, row, payload in loaded:
                for window_id, (global_values, patch_values) in enumerate(_window_features(payload, row, requested_k)):
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
            if method["global"]["enabled"]:
                scores["global_spatial_raw"], _ = score_gaussian_aggregate_float64(global_values, parameters["global_spatial"], GLOBAL_SPATIAL_AGGREGATION, device=device, compute_percentile=False)
                global_temporal = torch.nn.functional.normalize(torch.from_numpy(global_values[:, 1:] - global_values[:, :-1]), p=2, dim=-1, eps=1e-12).numpy()
                scores["global_t1_raw"], _ = score_gaussian_aggregate_float64(global_temporal, parameters["global_t1"], GLOBAL_TEMPORAL_AGGREGATION, device=device, compute_percentile=False)
            patch_tensor = torch.from_numpy(patch_values)
            if method["local"]["enabled"] and method["local"]["spatial_enabled"]:
                scores["patch_spatial_raw"], _ = score_gaussian_aggregate_float64(patch_values, parameters["patch_spatial"], LOCAL_AGGREGATION, device=device, compute_percentile=False)
            if method["local"]["enabled"] and method["local"]["temporal_enabled"]:
                temporal_features = local_d1_features(patch_tensor) if int(method["local"]["temporal_order"]) == 1 else local_d2_features(patch_tensor)
                scores["patch_temporal_raw"], _ = score_gaussian_aggregate_float64(temporal_features, parameters["patch_temporal"], LOCAL_AGGREGATION, device=device, compute_percentile=False)
            for index, (record, _, _) in enumerate(batch_windows):
                record.update({name: float(values[index]) for name, values in scores.items()})
                records.append(record)
            completed += len(loaded)
            if report is not None:
                report(completed, len(indexed_rows))
    if not records:
        raise ValueError(f"{dataset} 没有满足 16 帧窗口条件的视频")
    return pd.DataFrame(records)


def _score_rows_worker(
    repository_root: str,
    cache_root: str,
    context,
    dataset: str,
    config: dict,
    parameters: dict[str, StableGaussianParams],
    device: str,
    rows: pd.DataFrame,
    progress_queue=None,
) -> pd.DataFrame:
    """独立 CUDA 进程的评测分片入口；必须保持模块级以支持 spawn。"""

    # 每个 worker 最多发约 100 次进度。batch 大小未必整除该步长，必须按“跨过步长”而非
    # “恰好整除”判断，否则运行正常时 progress.json 也可能长期停在 0%。
    report_stride = max(1, len(rows) // 100)
    last_reported = 0

    def worker_report(done: int, total: int) -> None:
        nonlocal last_reported
        if progress_queue is not None and (done == total or done - last_reported >= report_stride):
            progress_queue.put((device, done, total))
            last_reported = done

    return _score_windows(
        Path(repository_root), rows, Path(cache_root), context, dataset, config,
        parameters, device, report=worker_report if progress_queue is not None else None,
    )


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
) -> pd.DataFrame:
    """单卡使用预取流水线；双卡按稳定行序切分 evaluation 并合并。"""

    if len(devices) == 1 or len(rows) < 2:
        return _score_windows(repository_root, rows, cache_root, context, dataset, config, parameters, devices[0], report=report)
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
                dataset, config, parameters, device, chunk, progress_queue,
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


def run_from_cache(
    repository_root: Path,
    config: dict,
    *,
    report: Callable[[dict], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
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
    metadata: dict[str, object] = {"cache_root": str(cache_root), "cache_contract_sha256": context.contract_sha256, "datasets": {}}
    for dataset in config["data"]["datasets"]:
        dataset = str(dataset)
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
        calibration_windows_features = []
        for completed, (_, row) in enumerate(selected_calibration.iterrows(), start=1):
            payload = _load_cache_payload(repository_root, cache_root, row, context, packed_reader)
            calibration_windows_features.extend(_window_features(payload, row, int(config["sampling"]["num_windows"])))
            if report and (completed == len(selected_calibration) or completed % max(1, len(selected_calibration) // 20) == 0):
                report({"current_dataset": dataset, "phase": "calibration_load", "completed": completed, "total": len(selected_calibration), "message": f"[{dataset}] 校准缓存 {completed}/{len(selected_calibration)}"})
        if report:
            report({"current_dataset": dataset, "phase": "fit", "message": f"[{dataset}] 拟合 Global/Local 高斯参数"})
        parameters = _fit_parameters(calibration_windows_features, config, devices[0])
        if report:
            report({"current_dataset": dataset, "phase": "calibration_score", "completed": 0, "total": len(selected_calibration), "message": f"[{dataset}] 评分校准窗口"})
        calibration_started = monotonic()
        calibration_windows = _score_windows(
            repository_root, selected_calibration, cache_root, context, dataset, config, parameters, devices[0],
            report=(lambda done, total: report({"current_dataset": dataset, "phase": "calibration_score", "completed": done, "total": total, "message": f"[{dataset}] 校准评分 {done}/{total}"}) if report and (done == total or done % max(1, total // 20) == 0) else None),
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
        evaluation_windows = _score_evaluation(repository_root, evaluation, cache_root, context, dataset, config, parameters, devices, evaluation_progress)
        if report:
            report({"current_dataset": dataset, "phase": "aggregate", "message": f"[{dataset}] 仅用 calibration real 执行窗口校准和视频聚合"})
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
            "devices": devices,
            "calibration_score_seconds": monotonic() - calibration_started,
            "evaluation_score_seconds": monotonic() - evaluation_started,
        }
        if report:
            report({"current_dataset": dataset, "phase": "dataset_completed", "message": f"[{dataset}] 完成：evaluation {len(videos)} 条视频"})
    windows = pd.concat(all_windows, ignore_index=True)
    windows = windows.drop(columns=["_input_order"], errors="ignore")
    return windows, pd.concat(all_videos, ignore_index=True), metadata


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
