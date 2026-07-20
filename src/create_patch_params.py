"""为 PatchSTALL 创建 patch 级校准参数（.npz）。

低内存实现：

1. 第一遍：流式读取真实视频 patch cache，并对 patch token / patch transition
   做 reservoir sampling，用于白化拟合。
2. 第二遍：再次流式读取，并为每个真实视频计算一个校准分数。

这样可以避免一次性把所有真实视频堆到内存中。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch

from dataset_utils import _is_missing_window, load_csv
from dataset_utils_patch import _get_patch_cache_path
from patch_math import bottomk_mean
from patch_matching import patch_temporal_delta
from stall import log_likelihood, whitening_transform as apply_whitening
from whitening_transform import WhiteningTransform


def _fit_whitening(flat_mat: np.ndarray) -> WhiteningTransform:
    return WhiteningTransform(data=flat_mat)


def _get_mu_W(wt: WhiteningTransform):
    return wt.mean_.cpu().numpy(), wt.whitening_matrix_.cpu().numpy()


def temporal_run_bottomk_mean(arr: np.ndarray, ratio: float, run_length: int) -> np.ndarray:
    """最低 k 个连续同 patch 似然 run 的均值。

    输入为 [N, T, P]。先在时间连续窗口内对每个 patch 求均值，再对每个视频
    的所有 patch-run 取最低 ratio。
    """
    if arr.ndim != 3:
        raise ValueError(f"期望 [N,T,P] 似然数组，实际为 {arr.shape}")
    if run_length <= 1:
        return bottomk_mean(arr, ratio)
    if arr.shape[1] < run_length:
        return bottomk_mean(arr, ratio)

    windows = []
    for start in range(0, arr.shape[1] - run_length + 1):
        windows.append(arr[:, start : start + run_length, :].mean(axis=1))
    run_scores = np.stack(windows, axis=1)  # [N, T-run+1, P]
    return bottomk_mean(run_scores, ratio)


def spatiotemporal_run_bottomk_mean(
    arr: np.ndarray,
    ratio: float,
    run_length: int,
    grid_size: tuple[int, int] | None,
    region_size: int,
) -> np.ndarray:
    """在局部时空似然窗口上取 bottom-k。

    输入为 [N, T, P]。先对连续帧求平均，再对每个 patch 周围的局部空间邻域
    求平均。目标是捕获持续局部伪影，而不是孤立单 token 似然异常值。
    """
    if arr.ndim != 3:
        raise ValueError(f"期望 [N,T,P] 似然数组，实际为 {arr.shape}")
    if grid_size is None:
        raise ValueError("spatiotemporal_run_bottomk_mean 需要 grid_size")
    gh, gw = grid_size
    if arr.shape[2] != gh * gw:
        raise ValueError(f"Patch 数量 {arr.shape[2]} 与 grid_size={grid_size} 不匹配")
    if run_length <= 1 or arr.shape[1] < run_length:
        time_scores = arr
    else:
        windows = []
        for start in range(0, arr.shape[1] - run_length + 1):
            windows.append(arr[:, start : start + run_length, :].mean(axis=1))
        time_scores = np.stack(windows, axis=1)  # [N, T-run+1, P]

    if region_size <= 1:
        return bottomk_mean(time_scores, ratio)
    if region_size % 2 == 0:
        raise ValueError("--aggregation-region-size 必须为奇数，以便形成中心邻域")

    x = time_scores.reshape(time_scores.shape[0], time_scores.shape[1], gh, gw)
    pad = region_size // 2
    padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="edge")
    spatial_sum = np.zeros_like(x, dtype=np.float32)
    for dy in range(region_size):
        for dx in range(region_size):
            spatial_sum += padded[:, :, dy : dy + gh, dx : dx + gw]
    local_scores = spatial_sum / float(region_size * region_size)
    return bottomk_mean(local_scores.reshape(arr.shape[0], time_scores.shape[1], gh * gw), ratio)


def aggregate_scores(
    arr: np.ndarray,
    mode: str,
    bottomk_ratio: float,
    temporal_run_length: int = 3,
    grid_size: tuple[int, int] | None = None,
    aggregation_region_size: int = 3,
) -> np.ndarray:
    if mode == "mean":
        return arr.reshape(arr.shape[0], -1).mean(axis=1)
    if mode == "min":
        return arr.reshape(arr.shape[0], -1).min(axis=1)
    if mode == "bottomk_mean":
        return bottomk_mean(arr, bottomk_ratio)
    if mode == "temporal_run_bottomk_mean":
        return temporal_run_bottomk_mean(arr, bottomk_ratio, temporal_run_length)
    if mode == "spatiotemporal_run_bottomk_mean":
        return spatiotemporal_run_bottomk_mean(
            arr,
            bottomk_ratio,
            temporal_run_length,
            grid_size,
            aggregation_region_size,
        )
    raise ValueError(f"未知 aggregation mode: {mode}")


def iter_real_patch_cache(
    csv_path: str,
    patch_emb_cache_dir: str,
    duration_sec: int = 2,
    compact: bool = True,
    max_real_videos: int | None = None,
) -> Iterator[tuple[str, dict]]:
    df = load_csv(csv_path)
    window_col = f"{duration_sec}_sec_idxs"
    reals = df[df["subset"] == "real"].reset_index(drop=True)
    cache_root = Path(patch_emb_cache_dir)

    yielded = 0
    for _, row in reals.iterrows():
        if compact and _is_missing_window(row.get(window_col)):
            continue

        stem = Path(row["video_path"]).stem
        cache_path = _get_patch_cache_path(
            cache_root,
            row["subset"],
            row["source_model"],
            stem,
            duration_sec,
            compact,
        )
        if not cache_path.exists():
            continue

        payload = torch.load(cache_path, weights_only=True)
        yielded += 1
        yield row["video_path"], payload
        if max_real_videos is not None and yielded >= max_real_videos:
            break


def reservoir_update(
    reservoir: np.ndarray | None,
    stream_mat: np.ndarray,
    max_samples: int,
    seen: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, int]:
    """从 stream_mat 中 reservoir-sample 行到固定大小的 numpy buffer。"""
    if stream_mat.ndim != 2:
        raise ValueError(f"期望 2-D 矩阵，实际形状为 {stream_mat.shape}")

    if reservoir is None:
        take = min(max_samples, len(stream_mat))
        reservoir = np.empty((max_samples, stream_mat.shape[1]), dtype=np.float32)
        reservoir[:take] = stream_mat[:take]
        seen = take
        start_idx = take
    else:
        start_idx = 0

    for row in stream_mat[start_idx:]:
        seen += 1
        if seen <= max_samples:
            reservoir[seen - 1] = row
            continue
        j = rng.randint(0, seen)
        if j < max_samples:
            reservoir[j] = row

    return reservoir, seen


def collect_fit_samples(
    csv_path: str,
    patch_emb_cache_dir: str,
    duration_sec: int,
    compact: bool,
    max_patches_for_fit: int,
    seed: int,
    patch_temp_mode: str,
    match_radius: int,
    top_m: int,
    temperature: float,
    lambda_dist: float,
    patch_region_size: int,
    max_real_videos: int | None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    rng = np.random.RandomState(seed)

    patch_reservoir = None
    temp_reservoir = None
    patch_seen = 0
    temp_seen = 0
    grid_size = None
    video_count = 0

    print("流式读取真实 patch cache，用于拟合采样...", flush=True)
    for video_count, (_, payload) in enumerate(
        iter_real_patch_cache(csv_path, patch_emb_cache_dir, duration_sec, compact, max_real_videos), start=1
    ):
        patch = payload["patch"].numpy().astype(np.float32)  # [T, P, D]
        this_grid = tuple(int(x) for x in payload["grid_size"])
        if grid_size is None:
            grid_size = this_grid
        elif grid_size != this_grid:
            raise ValueError(f"Patch 网格尺寸不一致: {grid_size} vs {this_grid}")

        flat_patch = patch.reshape(-1, patch.shape[-1])
        patch_reservoir, patch_seen = reservoir_update(
            patch_reservoir, flat_patch, max_patches_for_fit, patch_seen, rng
        )

        patch_temp = patch_temporal_delta(
            patch,
            grid_size=this_grid,
            mode=patch_temp_mode,
            radius=match_radius,
            top_m=top_m,
            temperature=temperature,
            lambda_dist=lambda_dist,
            region_size=patch_region_size,
        )  # [T-1, P, D]
        flat_temp = patch_temp.reshape(-1, patch_temp.shape[-1]).astype(np.float32)
        temp_reservoir, temp_seen = reservoir_update(
            temp_reservoir, flat_temp, max_patches_for_fit, temp_seen, rng
        )

        if video_count % 200 == 0:
            print(
                f"  sampled_from_videos={video_count} "
                f"patch_seen={patch_seen} temp_seen={temp_seen}",
                flush=True,
            )

    if video_count == 0 or patch_reservoir is None or temp_reservoir is None:
        raise ValueError("未找到真实视频 patch cache 文件。请先构建 patch cache。")

    patch_fit = patch_reservoir[: min(max_patches_for_fit, patch_seen)]
    temp_fit = temp_reservoir[: min(max_patches_for_fit, temp_seen)]
    return patch_fit, temp_fit, grid_size


def compute_calibration_scores(
    csv_path: str,
    patch_emb_cache_dir: str,
    duration_sec: int,
    compact: bool,
    mu_patch_spat: np.ndarray,
    W_patch_spat: np.ndarray,
    mu_patch_temp: np.ndarray,
    W_patch_temp: np.ndarray,
    aggregation: str,
    bottomk_ratio: float,
    temporal_run_length: int,
    aggregation_region_size: int,
    patch_temp_mode: str,
    match_radius: int,
    top_m: int,
    temperature: float,
    lambda_dist: float,
    patch_region_size: int,
    max_real_videos: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    spat_scores = []
    temp_scores = []

    print("流式读取真实 patch cache，用于校准打分...", flush=True)
    for i, (_, payload) in enumerate(
        iter_real_patch_cache(csv_path, patch_emb_cache_dir, duration_sec, compact, max_real_videos), start=1
    ):
        patch_np = payload["patch"].numpy().astype(np.float32)
        patch = patch_np[np.newaxis]  # [1, T, P, D]
        grid_size = tuple(int(x) for x in payload["grid_size"])

        patch_spat_ll = log_likelihood(
            apply_whitening(patch, mu_patch_spat, W_patch_spat)
        )  # [1, T, P]
        spat_scores.append(
            float(
                aggregate_scores(
                    patch_spat_ll,
                    aggregation,
                    bottomk_ratio,
                    temporal_run_length,
                    grid_size,
                    aggregation_region_size,
                )[0]
            )
        )

        patch_temp = patch_temporal_delta(
            patch_np,
            grid_size=grid_size,
            mode=patch_temp_mode,
            radius=match_radius,
            top_m=top_m,
            temperature=temperature,
            lambda_dist=lambda_dist,
            region_size=patch_region_size,
        )[np.newaxis]  # [1, T-1, P, D]
        patch_temp_ll = log_likelihood(
            apply_whitening(patch_temp, mu_patch_temp, W_patch_temp)
        )  # [1, T-1, P]
        temp_scores.append(
            float(
                aggregate_scores(
                    patch_temp_ll,
                    aggregation,
                    bottomk_ratio,
                    temporal_run_length,
                    grid_size,
                    aggregation_region_size,
                )[0]
            )
        )

        if i % 200 == 0:
            print(f"  已打分视频数={i}", flush=True)

    return np.array(spat_scores, dtype=np.float32), np.array(temp_scores, dtype=np.float32)


def build_patch_params(
    csv_path: str,
    patch_emb_cache_dir: str,
    duration_sec: int = 2,
    compact: bool = True,
    max_patches_for_fit: int = 300_000,
    aggregation: str = "bottomk_mean",
    bottomk_ratio: float = 0.05,
    temporal_run_length: int = 3,
    aggregation_region_size: int = 3,
    seed: int = 42,
    patch_temp_mode: str = "same_grid",
    match_radius: int = 2,
    top_m: int = 4,
    temperature: float = 0.07,
    lambda_dist: float = 0.01,
    patch_region_size: int = 1,
    max_real_videos: int | None = None,
) -> dict:
    patch_fit, temp_fit, grid_size = collect_fit_samples(
        csv_path=csv_path,
        patch_emb_cache_dir=patch_emb_cache_dir,
        duration_sec=duration_sec,
        compact=compact,
        max_patches_for_fit=max_patches_for_fit,
        seed=seed,
        patch_temp_mode=patch_temp_mode,
        match_radius=match_radius,
        top_m=top_m,
        temperature=temperature,
        lambda_dist=lambda_dist,
        patch_region_size=patch_region_size,
        max_real_videos=max_real_videos,
    )

    print(f"使用 {len(patch_fit)} 个 patch token 拟合 patch spatial whitening", flush=True)
    wt_spat = _fit_whitening(patch_fit)
    mu_patch_spat, W_patch_spat = _get_mu_W(wt_spat)

    print(f"使用 {len(temp_fit)} 个 patch transition 拟合 patch temporal whitening", flush=True)
    wt_temp = _fit_whitening(temp_fit)
    mu_patch_temp, W_patch_temp = _get_mu_W(wt_temp)

    calib_patch_spat_scores, calib_patch_temp_scores = compute_calibration_scores(
        csv_path=csv_path,
        patch_emb_cache_dir=patch_emb_cache_dir,
        duration_sec=duration_sec,
        compact=compact,
        mu_patch_spat=mu_patch_spat,
        W_patch_spat=W_patch_spat,
        mu_patch_temp=mu_patch_temp,
        W_patch_temp=W_patch_temp,
        aggregation=aggregation,
        bottomk_ratio=bottomk_ratio,
        temporal_run_length=temporal_run_length,
        aggregation_region_size=aggregation_region_size,
        patch_temp_mode=patch_temp_mode,
        match_radius=match_radius,
        top_m=top_m,
        temperature=temperature,
        lambda_dist=lambda_dist,
        patch_region_size=patch_region_size,
        max_real_videos=max_real_videos,
    )

    return {
        "mu_patch_spat": mu_patch_spat.astype(np.float32),
        "W_patch_spat": W_patch_spat.astype(np.float32),
        "calib_patch_spat_scores": calib_patch_spat_scores.astype(np.float32),
        "mu_patch_temp": mu_patch_temp.astype(np.float32),
        "W_patch_temp": W_patch_temp.astype(np.float32),
        "calib_patch_temp_scores": calib_patch_temp_scores.astype(np.float32),
        "patch_grid_size": np.array(grid_size, dtype=np.int32),
        "duration": np.array([duration_sec], dtype=np.int32),
        "aggregation_config": np.array(
            json.dumps(
                {
                    "mode": aggregation,
                    "bottomk_ratio": bottomk_ratio,
                    "temporal_run_length": temporal_run_length,
                    "aggregation_region_size": aggregation_region_size,
                    "seed": seed,
                    "max_patches_for_fit": max_patches_for_fit,
                    "implementation": "streaming_two_pass",
                    "temporal_feature_version": "l2_normalized_delta_v1",
                    "max_real_videos": max_real_videos,
                    "patch_temp_mode": patch_temp_mode,
                    "match_radius": match_radius,
                    "top_m": top_m,
                    "temperature": temperature,
                    "lambda_dist": lambda_dist,
                    "patch_region_size": patch_region_size,
                }
            )
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="从真实视频 patch cache 创建 patch 级校准参数。"
    )
    parser.add_argument("--csv", required=True, help="video_index.py 生成的 CSV")
    parser.add_argument("--patch-emb-cache", required=True, help="patch cache 根目录")
    parser.add_argument("--output", required=True, help="输出 .npz 路径")
    parser.add_argument("--duration", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("--compact", action="store_true", default=False)
    parser.add_argument("--real-only", action="store_true", default=False)
    parser.add_argument("--max-patches-for-fit", type=int, default=300000)
    parser.add_argument(
        "--aggregation",
        choices=[
            "mean",
            "min",
            "bottomk_mean",
            "temporal_run_bottomk_mean",
            "spatiotemporal_run_bottomk_mean",
        ],
        default="bottomk_mean",
    )
    parser.add_argument("--bottomk-ratio", type=float, default=0.05)
    parser.add_argument("--temporal-run-length", type=int, default=3)
    parser.add_argument("--aggregation-region-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-real-videos",
        type=int,
        default=None,
        help="限制真实校准视频数量，用于 debug/smoke run。",
    )
    parser.add_argument(
        "--patch-temp-mode",
        choices=[
            "same_grid",
            "same_grid_lag1",
            "same_grid_multilag",
            "same_grid_second_order",
            "same_grid_third_order",
            "same_grid_fourth_order",
            "same_grid_multilag_second_order",
            "motion_hard",
            "motion_soft",
        ],
        default="same_grid",
    )
    parser.add_argument("--match-radius", type=int, default=2)
    parser.add_argument("--top-m", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda-dist", type=float, default=0.01)
    parser.add_argument("--patch-region-size", type=int, default=1)
    args = parser.parse_args()

    if not args.real_only:
        raise ValueError("当前版本只支持 --real-only")

    params = build_patch_params(
        csv_path=args.csv,
        patch_emb_cache_dir=args.patch_emb_cache,
        duration_sec=args.duration,
        compact=args.compact,
        max_patches_for_fit=args.max_patches_for_fit,
        aggregation=args.aggregation,
        bottomk_ratio=args.bottomk_ratio,
        temporal_run_length=args.temporal_run_length,
        aggregation_region_size=args.aggregation_region_size,
        seed=args.seed,
        patch_temp_mode=args.patch_temp_mode,
        match_radius=args.match_radius,
        top_m=args.top_m,
        temperature=args.temperature,
        lambda_dist=args.lambda_dist,
        patch_region_size=args.patch_region_size,
        max_real_videos=args.max_real_videos,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **params)

    print(f"已保存 patch params: {out_path}")
    for key, value in params.items():
        if hasattr(value, "shape"):
            print(f"  {key}: {value.shape} {value.dtype}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
