"""PatchSTALL 的快速 patch-only 评测。

本脚本刻意不加载 DINOv3。它读取已有 patch cache 文件，并用 torch batch 对
compact、固定形状视频打分。主要用于需要大量 score-weight sweep 的 patch-only
消融。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dataset_utils import _is_missing_window, load_csv
from dataset_utils_patch import _get_patch_cache_path
from metrics import Score, ScoreDirection, get_results_df, print_results


def _load_agg_config(npz_value) -> dict:
    if isinstance(npz_value, np.ndarray):
        npz_value = npz_value.item()
    return json.loads(str(npz_value))


def _as_tensor(data, key: str, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(data[key].astype(np.float32)).to(device=device)


class FastPatchScorer:
    def __init__(self, params_path: str, device: str = "cuda"):
        self.params_path = params_path
        if device == "cuda" and torch.cuda.is_available():
            self.devices = [torch.device(f"cuda:{idx}") for idx in range(torch.cuda.device_count())]
        else:
            self.devices = [torch.device("cpu")]
        self.device = self.devices[0]
        data = np.load(params_path, allow_pickle=True)

        self.mu_spat = _as_tensor(data, "mu_patch_spat", self.device)
        self.W_spat = _as_tensor(data, "W_patch_spat", self.device)
        self.mu_temp = _as_tensor(data, "mu_patch_temp", self.device)
        self.W_temp = _as_tensor(data, "W_patch_temp", self.device)
        self._param_cache = {
            self.device: (self.mu_spat, self.W_spat, self.mu_temp, self.W_temp)
        }
        self._executor = ThreadPoolExecutor(max_workers=len(self.devices)) if len(self.devices) > 1 else None
        self.calib_spat = np.sort(data["calib_patch_spat_scores"].astype(np.float32))
        self.calib_temp = np.sort(data["calib_patch_temp_scores"].astype(np.float32))
        self.patch_grid_size = tuple(int(x) for x in data["patch_grid_size"].tolist())

        self.aggregation_config = _load_agg_config(data["aggregation_config"])
        self.params_mode = self.aggregation_config.get("patch_temp_mode", "same_grid")
        self.params_bottomk_ratio = float(self.aggregation_config.get("bottomk_ratio", 0.05))
        self.params_temporal_run_length = int(self.aggregation_config.get("temporal_run_length", 3))
        self.params_aggregation_region_size = int(self.aggregation_config.get("aggregation_region_size", 3))
        self.params_patch_region_size = int(self.aggregation_config.get("patch_region_size", 1))
        self.temporal_feature_version = self.aggregation_config.get("temporal_feature_version")
        if self.temporal_feature_version != "l2_normalized_delta_v1":
            raise ValueError(
                f"参数由不支持的 temporal feature version 创建: {self.temporal_feature_version!r}"
            )

    def validate(self, patch_temp_mode: str):
        if patch_temp_mode != self.params_mode:
            raise ValueError(
                f"Patch params 期望 patch_temp_mode={self.params_mode}，实际为 {patch_temp_mode}"
            )
        supported = {
            "same_grid",
            "same_grid_lag1",
            "same_grid_multilag",
            "same_grid_second_order",
            "same_grid_multilag_second_order",
        }
        if patch_temp_mode not in supported:
            raise ValueError(
                f"Fast scorer 只支持 same-grid 系列模式，实际为 {patch_temp_mode}"
            )

    @staticmethod
    def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(x, p=2, dim=-1, eps=1e-12)

    def _params_for_device(self, device: torch.device):
        if device not in self._param_cache:
            self._param_cache[device] = (
                self.mu_spat.to(device=device, non_blocking=True),
                self.W_spat.to(device=device, non_blocking=True),
                self.mu_temp.to(device=device, non_blocking=True),
                self.W_temp.to(device=device, non_blocking=True),
            )
        return self._param_cache[device]

    @staticmethod
    def pool_patch_regions(
        patch: torch.Tensor,
        grid_size: tuple[int, int],
        region_size: int,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        if region_size <= 1:
            return patch, grid_size
        gh, gw = grid_size
        if patch.shape[2] != gh * gw:
            raise ValueError(f"Patch 数量 {patch.shape[2]} 与网格 {grid_size} 不匹配")
        pooled_h = gh // region_size
        pooled_w = gw // region_size
        if pooled_h < 1 or pooled_w < 1:
            raise ValueError(f"region_size={region_size} 对网格 {grid_size} 过大")
        crop_h = pooled_h * region_size
        crop_w = pooled_w * region_size
        x = patch.reshape(patch.shape[0], patch.shape[1], gh, gw, patch.shape[-1])
        x = x[:, :, :crop_h, :crop_w, :]
        x = x.reshape(
            patch.shape[0],
            patch.shape[1],
            pooled_h,
            region_size,
            pooled_w,
            region_size,
            patch.shape[-1],
        )
        pooled = x.mean(dim=(3, 5)).reshape(patch.shape[0], patch.shape[1], pooled_h * pooled_w, patch.shape[-1])
        return pooled, (pooled_h, pooled_w)

    def temporal_features(self, patch: torch.Tensor, mode: str, region_size: int) -> torch.Tensor:
        if region_size > 1:
            patch, _ = self.pool_patch_regions(patch, self.patch_grid_size, region_size)

        if mode in {"same_grid", "same_grid_lag1"}:
            return self._l2_normalize(patch[:, 1:] - patch[:, :-1])

        chunks = []
        if mode in {"same_grid_multilag", "same_grid_multilag_second_order"}:
            for lag in (1, 2, 4):
                if patch.shape[1] > lag:
                    chunks.append(self._l2_normalize(patch[:, lag:] - patch[:, :-lag]))

        if mode in {"same_grid_second_order", "same_grid_multilag_second_order"}:
            if patch.shape[1] < 3:
                raise ValueError(f"视频帧数过少，无法使用 {mode}: T={patch.shape[1]}")
            accel = patch[:, 2:] - 2.0 * patch[:, 1:-1] + patch[:, :-2]
            chunks.append(self._l2_normalize(accel))

        if not chunks:
            raise ValueError(f"不支持或无效的 temporal mode: {mode}")
        return torch.cat(chunks, dim=1)

    @staticmethod
    def log_likelihood_from_white(white: torch.Tensor) -> torch.Tensor:
        # 保留 stall.log_likelihood 使用的精确高斯 log-likelihood 尺度。
        # 校准百分位会与已保存 raw score 比较，因此即使常数项不影响排序也不能丢弃。
        dim = white.shape[-1]
        return -0.5 * (dim * np.log(2.0 * np.pi) + torch.sum(white * white, dim=-1))

    @staticmethod
    def bottomk_mean(ll: torch.Tensor, ratio: float) -> torch.Tensor:
        flat = ll.reshape(ll.shape[0], -1)
        k = max(1, int(np.ceil(flat.shape[1] * ratio)))
        vals = torch.topk(flat, k=k, largest=False, dim=1).values
        return vals.mean(dim=1)

    def temporal_run_bottomk_mean(
        self, ll: torch.Tensor, ratio: float, run_length: int
    ) -> torch.Tensor:
        if run_length <= 1 or ll.shape[1] < run_length:
            return self.bottomk_mean(ll, ratio)
        windows = []
        for start in range(0, ll.shape[1] - run_length + 1):
            windows.append(ll[:, start : start + run_length, :].mean(dim=1))
        run_scores = torch.stack(windows, dim=1)
        return self.bottomk_mean(run_scores, ratio)

    def spatiotemporal_run_bottomk_mean(
        self,
        ll: torch.Tensor,
        ratio: float,
        run_length: int,
        aggregation_region_size: int,
    ) -> torch.Tensor:
        if aggregation_region_size <= 1:
            return self.temporal_run_bottomk_mean(ll, ratio, run_length)
        if aggregation_region_size % 2 == 0:
            raise ValueError("aggregation_region_size 必须为奇数，以便形成中心邻域")
        if ll.ndim != 3:
            raise ValueError(f"期望 [N,T,P] 似然张量，实际为 {tuple(ll.shape)}")
        gh, gw = self.patch_grid_size
        if ll.shape[2] != gh * gw:
            raise ValueError(f"Patch 数量 {ll.shape[2]} 与网格 {self.patch_grid_size} 不匹配")

        if run_length <= 1 or ll.shape[1] < run_length:
            time_scores = ll
        else:
            windows = []
            for start in range(0, ll.shape[1] - run_length + 1):
                windows.append(ll[:, start : start + run_length, :].mean(dim=1))
            time_scores = torch.stack(windows, dim=1)

        x = time_scores.reshape(time_scores.shape[0], time_scores.shape[1], gh, gw)
        k = aggregation_region_size
        pad = k // 2
        # replicate padding 与 create_patch_params.py 的边界 padding 一致。
        padded = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="replicate")
        local_sum = torch.zeros_like(x)
        for dy in range(k):
            for dx in range(k):
                local_sum = local_sum + padded[:, :, dy : dy + gh, dx : dx + gw]
        local_scores = local_sum / float(k * k)
        return self.bottomk_mean(local_scores.reshape(ll.shape[0], time_scores.shape[1], gh * gw), ratio)

    def _aggregate(
        self,
        ll: torch.Tensor,
        mode: str,
        bottomk_ratio: float,
        temporal_run_length: int,
    ) -> torch.Tensor:
        if mode == "mean":
            return ll.reshape(ll.shape[0], -1).mean(dim=1)
        if mode == "min":
            return ll.reshape(ll.shape[0], -1).amin(dim=1)
        if mode == "bottomk_mean":
            return self.bottomk_mean(ll, bottomk_ratio)
        if mode == "temporal_run_bottomk_mean":
            return self.temporal_run_bottomk_mean(ll, bottomk_ratio, temporal_run_length)
        if mode == "spatiotemporal_run_bottomk_mean":
            return self.spatiotemporal_run_bottomk_mean(
                ll,
                bottomk_ratio,
                temporal_run_length,
                self.params_aggregation_region_size,
            )
        raise ValueError(f"未知 aggregation mode: {mode}")

    @staticmethod
    def percentile(scores: np.ndarray, calib_sorted: np.ndarray) -> np.ndarray:
        return np.searchsorted(calib_sorted, scores, side="right") / len(calib_sorted)

    @torch.inference_mode()
    def score_batch_on_device(
        self,
        patch_batch: torch.Tensor | np.ndarray,
        device: torch.device,
        patch_temp_mode: str,
        patch_spat_weight: float,
        patch_temp_weight: float,
        aggregation: str,
        bottomk_ratio: float,
        temporal_run_length: int,
        patch_region_size: int,
    ) -> dict[str, np.ndarray]:
        if isinstance(patch_batch, np.ndarray):
            patch = torch.from_numpy(patch_batch.astype(np.float32))
        else:
            patch = patch_batch
            if patch.dtype != torch.float32:
                patch = patch.float()
        patch = patch.to(device, non_blocking=True)

        mu_spat, W_spat, mu_temp, W_temp = self._params_for_device(device)

        spat_white = torch.matmul(patch - mu_spat, W_spat)
        spat_ll = self.log_likelihood_from_white(spat_white)
        spat_agg = self._aggregate(spat_ll, aggregation, bottomk_ratio, temporal_run_length)

        temp = self.temporal_features(patch, patch_temp_mode, patch_region_size)
        temp_white = torch.matmul(temp - mu_temp, W_temp)
        temp_ll = self.log_likelihood_from_white(temp_white)
        temp_agg = self._aggregate(temp_ll, aggregation, bottomk_ratio, temporal_run_length)

        spat_agg_np = spat_agg.detach().cpu().numpy()
        temp_agg_np = temp_agg.detach().cpu().numpy()
        spat_pct = self.percentile(spat_agg_np, self.calib_spat)
        temp_pct = self.percentile(temp_agg_np, self.calib_temp)
        denom = max(patch_spat_weight + patch_temp_weight, 1e-8)
        patch_final = (patch_spat_weight * spat_pct + patch_temp_weight * temp_pct) / denom
        return {
            "patch_spat_percentile": spat_pct.astype(np.float32),
            "patch_temp_percentile": temp_pct.astype(np.float32),
            "patch_final_score": patch_final.astype(np.float32),
            "final_score": patch_final.astype(np.float32),
        }

    @torch.inference_mode()
    def score_batch(
        self,
        patch_batch: torch.Tensor | np.ndarray,
        patch_temp_mode: str,
        patch_spat_weight: float,
        patch_temp_weight: float,
        aggregation: str,
        bottomk_ratio: float,
        temporal_run_length: int,
        patch_region_size: int,
    ) -> dict[str, np.ndarray]:
        if len(self.devices) == 1 or patch_batch.shape[0] < 2:
            return self.score_batch_on_device(
                patch_batch,
                self.device,
                patch_temp_mode,
                patch_spat_weight,
                patch_temp_weight,
                aggregation,
                bottomk_ratio,
                temporal_run_length,
                patch_region_size,
            )

        chunks = torch.tensor_split(patch_batch, len(self.devices), dim=0)
        jobs = [
            (chunk, device)
            for chunk, device in zip(chunks, self.devices)
            if chunk.shape[0] > 0
        ]
        futures = [
            self._executor.submit(
                self.score_batch_on_device,
                chunk,
                device,
                patch_temp_mode,
                patch_spat_weight,
                patch_temp_weight,
                aggregation,
                bottomk_ratio,
                temporal_run_length,
                patch_region_size,
            )
            for chunk, device in jobs
        ]
        parts = [future.result() for future in futures]
        return {
            key: np.concatenate([part[key] for part in parts], axis=0)
            for key in parts[0].keys()
        }


def iter_cache_jobs(csv_path: str, patch_cache_root: str, duration: int, compact: bool, debug_n: int | None):
    df = load_csv(csv_path)
    window_col = f"{duration}_sec_idxs"
    if debug_n is not None:
        df = (
            df.groupby(["subset", "source_model"], group_keys=False)
            .apply(lambda g: g.head(debug_n))
            .reset_index(drop=True)
        )

    root = Path(patch_cache_root)
    for _, row in df.iterrows():
        if compact and _is_missing_window(row.get(window_col)):
            continue
        video_path = row["video_path"]
        stem = Path(video_path).stem
        cache_path = _get_patch_cache_path(
            root, row["subset"], row["source_model"], stem, duration, compact
        )
        if not cache_path.exists():
            raise FileNotFoundError(f"缺少 patch cache: {cache_path}")
        yield {
            "subset": row["subset"],
            "source_model": row["source_model"],
            "filename": Path(video_path).name,
            "video_path": video_path,
            "cache_path": cache_path,
        }


def load_cache_batch(jobs: list[dict]) -> tuple[torch.Tensor, tuple[int, int]]:
    patches = []
    grid_size = None
    shape = None
    for job in jobs:
        payload = torch.load(job["cache_path"], weights_only=True, map_location="cpu")
        patch = payload["patch"]
        if patch.dtype != torch.float32:
            patch = patch.float()
        this_grid = tuple(int(x) for x in payload["grid_size"])
        if grid_size is None:
            grid_size = this_grid
            shape = patch.shape
        elif grid_size != this_grid or patch.shape != shape:
            raise ValueError(
                f"Fast batching 要求固定 grid/shape。实际为 {patch.shape}/{this_grid}，"
                f"期望 {shape}/{grid_size}，文件 {job['cache_path']}"
            )
        patches.append(patch)
    return torch.stack(patches, dim=0), grid_size


def run(args) -> pd.DataFrame:
    scorer = FastPatchScorer(args.patch_params, device=args.score_device)
    scorer.validate(args.patch_temp_mode)
    aggregation = args.aggregation or scorer.aggregation_config.get("mode", "bottomk_mean")
    bottomk_ratio = args.bottomk_ratio if args.bottomk_ratio is not None else scorer.params_bottomk_ratio
    temporal_run_length = (
        args.temporal_run_length
        if args.temporal_run_length is not None
        else scorer.params_temporal_run_length
    )
    patch_region_size = (
        args.patch_region_size
        if args.patch_region_size is not None
        else scorer.params_patch_region_size
    )
    if patch_region_size != scorer.params_patch_region_size:
        raise ValueError(
            f"Patch params 期望 patch_region_size={scorer.params_patch_region_size}，"
            f"实际为 {patch_region_size}"
        )

    jobs = list(iter_cache_jobs(args.csv, args.patch_emb_cache, args.duration, args.compact, args.debug_n))
    rows = []
    print(
        f"快速 patch 打分: {len(jobs)} 个 cached videos, devices={','.join(str(d) for d in scorer.devices)}, "
        f"batch={args.score_batch_size}, aggregation={aggregation}, bottomk={bottomk_ratio}, "
        f"run_length={temporal_run_length}, patch_region_size={patch_region_size}",
        flush=True,
    )

    for start in tqdm(range(0, len(jobs), args.score_batch_size), desc="快速打分", unit="batch"):
        batch_jobs = jobs[start : start + args.score_batch_size]
        patch_batch, grid_size = load_cache_batch(batch_jobs)
        if grid_size != scorer.patch_grid_size:
            raise ValueError(f"参数 grid_size={scorer.patch_grid_size}, cache grid_size={grid_size}")
        scores = scorer.score_batch(
            patch_batch,
            patch_temp_mode=args.patch_temp_mode,
            patch_spat_weight=args.patch_spat_weight,
            patch_temp_weight=args.patch_temp_weight,
            aggregation=aggregation,
            bottomk_ratio=bottomk_ratio,
            temporal_run_length=temporal_run_length,
            patch_region_size=patch_region_size,
        )

        for i, job in enumerate(batch_jobs):
            rows.append(
                {
                    "subset": job["subset"],
                    "source_model": job["source_model"],
                    "filename": job["filename"],
                    "fusion": "patch_only",
                    "global_weight": None,
                    "patch_spat_weight": args.patch_spat_weight,
                    "patch_temp_weight": args.patch_temp_weight,
                    "patch_temp_mode": args.patch_temp_mode,
                    "match_radius": None,
                    "top_m": None,
                    "temperature": None,
                    "lambda_dist": None,
                    "aggregation": aggregation,
                    "bottomk_ratio": bottomk_ratio,
                    "temporal_run_length": temporal_run_length,
                    "patch_region_size": patch_region_size,
                    "patch_spat_percentile": float(scores["patch_spat_percentile"][i]),
                    "patch_temp_percentile": float(scores["patch_temp_percentile"][i]),
                    "patch_final_score": float(scores["patch_final_score"][i]),
                    "final_score": float(scores["final_score"][i]),
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="从 patch cache 快速执行 GPU patch-only 评测。")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--patch-emb-cache", required=True)
    parser.add_argument("--patch-params", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--duration", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("--compact", action="store_true", default=False)
    parser.add_argument("--debug-n", type=int, default=None)
    parser.add_argument("--score-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--patch-spat-weight", type=float, default=0.7)
    parser.add_argument("--patch-temp-weight", type=float, default=0.3)
    parser.add_argument(
        "--aggregation",
        choices=[
            "mean",
            "min",
            "bottomk_mean",
            "temporal_run_bottomk_mean",
            "spatiotemporal_run_bottomk_mean",
        ],
        default=None,
    )
    parser.add_argument("--bottomk-ratio", type=float, default=None)
    parser.add_argument("--temporal-run-length", type=int, default=None)
    parser.add_argument("--patch-region-size", type=int, default=None)
    parser.add_argument(
        "--patch-temp-mode",
        choices=[
            "same_grid",
            "same_grid_lag1",
            "same_grid_multilag",
            "same_grid_second_order",
            "same_grid_multilag_second_order",
        ],
        required=True,
    )
    args = parser.parse_args()

    df = run(args)
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"已保存逐视频分数 -> {out}")

    results_df = get_results_df(
        df[["subset", "source_model"]],
        {
            "final_score": Score(
                value=df["final_score"].to_numpy(),
                direction=ScoreDirection.HIGHER_IS_REAL,
            )
        },
    )
    print_results(results_df)


if __name__ == "__main__":
    main()
