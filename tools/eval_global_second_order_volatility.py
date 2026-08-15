#!/usr/bin/env python3
"""评测 D3-style 全局二阶时序波动分支。

本工具不重新抽 DINOv3。它读取 ``video_index.py`` 生成的 enriched CSV 和
已有 global embedding cache，计算一个训练无关的全局二阶波动分数：

    1. 对一个固定窗口内的帧级 embedding 计算相邻帧距离 ``d_t``；
    2. 计算距离序列的二阶变化 ``d_{t+1} - d_t``；
    3. 取标准差作为真实视频倾向分数；
    4. 用独立 real calibration CSV 把 raw volatility 映射为百分位分数。

这不是 D3 官方复现：D3 主结果使用 XCLIP-B/16、中心裁剪/resize 后的帧和
自己的抽帧代码。本工具的目标是把 D3 的关键统计思想接入当前 STALL/DINOv3
缓存体系，用于快速验证它能否补强 original STALL/global-only。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset_utils import _is_missing_window, load_csv
from alpha_stalled.metrics import ScoreDirection, build_results_table


KEY_COLUMNS = ["subset", "source_model", "filename"]


@dataclass(frozen=True)
class ScoreConfig:
    duration: int
    fallback_durations: tuple[int, ...]
    distance: str
    standardize_embeddings: bool


def _load_embedding(cache_root: Path, row: pd.Series) -> np.ndarray:
    """读取一个视频的 global embedding cache。"""
    stem = Path(str(row["video_path"])).stem
    cache_path = cache_root / str(row["subset"]) / str(row["source_model"]) / f"{stem}.pt"
    if not cache_path.exists():
        raise FileNotFoundError(f"缺少 embedding cache: {cache_path}")

    payload = torch.load(cache_path, weights_only=True, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("embs", "emb", "global"):
            if key in payload:
                arr = payload[key]
                break
        else:
            tensor_values = [v for v in payload.values() if hasattr(v, "shape")]
            if not tensor_values:
                raise ValueError(f"{cache_path} 中未找到 tensor/array payload")
            arr = tensor_values[0]
    else:
        arr = payload

    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"{cache_path} embedding 形状应为 [T,D]，实际为 {arr.shape}")
    return arr


def _window_indices(row: pd.Series, durations: tuple[int, ...]) -> tuple[list[int], int] | None:
    """按优先级选择第一个可用窗口，返回 ``(frame_indices, used_duration)``。"""
    for duration in durations:
        value = row.get(f"{duration}_sec_idxs")
        if _is_missing_window(value):
            continue
        return [int(x) for x in json.loads(value)], int(duration)
    return None


def _slice_window(emb: np.ndarray, indices: list[int]) -> np.ndarray:
    """按 enriched CSV 窗口切片。

    旧 cache 有两种形式：完整 downsample embedding，或已经 compact 到窗口。
    如果 CSV frame index 超出 cache 长度，则退化为取 cache 的前 ``len(indices)``
    帧；这与当前 STALL cache loader 的兼容策略一致。
    """
    if not indices:
        return emb[:0]
    if max(indices) < len(emb):
        return emb[indices]
    return emb[: len(indices)]


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return x / denom


def raw_volatility_score(emb: np.ndarray, config: ScoreConfig) -> float:
    """计算单视频全局二阶波动 raw score，越大越像真实视频。"""
    if config.standardize_embeddings:
        emb = _normalize_rows(emb)
    if len(emb) < 3:
        return float("nan")

    left = emb[:-1]
    right = emb[1:]
    if config.distance == "l2":
        first_order = np.linalg.norm(right - left, axis=1)
    elif config.distance == "cosine_distance":
        left_n = _normalize_rows(left)
        right_n = _normalize_rows(right)
        first_order = 1.0 - np.sum(left_n * right_n, axis=1)
    elif config.distance == "cosine_similarity":
        left_n = _normalize_rows(left)
        right_n = _normalize_rows(right)
        first_order = np.sum(left_n * right_n, axis=1)
    else:
        raise ValueError(f"未知 distance: {config.distance}")

    second_order = first_order[1:] - first_order[:-1]
    if len(second_order) == 0:
        return float("nan")
    return float(np.std(second_order, ddof=0))


def score_index(
    csv_path: Path,
    emb_cache: Path,
    config: ScoreConfig,
    *,
    real_only: bool = False,
) -> pd.DataFrame:
    df = load_csv(str(csv_path))
    if real_only:
        df = df[df["subset"].astype(str) == "real"].copy()

    rows: list[dict] = []
    skipped_missing_window = 0
    skipped_nan = 0
    for _, row in df.iterrows():
        selected = _window_indices(row, config.fallback_durations)
        if selected is None:
            skipped_missing_window += 1
            continue
        indices, used_duration = selected
        emb = _load_embedding(emb_cache, row)
        window = _slice_window(emb, indices)
        score = raw_volatility_score(window, config)
        if not np.isfinite(score):
            skipped_nan += 1
            continue
        rows.append(
            {
                "subset": str(row["subset"]),
                "source_model": str(row["source_model"]),
                "filename": Path(str(row["video_path"])).name,
                "duration": used_duration,
                "duration_policy": ",".join(str(x) for x in config.fallback_durations),
                "distance": config.distance,
                "standardize_embeddings": config.standardize_embeddings,
                "global_second_order_volatility_raw": score,
            }
        )

    out = pd.DataFrame(rows)
    print(
        f"{csv_path}: scored={len(out)} skipped_missing_window={skipped_missing_window} "
        f"skipped_nan={skipped_nan}",
        flush=True,
    )
    return out


def add_real_percentile(eval_scores: pd.DataFrame, calib_scores: pd.DataFrame) -> pd.DataFrame:
    calib = np.sort(
        calib_scores["global_second_order_volatility_raw"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=np.float64)
    )
    if len(calib) == 0:
        raise ValueError("real calibration 分数为空，无法做百分位校准")

    out = eval_scores.copy()
    raw = out["global_second_order_volatility_raw"].to_numpy(dtype=np.float64)
    out["global_second_order_volatility_score"] = (
        np.searchsorted(calib, raw, side="right") / float(len(calib))
    )
    return out


def metrics_for_score(df: pd.DataFrame, score_col: str, seed: int) -> pd.DataFrame:
    return build_results_table(
        df[["subset", "source_model", score_col]].rename(columns={score_col: "final_score"}),
        {"final_score": ScoreDirection.HIGHER_IS_REAL},
        seed=seed,
        skip_global_compare=True,
        verbose=False,
    )


def _read_component(path: Path, score_col: str, out_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in KEY_COLUMNS + [score_col] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} 缺少列: {missing}")
    out = df[KEY_COLUMNS + [score_col]].copy()
    for col in KEY_COLUMNS:
        out[col] = out[col].astype(str)
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce")
    return out.rename(columns={score_col: out_col})


def sweep_fusion(
    global_csv: Path,
    volatility_scores: pd.DataFrame,
    global_score_col: str,
    alphas: list[float],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """融合 original STALL/global-only 与全局二阶波动分支。

    ``alpha`` 是 global-only 权重，``1-alpha`` 是 volatility 权重。
    """
    global_df = _read_component(global_csv, global_score_col, "global_score")
    vol_df = volatility_scores[KEY_COLUMNS + ["global_second_order_volatility_score"]].copy()
    for col in KEY_COLUMNS:
        vol_df[col] = vol_df[col].astype(str)
    vol_df = vol_df.rename(columns={"global_second_order_volatility_score": "volatility_score"})

    merged = global_df.merge(vol_df, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["global_score", "volatility_score"]
    )
    if merged.empty:
        raise ValueError("global 与 volatility 分数没有可用交集")

    metric_rows: list[pd.DataFrame] = []
    score_rows: list[pd.DataFrame] = []
    for alpha in alphas:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha 必须位于 [0,1]: {alpha}")
        fused = merged.copy()
        fused["alpha_global"] = float(alpha)
        fused["final_score"] = (
            float(alpha) * fused["global_score"]
            + (1.0 - float(alpha)) * fused["volatility_score"]
        )
        metrics = metrics_for_score(fused, "final_score", seed=seed)
        metrics.insert(0, "alpha_global", float(alpha))
        metric_rows.append(metrics)
        score_rows.append(fused)

    return pd.concat(metric_rows, ignore_index=True), pd.concat(score_rows, ignore_index=True)


def parse_alphas(value: str) -> list[float]:
    if ":" in value:
        start_s, stop_s, step_s = value.split(":")
        start = float(start_s)
        stop = float(stop_s)
        step = float(step_s)
        if step <= 0:
            raise ValueError("alpha step 必须 > 0")
        values = []
        current = start
        while current <= stop + 1e-12:
            values.append(round(current, 10))
            current += step
        return values
    return [float(x) for x in value.split(",") if x.strip()]


def parse_durations(value: str | None, primary_duration: int) -> tuple[int, ...]:
    if value is None or not value.strip():
        return (int(primary_duration),)
    out = tuple(int(x) for x in value.split(",") if x.strip())
    if not out:
        raise ValueError("--fallback-durations 不能为空")
    invalid = [x for x in out if x not in {1, 2, 3, 4}]
    if invalid:
        raise ValueError(f"fallback duration 只能在 1/2/3/4 中选择: {invalid}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="评测 D3-style 全局二阶时序波动分支。")
    parser.add_argument("--eval-csv", type=Path, required=True)
    parser.add_argument("--calib-real-csv", type=Path, required=True)
    parser.add_argument("--emb-cache", type=Path, required=True)
    parser.add_argument("--duration", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument(
        "--fallback-durations",
        default=None,
        help=(
            "逗号分隔的窗口优先级，例如 '2,1' 表示优先 2s，缺失时回退到 1s。"
            "默认只使用 --duration。"
        ),
    )
    parser.add_argument(
        "--distance",
        choices=["l2", "cosine_distance", "cosine_similarity"],
        default="l2",
    )
    parser.add_argument("--standardize-embeddings", action="store_true", default=False)
    parser.add_argument("--output-score-csv", type=Path, required=True)
    parser.add_argument("--output-metrics-csv", type=Path, required=True)
    parser.add_argument("--global-csv", type=Path, default=None)
    parser.add_argument("--global-score-col", default="final_score")
    parser.add_argument("--alphas", default="0:1:0.05")
    parser.add_argument("--output-fusion-metrics-csv", type=Path, default=None)
    parser.add_argument("--output-fusion-scores-csv", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = ScoreConfig(
        duration=args.duration,
        fallback_durations=parse_durations(args.fallback_durations, args.duration),
        distance=args.distance,
        standardize_embeddings=args.standardize_embeddings,
    )
    calib_scores = score_index(
        args.calib_real_csv,
        args.emb_cache,
        config,
        real_only=True,
    )
    eval_scores = score_index(args.eval_csv, args.emb_cache, config)
    scored = add_real_percentile(eval_scores, calib_scores)
    metrics = metrics_for_score(scored, "global_second_order_volatility_score", seed=args.seed)

    args.output_score_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output_score_csv, index=False)
    metrics.to_csv(args.output_metrics_csv, index=False)
    print(f"已保存 volatility 分数: {args.output_score_csv}")
    print(f"已保存 volatility 指标: {args.output_metrics_csv}")

    avg = metrics[metrics["Generative Model"] == "Average"]
    if not avg.empty:
        row = avg.iloc[0]
        print(
            "Volatility-only Average: "
            f"AUC={row['final_score AUC']:.4f} AP={row['final_score AP']:.4f}",
            flush=True,
        )

    if args.global_csv is not None:
        if args.output_fusion_metrics_csv is None or args.output_fusion_scores_csv is None:
            raise ValueError(
                "提供 --global-csv 时必须同时提供 --output-fusion-metrics-csv "
                "和 --output-fusion-scores-csv"
            )
        fusion_metrics, fusion_scores = sweep_fusion(
            args.global_csv,
            scored,
            args.global_score_col,
            parse_alphas(args.alphas),
            seed=args.seed,
        )
        args.output_fusion_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        args.output_fusion_scores_csv.parent.mkdir(parents=True, exist_ok=True)
        fusion_metrics.to_csv(args.output_fusion_metrics_csv, index=False)
        fusion_scores.to_csv(args.output_fusion_scores_csv, index=False)
        print(f"已保存融合指标: {args.output_fusion_metrics_csv}")
        print(f"已保存融合分数: {args.output_fusion_scores_csv}")
        avg_rows = fusion_metrics[fusion_metrics["Generative Model"] == "Average"].copy()
        if not avg_rows.empty:
            best_auc = avg_rows.sort_values(
                ["final_score AUC", "final_score AP", "alpha_global"],
                ascending=[False, False, True],
            ).iloc[0]
            best_ap = avg_rows.sort_values(
                ["final_score AP", "final_score AUC", "alpha_global"],
                ascending=[False, False, True],
            ).iloc[0]
            print(
                "Best fusion by AUC: "
                f"alpha_global={best_auc['alpha_global']:.2f} "
                f"AUC={best_auc['final_score AUC']:.4f} "
                f"AP={best_auc['final_score AP']:.4f}",
                flush=True,
            )
            print(
                "Best fusion by AP: "
                f"alpha_global={best_ap['alpha_global']:.2f} "
                f"AUC={best_ap['final_score AUC']:.4f} "
                f"AP={best_ap['final_score AP']:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
