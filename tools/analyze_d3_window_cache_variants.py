#!/usr/bin/env python3
"""用已有 DINOv3 full embedding cache 复查 D3 window 二阶统计。

目的：

`eval_d3_exact_from_frames.py` 会按 D3 协议从 JPEG frame folders 重新读图、
裁边、resize，再抽 DINOv3 embedding。本脚本不重新跑 encoder，而是读取
`cache/embeddings/genvideo` 中的 full DINOv3 embedding cache，并按 D3
计划的 `d3_start_time` 近似切出同一 3s@8fps 窗口，再计算二阶统计。

这能拆分两个因素：

1. D3 random start/window 是否本身弱；
2. D3 frame-folder/JPEG/crop/resize 预处理是否让 DINOv3 二阶信号变弱。

输出用于诊断，不是 D3-XCLIP 复现。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = (
    REPO_ROOT
    / "results/journal_experiments/global_second_order_volatility/genvideo/"
    / "d3_comparable_protocol/d3_exact_protocol/d3_exact_eval/dinov3_l2_smoke100"
)
DEFAULT_PROTOCOL_DIR = (
    REPO_ROOT
    / "results/journal_experiments/global_second_order_volatility/genvideo/"
    / "d3_comparable_protocol/d3_exact_protocol"
)
DEFAULT_INDEX = REPO_ROOT / "cache/indexes/genvideo.csv"
DEFAULT_EMB_CACHE = REPO_ROOT / "cache/embeddings/genvideo"
DEFAULT_EXISTING_VOL = (
    REPO_ROOT
    / "results/journal_experiments/global_second_order_volatility/genvideo/"
    / "genvideo_holdout_d3style_dinov3_l2_2s_fallback_1s_scores.csv"
)


def infer_source(path: Path) -> str:
    return path.name.replace("_head1000_DINOv3-L-local_l2_scores.csv", "")


def load_d3_score_rows(eval_dir: Path, pattern: str) -> pd.DataFrame:
    frames = []
    for path in sorted(eval_dir.glob(pattern)):
        df = pd.read_csv(path)
        if df.empty:
            continue
        source = infer_source(path)
        df["eval_source_model"] = source
        df["subset"] = np.where(df["label"].astype(int).eq(0), "real", "annotated")
        df["source_model"] = np.where(df["subset"].eq("real"), "MSR-VTT", source)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"{eval_dir} 中没有匹配 {pattern} 的 D3 score CSV")
    out = pd.concat(frames, ignore_index=True)
    for col in ["subset", "source_model", "filename"]:
        out[col] = out[col].astype(str)
    return out


def load_index(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["filename"] = df["video_path"].map(lambda x: Path(str(x)).name)
    required = {"subset", "source_model", "filename", "fps", "downsample_idxs"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} 缺少列: {sorted(missing)}")
    for col in ["subset", "source_model", "filename"]:
        df[col] = df[col].astype(str)
    return df


def load_plan(protocol_dir: Path) -> pd.DataFrame:
    plan = pd.read_csv(protocol_dir / "d3_exact_video_plan.csv")
    required = {
        "subset",
        "source_model",
        "filename",
        "d3_start_time",
        "expected_d3_read_frames",
    }
    missing = required.difference(plan.columns)
    if missing:
        raise ValueError(f"d3_exact_video_plan.csv 缺少列: {sorted(missing)}")
    for col in ["subset", "source_model", "filename"]:
        plan[col] = plan[col].astype(str)
    return plan


def load_existing_vol(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["subset", "source_model", "filename"]:
        df[col] = df[col].astype(str)
    df["existing_volatility_raw"] = pd.to_numeric(
        df["global_second_order_volatility_raw"], errors="coerce"
    )
    return df[["subset", "source_model", "filename", "existing_volatility_raw"]]


def cache_path(cache_root: Path, subset: str, source_model: str, filename: str) -> Path:
    return cache_root / subset / source_model / f"{Path(filename).stem}.pt"


def load_emb(path: Path) -> np.ndarray:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        for key in ("embs", "emb", "global"):
            if key in payload:
                payload = payload[key]
                break
    if hasattr(payload, "numpy"):
        payload = payload.numpy()
    arr = np.asarray(payload, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"{path} embedding shape 应为 [T,D]，实际 {arr.shape}")
    return arr


def d3_window_positions(downsample_idxs: list[int], fps: float, start_time: int, read_frames: int) -> list[int]:
    start_frame = float(start_time) * float(fps)
    positions = [i for i, frame_idx in enumerate(downsample_idxs) if float(frame_idx) >= start_frame]
    return positions[:read_frames]


def normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x / denom


def variant_scores(emb: np.ndarray) -> dict[str, float]:
    if len(emb) < 3:
        return {k: np.nan for k in VARIANT_ORDER}
    first_l2 = np.linalg.norm(emb[1:] - emb[:-1], axis=1)
    second_l2 = first_l2[1:] - first_l2[:-1]

    emb_n = normalize_rows(emb)
    first_cos_dist = 1.0 - np.sum(emb_n[1:] * emb_n[:-1], axis=1)
    second_cos_dist = first_cos_dist[1:] - first_cos_dist[:-1]

    diff = emb[1:] - emb[:-1]
    diff_n = normalize_rows(diff)
    first_diff_l2 = np.linalg.norm(diff_n[1:] - diff_n[:-1], axis=1)

    return {
        "cache_d3win_l2_std": float(np.std(second_l2, ddof=0)),
        "cache_d3win_l2_std_sample": float(np.std(second_l2, ddof=1)) if len(second_l2) > 1 else np.nan,
        "cache_d3win_l2_abs_mean": float(np.mean(np.abs(second_l2))),
        "cache_d3win_cosdist_std": float(np.std(second_cos_dist, ddof=0)),
        "cache_d3win_cosdist_abs_mean": float(np.mean(np.abs(second_cos_dist))),
        "cache_d3win_diffnorm_l2_mean": float(np.mean(first_diff_l2)),
        "cache_d3win_diffnorm_l2_std": float(np.std(first_diff_l2, ddof=0)),
    }


VARIANT_ORDER = [
    "d3_framefolder_raw",
    "existing_volatility_raw",
    "cache_d3win_l2_std",
    "cache_d3win_l2_std_sample",
    "cache_d3win_l2_abs_mean",
    "cache_d3win_cosdist_std",
    "cache_d3win_cosdist_abs_mean",
    "cache_d3win_diffnorm_l2_mean",
    "cache_d3win_diffnorm_l2_std",
]


def metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for source, sub in df.groupby("eval_source_model", sort=True):
        for method in VARIANT_ORDER:
            valid = sub[["label", method]].replace([np.inf, -np.inf], np.nan).dropna()
            if valid["label"].nunique() != 2:
                continue
            y_real = 1 - valid["label"].astype(int).to_numpy()
            score = valid[method].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "source_model": source,
                    "method": method,
                    "n_valid": int(len(valid)),
                    "n_real": int((valid["label"].astype(int) == 0).sum()),
                    "n_fake": int((valid["label"].astype(int) == 1).sum()),
                    "ap": float(average_precision_score(y_real, score)),
                    "auc": float(roc_auc_score(y_real, score)),
                }
            )
    per = pd.DataFrame(rows)
    macro = (
        per.groupby("method", as_index=False)
        .agg(
            n_sources=("source_model", "nunique"),
            mean_ap=("ap", "mean"),
            mean_auc=("auc", "mean"),
            min_ap=("ap", "min"),
        )
    )
    order = {name: i for i, name in enumerate(VARIANT_ORDER)}
    macro["_order"] = macro["method"].map(order)
    macro = macro.sort_values(["mean_ap", "_order"], ascending=[False, True]).drop(columns=["_order"])
    return per, macro


def write_markdown(path: Path, per: pd.DataFrame, macro: pd.DataFrame, scored: pd.DataFrame) -> None:
    corr = scored[["d3_framefolder_raw", "cache_d3win_l2_std", "existing_volatility_raw"]].corr()
    macro_map = macro.set_index("method")
    d3_ap = float(macro_map.loc["d3_framefolder_raw", "mean_ap"])
    cache_ap = float(macro_map.loc["cache_d3win_l2_std", "mean_ap"])
    existing_ap = float(macro_map.loc["existing_volatility_raw", "mean_ap"])
    lines = [
        "# D3 window cache-variant 诊断",
        "",
        "本实验在同一批 D3 diagnostic rows 上比较三类二阶信号：",
        "",
        "1. `d3_framefolder_raw`：从 D3 frame folders 重新读 JPEG、裁边/resize 后提 DINOv3 的 raw score。",
        "2. `cache_d3win_*`：从已有 full embedding cache 中按 D3 start time 近似切 8/16 帧窗口后计算 raw score。",
        "3. `existing_volatility_raw`：已有 STALL/DINOv3 2s fallback 1s window-cache volatility raw score。",
        "",
        "## Macro summary",
        "",
        "| method | mean AP | mean AUC | min AP |",
        "|---|---:|---:|---:|",
    ]
    for r in macro.itertuples(index=False):
        lines.append(f"| `{r.method}` | {r.mean_ap:.4f} | {r.mean_auc:.4f} | {r.min_ap:.4f} |")

    lines.extend(
        [
            "",
            "## Key correlations",
            "",
            "| pair | correlation |",
            "|---|---:|",
            f"| D3 framefolder raw vs cache D3-window l2 std | {corr.loc['d3_framefolder_raw', 'cache_d3win_l2_std']:.4f} |",
            f"| cache D3-window l2 std vs existing volatility raw | {corr.loc['cache_d3win_l2_std', 'existing_volatility_raw']:.4f} |",
            f"| D3 framefolder raw vs existing volatility raw | {corr.loc['d3_framefolder_raw', 'existing_volatility_raw']:.4f} |",
            "",
            "## Interpretation boundary",
            "",
            "- 如果 `cache_d3win_*` 接近 `existing_volatility_raw`，则 D3-DINOv3 低分主要来自 frame-folder/JPEG/crop/resize 预处理。",
            "- 如果 `cache_d3win_*` 接近 `d3_framefolder_raw`，则 D3 random start/window 本身是主要差异。",
            "- 如果二者都明显低于 `existing_volatility_raw`，则 window 与预处理都可能改变了 DINOv3 二阶信号。",
            "",
            "## 本轮判定",
            "",
            f"`cache_d3win_l2_std` 的 mean AP 为 `{cache_ap:.4f}`，接近 `d3_framefolder_raw` 的 `{d3_ap:.4f}`，但明显低于 `existing_volatility_raw` 的 `{existing_ap:.4f}`；同时 D3 frame-folder raw 与 cache D3-window l2 std 的相关性为 `{corr.loc['d3_framefolder_raw', 'cache_d3win_l2_std']:.4f}`。因此，DINOv3 分支上的低分主要来自 D3 random-start/window 策略，而不是 JPEG/crop/resize 预处理。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析 D3 window 在已有 full embedding cache 上的二阶变体。")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--pattern", default="*_head1000_DINOv3-L-local_l2_scores.csv")
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL_DIR)
    parser.add_argument("--index-csv", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--emb-cache", type=Path, default=DEFAULT_EMB_CACHE)
    parser.add_argument("--existing-volatility-csv", type=Path, default=DEFAULT_EXISTING_VOL)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_EVAL_DIR / "d3_window_cache_variant_diagnostic",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    d3 = load_d3_score_rows(args.eval_dir, args.pattern)
    index = load_index(args.index_csv)
    plan = load_plan(args.protocol_dir)
    existing = load_existing_vol(args.existing_volatility_csv)

    merged = (
        d3.merge(index, on=["subset", "source_model", "filename"], how="left", validate="many_to_one")
        .merge(
            plan[
                [
                    "subset",
                    "source_model",
                    "filename",
                    "d3_start_time",
                    "expected_d3_read_frames",
                ]
            ],
            on=["subset", "source_model", "filename"],
            how="left",
            validate="many_to_one",
        )
        .merge(existing, on=["subset", "source_model", "filename"], how="left", validate="many_to_one")
    )
    missing = merged[["fps", "downsample_idxs", "d3_start_time", "expected_d3_read_frames"]].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"有 {int(missing.sum())} 行缺少 index/plan 信息")

    rows = []
    for row in merged.itertuples(index=False):
        path = cache_path(args.emb_cache, row.subset, row.source_model, row.filename)
        status = "ok"
        error = ""
        try:
            emb = load_emb(path)
            downsample = json.loads(row.downsample_idxs)
            positions = d3_window_positions(
                downsample,
                float(row.fps),
                int(row.d3_start_time),
                int(row.expected_d3_read_frames),
            )
            if len(positions) < 3:
                raise ValueError(f"D3 window positions insufficient: {len(positions)}")
            scores = variant_scores(emb[positions])
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            scores = {k: np.nan for k in VARIANT_ORDER if k not in {"d3_framefolder_raw", "existing_volatility_raw"}}
        item = row._asdict()
        item.update(scores)
        item["d3_framefolder_raw"] = float(row.d3_second_order_std)
        item["status_cache_variant"] = status
        item["error_cache_variant"] = error
        rows.append(item)

    scored = pd.DataFrame(rows)
    if (scored["status_cache_variant"] != "ok").any():
        n_bad = int((scored["status_cache_variant"] != "ok").sum())
        print(f"warning: cache variant failed rows={n_bad}")
    per, macro = metrics(scored[scored["status_cache_variant"] == "ok"].copy())

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output_prefix.with_suffix(".scores.csv"), index=False)
    per.to_csv(args.output_prefix.with_suffix(".per_source.csv"), index=False)
    macro.to_csv(args.output_prefix.with_suffix(".macro.csv"), index=False)
    write_markdown(args.output_prefix.with_suffix(".md"), per, macro, scored)
    print(f"rows={len(scored)} ok={(scored['status_cache_variant'] == 'ok').sum()}")
    print(f"scores={args.output_prefix.with_suffix('.scores.csv')}")
    print(f"per_source={args.output_prefix.with_suffix('.per_source.csv')}")
    print(f"macro={args.output_prefix.with_suffix('.macro.csv')}")
    print(f"markdown={args.output_prefix.with_suffix('.md')}")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
