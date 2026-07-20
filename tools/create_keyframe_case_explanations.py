"""为 patch anomaly 案例补充原视频关键帧和人工解释表。

脚本读取 `selected_patch_cases.csv` 与 VideoFeedback index，抽取与最大局部
二阶时序异常相邻的原视频帧，并生成带关键帧、score、patch anomaly map
和 anomaly timeline 的复合图。自动生成的 observation 是审稿写作草稿；
最终论文若使用具体语义原因，仍建议作者人工复核视频。
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from create_patch_case_visualizations import (  # noqa: E402
    FastPatchScorer,
    compute_anomaly_fields,
)


def _resolve_video_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([root / path, root.parent / path])
        if path.parts and path.parts[0] == root.name:
            candidates.append(root.parent / path)
            candidates.append(root / Path(*path.parts[1:]))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"找不到视频文件: {raw_path}")


def _parse_indices(value: object) -> list[int]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [int(x) for x in value]
    text = str(value).strip()
    if not text:
        return []
    parsed = ast.literal_eval(text)
    return [int(x) for x in parsed]


def _read_frame(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"无法读取 {video_path} 的第 {frame_index} 帧")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _match_index(selected: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in selected.itertuples(index=False):
        subset = "fake" if row.subset == "annotated" else row.subset
        matches = index_df[
            (index_df["subset"] == row.subset)
            & (index_df["source_model"] == row.source_model)
            & (index_df["video_path"].astype(str).str.endswith(f"/{subset}/{row.source_model}/{row.filename}"))
        ]
        if matches.empty:
            matches = index_df[
                (index_df["subset"] == row.subset)
                & (index_df["source_model"] == row.source_model)
                & (index_df["video_path"].astype(str).str.endswith(f"/{row.filename}"))
            ]
        if matches.empty:
            raise RuntimeError(f"index 中找不到案例: {row.subset}/{row.source_model}/{row.filename}")
        item = row._asdict()
        for key, value in matches.iloc[0].to_dict().items():
            item[f"index_{key}"] = value
        rows.append(item)
    return pd.DataFrame(rows)


def _draft_observation(row: pd.Series, max_anomaly: float) -> tuple[str, str]:
    category = str(row["category"])
    if category == "generated_alpha_still_high":
        return (
            "残余难例",
            "该生成视频在 global 与 patch 分支下均保持较高真实百分位，说明局部时序异常不足以抵消整体真实感；可作为生成器高保真样本的上界案例。",
        )
    if category == "generated_score_increased_by_alpha":
        return (
            "负迁移案例",
            "patch 分支显著抬高生成视频分数，说明局部二阶时序证据在该样本上更接近真实校准分布；该例适合说明融合需要保留全局分支约束。",
        )
    if category == "generated_patch_global_conflict":
        return (
            "分支冲突",
            "global 分支判为明显异常而 patch 分支给出较高真实百分位，显示局部运动统计与全局外观证据不一致；该例用于说明 patch 证据的互补性和边界。",
        )
    if category == "real_alpha_low":
        return (
            "真实误伤",
            "真实视频在 global 与 patch 分支上均处于低百分位，通常应归入真实域偏移、低质量或低运动边界；若论文讨论具体视觉原因，应人工复核原片段。",
        )
    return (
        "解释候选",
        f"该样本最大 robust anomaly={max_anomaly:.3f}，适合作为人工复核候选。",
    )


def _plot_tag(tag: str) -> str:
    return {
        "残余难例": "residual hard case",
        "负迁移案例": "negative transfer",
        "分支冲突": "branch conflict",
        "真实误伤": "real false positive",
        "解释候选": "audit candidate",
    }.get(tag, tag)


def build_keyframe_cases(
    root: Path,
    selected_path: Path,
    index_path: Path,
    patch_params: Path,
    out_dir: Path,
    duration: int,
    score_device: str,
) -> tuple[pd.DataFrame, list[dict[str, np.ndarray]]]:
    selected = pd.read_csv(selected_path)
    index_df = pd.read_csv(index_path)
    selected = _match_index(selected, index_df)

    scorer = FastPatchScorer(str(patch_params), device=score_device)
    scorer.validate("same_grid_second_order")
    fields = compute_anomaly_fields(selected, scorer, "same_grid_second_order", patch_region_size=1)

    keyframes_dir = out_dir / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    enriched_rows = []
    for idx, (row, field) in enumerate(zip(selected.itertuples(index=False), fields)):
        temporal = field["temporal_anomaly"]
        max_step = int(np.argmax(temporal))
        window_idxs = _parse_indices(getattr(row, f"index_{duration}_sec_idxs"))
        if window_idxs:
            window_pos = min(max_step + 1, len(window_idxs) - 1)
            frame_index = int(window_idxs[window_pos])
        else:
            frame_index = int(float(getattr(row, "index_num_frames")) // 2)
        video_path = _resolve_video_path(root, str(getattr(row, "index_video_path")))
        frame = _read_frame(video_path, frame_index)
        image_name = f"{idx+1:02d}_{row.source_model}_{Path(row.filename).stem}_frame{frame_index}.png".replace("/", "_")
        image_path = keyframes_dir / image_name
        plt.imsave(image_path, frame)
        tag, observation = _draft_observation(pd.Series(row._asdict()), float(temporal[max_step]))
        item = row._asdict()
        item.update(
            {
                "resolved_video_path": str(video_path),
                "keyframe_index": frame_index,
                "keyframe_time_sec": frame_index / float(getattr(row, "index_fps")),
                "max_anomaly_step": max_step,
                "max_robust_anomaly": float(temporal[max_step]),
                "keyframe_png": str(image_path.relative_to(root)),
                "interpretation_tag": tag,
                "plot_interpretation_tag": _plot_tag(tag),
                "draft_observation": observation,
            }
        )
        enriched_rows.append(item)
    enriched = pd.DataFrame(enriched_rows)
    enriched.to_csv(out_dir / "keyframe_case_explanations.csv", index=False)
    return enriched, fields


def plot_keyframe_cases(enriched: pd.DataFrame, fields: list[dict[str, np.ndarray]], out_stem: Path) -> None:
    n = len(enriched)
    fig, axes = plt.subplots(
        n,
        4,
        figsize=(13.5, 1.9 * n),
        gridspec_kw={"width_ratios": [1.35, 1.0, 1.0, 1.35]},
        constrained_layout=True,
    )
    if n == 1:
        axes = axes[None, :]

    for i, (row, field) in enumerate(zip(enriched.itertuples(index=False), fields)):
        ax_frame, ax_bar, ax_map, ax_curve = axes[i]
        frame = plt.imread(ROOT / row.keyframe_png)
        ax_frame.imshow(frame)
        ax_frame.set_title(f"{row.case_label}\nframe {row.keyframe_index} ({row.keyframe_time_sec:.2f}s)", fontsize=8)
        ax_frame.set_xticks([])
        ax_frame.set_yticks([])

        scores = [row.global_score, row.patch_score, row.alpha_score]
        ax_bar.barh(["global", "patch", "alpha"], scores, color=["#4C78A8", "#F58518", "#54A24B"])
        ax_bar.set_xlim(0, 1)
        ax_bar.set_title(row.plot_interpretation_tag, fontsize=8)
        ax_bar.tick_params(axis="both", labelsize=7)
        ax_bar.grid(axis="x", alpha=0.2)

        im = ax_map.imshow(field["spatial_anomaly"], cmap="magma", vmin=0, vmax=1)
        ax_map.set_title("patch anomaly map", fontsize=8)
        ax_map.set_xticks([])
        ax_map.set_yticks([])

        temporal = field["temporal_anomaly"]
        x = np.arange(len(temporal))
        ax_curve.plot(x, temporal, color="#B279A2", lw=1.6)
        ax_curve.scatter([row.max_anomaly_step], [row.max_robust_anomaly], color="#D62728", s=18, zorder=3)
        ax_curve.set_ylim(0, 1.02)
        ax_curve.set_xlabel("2nd-order step", fontsize=7)
        ax_curve.set_ylabel("top-10% anomaly", fontsize=7)
        ax_curve.tick_params(axis="both", labelsize=7)
        ax_curve.grid(alpha=0.2)

    fig.colorbar(im, ax=axes[:, 2], fraction=0.025, pad=0.02, label="robust anomaly")
    fig.suptitle("VideoFeedback failure/boundary cases with raw keyframes", fontsize=12)
    fig.savefig(out_stem.with_suffix(".svg"))
    fig.savefig(out_stem.with_suffix(".png"), dpi=240)
    plt.close(fig)


def write_summary(enriched: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        "# 原视频关键帧解释案例",
        "",
        "本目录在已有 patch anomaly case visualization 基础上补充原视频关键帧。关键帧选择规则为：",
        "先在同网格二阶 patch temporal anomaly 曲线中找到 top-10% patch anomaly 最高的时间步，",
        "再抽取对应二阶差分中心帧。该选择只用于事后解释，不参与训练、推理或调参。",
        "",
        "| case | source/file | keyframe | global | patch | alpha | 解释标签 | 写作观察 |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in enriched.itertuples(index=False):
        lines.append(
            f"| {row.case_label} | {row.subset}/{row.source_model}/`{row.filename}` | "
            f"{row.keyframe_index} ({row.keyframe_time_sec:.2f}s) | "
            f"{row.global_score:.4f} | {row.patch_score:.4f} | {row.alpha_score:.4f} | "
            f"{row.interpretation_tag} | {row.draft_observation} |"
        )
    lines.extend(
        [
            "",
            "## 使用边界",
            "",
            "- 图和表可以支撑“如何选择 failure / boundary cases”的可复现说明。",
            "- 当前 `draft_observation` 是基于分数关系和关键帧的审稿写作草稿；若主文声称具体语义原因，例如低运动、压缩伪影或文本-视频错配，应由作者人工观看原视频后确认。",
            "- 该图优先放入补充材料；主文若空间有限，可只保留 2–3 个代表案例。",
        ]
    )
    (out_dir / "keyframe_case_explanations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--selected",
        type=Path,
        default=ROOT / "results/journal_experiments/case_visualizations/selected_patch_cases.csv",
    )
    parser.add_argument("--index", type=Path, default=ROOT / "cache/indexes/videofeedback.csv")
    parser.add_argument(
        "--patch-params",
        type=Path,
        default=ROOT / "precomputed/patch_params_videofeedback_real_same_grid_second_order_region1_mean_v2.npz",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results/journal_experiments/keyframe_case_explanations",
    )
    parser.add_argument("--duration", type=int, default=2)
    parser.add_argument("--score-device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    enriched, fields = build_keyframe_cases(
        root=args.root,
        selected_path=args.selected,
        index_path=args.index,
        patch_params=args.patch_params,
        out_dir=args.out_dir,
        duration=args.duration,
        score_device=args.score_device,
    )
    plot_keyframe_cases(enriched, fields, args.out_dir / "keyframe_patch_anomaly_cases")
    write_summary(enriched, args.out_dir)
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
