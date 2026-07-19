"""生成期刊补充用 patch anomaly case visualization。

本脚本不重新提取 DINOv3 特征，只读取已有 patch cache 和真实视频校准参数。
输出包含：

- selected_patch_cases.csv：固定规则选择的代表案例；
- patch_anomaly_case_summary.md：案例选择依据和可写入论文的观察；
- patch_anomaly_cases.svg/png：每个案例的分数条、空间 anomaly map 和时序 anomaly 曲线。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval_patch_fast import FastPatchScorer  # noqa: E402


@dataclass(frozen=True)
class SelectionRule:
    dataset: str
    subset: str
    source_model: str
    category: str
    label: str


DEFAULT_RULES = [
    SelectionRule(
        "videofeedback",
        "annotated",
        "Text2Video-Zero",
        "generated_alpha_still_high",
        "T2V-Zero / hard generated",
    ),
    SelectionRule(
        "videofeedback",
        "annotated",
        "Text2Video-Zero",
        "generated_score_increased_by_alpha",
        "T2V-Zero / patch raises score",
    ),
    SelectionRule(
        "videofeedback",
        "annotated",
        "VideoCrafter2",
        "generated_patch_global_conflict",
        "VideoCrafter2 / patch-global conflict",
    ),
    SelectionRule(
        "videofeedback",
        "annotated",
        "LaVie-base",
        "generated_patch_global_conflict",
        "LaVie-base / patch-global conflict",
    ),
    SelectionRule(
        "videofeedback",
        "annotated",
        "AnimateDiff",
        "generated_patch_global_conflict",
        "AnimateDiff / patch-global conflict",
    ),
    SelectionRule(
        "videofeedback",
        "real",
        "Panda70M",
        "real_alpha_low",
        "Panda70M / real false-positive",
    ),
]


def _case_cache_path(root: Path, row: pd.Series, duration: int) -> Path:
    stem = Path(str(row["filename"])).stem
    return root / row["subset"] / row["source_model"] / f"{stem}_{duration}s.pt"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def select_cases(candidates: pd.DataFrame, cache_root: Path, duration: int) -> pd.DataFrame:
    rows = []
    used: set[tuple[str, str, str, str]] = set()
    for rule in DEFAULT_RULES:
        group = candidates[
            (candidates["dataset"] == rule.dataset)
            & (candidates["subset"] == rule.subset)
            & (candidates["source_model"] == rule.source_model)
            & (candidates["category"] == rule.category)
        ].copy()
        if group.empty:
            continue
        for _, row in group.iterrows():
            key = (row["dataset"], row["subset"], row["source_model"], row["filename"])
            if key in used:
                continue
            cache_path = _case_cache_path(cache_root, row, duration)
            if not cache_path.exists():
                continue
            item = row.to_dict()
            item["case_label"] = rule.label
            item["cache_path"] = _rel(cache_path)
            item["duration_sec"] = duration
            rows.append(item)
            used.add(key)
            break
    if not rows:
        raise RuntimeError("未选出任何可视化案例：请检查 failure_case_candidates.csv 和 patch cache。")
    return pd.DataFrame(rows)


def _robust01(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, [5, 95])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


@torch.inference_mode()
def compute_anomaly_fields(
    selected: pd.DataFrame,
    scorer: FastPatchScorer,
    patch_temp_mode: str,
    patch_region_size: int,
) -> list[dict[str, np.ndarray]]:
    out = []
    device = scorer.device
    for row in selected.itertuples(index=False):
        cache_path = Path(row.cache_path)
        if not cache_path.is_absolute():
            cache_path = ROOT / cache_path
        payload = torch.load(cache_path, weights_only=True, map_location="cpu")
        patch = payload["patch"].float().unsqueeze(0).to(device)
        grid_size = tuple(int(x) for x in payload["grid_size"])
        if grid_size != scorer.patch_grid_size:
            raise ValueError(f"{cache_path}: cache grid={grid_size}, params grid={scorer.patch_grid_size}")
        temp = scorer.temporal_features(patch, patch_temp_mode, patch_region_size)
        _, _, mu_temp, W_temp = scorer._params_for_device(device)
        temp_white = torch.matmul(temp - mu_temp, W_temp)
        temp_ll = scorer.log_likelihood_from_white(temp_white)[0].detach().cpu().numpy()

        anomaly = -temp_ll
        gh, gw = scorer.patch_grid_size
        spatial = np.percentile(anomaly, 90, axis=0).reshape(gh, gw)
        temporal = np.sort(anomaly, axis=1)[:, -max(1, int(np.ceil(anomaly.shape[1] * 0.10))) :].mean(axis=1)
        out.append(
            {
                "spatial_anomaly": _robust01(spatial),
                "temporal_anomaly": _robust01(temporal),
                "raw_spatial_anomaly": spatial,
                "raw_temporal_anomaly": temporal,
            }
        )
    return out


def write_summary(selected: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        "# Patch anomaly case visualization",
        "",
        "本目录基于 `failure_case_candidates.csv` 选取代表案例，并从已有 patch cache 与真实视频校准参数生成 patch-level anomaly map。",
        "案例只用于事后解释，不参与推理、训练或超参数选择。",
        "",
        "| case | dataset | subset/source | filename | global | patch | alpha | 说明 |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"| {row.case_label} | {row.dataset} | {row.subset}/{row.source_model} | "
            f"`{row.filename}` | {row.global_score:.4f} | {row.patch_score:.4f} | "
            f"{row.alpha_score:.4f} | {row.analysis_note} |"
        )
    lines.extend(
        [
            "",
            "图中空间 map 为同网格二阶时序 likelihood 的高异常区域汇总：先取负 log-likelihood 作为 anomaly，",
            "再对时间维取 90 分位并做每视频 robust 归一化。时序曲线为每个时间步 top-10% patch anomaly 的均值。",
            "因此，颜色/曲线越高表示该视频在局部二阶时序上越偏离真实视频校准分布。",
            "",
            "当前选择覆盖 VideoFeedback 中 paired bootstrap 显示稳定负迁移的 Text2Video-Zero 与 VideoCrafter2，",
            "并加入 LaVie-base/AnimateDiff 的 patch-global conflict 以及 Panda70M 真实误伤样本，用于说明 patch 分支的收益边界。",
        ]
    )
    (out_dir / "patch_anomaly_case_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_cases(selected: pd.DataFrame, fields: list[dict[str, np.ndarray]], out_stem: Path) -> None:
    n = len(selected)
    fig, axes = plt.subplots(
        n,
        3,
        figsize=(10.5, 1.65 * n),
        gridspec_kw={"width_ratios": [1.15, 1.0, 1.25]},
        constrained_layout=True,
    )
    if n == 1:
        axes = axes[None, :]
    cmap = "magma"
    for idx, (row, field) in enumerate(zip(selected.itertuples(index=False), fields)):
        ax_bar, ax_map, ax_curve = axes[idx]
        scores = [row.global_score, row.patch_score, row.alpha_score]
        colors = ["#4C78A8", "#F58518", "#54A24B"]
        ax_bar.barh(["global", "patch", "alpha"], scores, color=colors)
        ax_bar.set_xlim(0, 1)
        ax_bar.set_title(row.case_label, fontsize=9)
        ax_bar.tick_params(axis="both", labelsize=8)
        ax_bar.grid(axis="x", alpha=0.2)

        im = ax_map.imshow(field["spatial_anomaly"], cmap=cmap, vmin=0, vmax=1)
        ax_map.set_title(f"{row.source_model}/{row.filename}", fontsize=8)
        ax_map.set_xticks([])
        ax_map.set_yticks([])

        temporal = field["temporal_anomaly"]
        ax_curve.plot(np.arange(len(temporal)), temporal, color="#B279A2", lw=1.8)
        ax_curve.fill_between(np.arange(len(temporal)), 0, temporal, color="#B279A2", alpha=0.18)
        ax_curve.set_ylim(0, 1.02)
        ax_curve.set_xlabel("second-order time step", fontsize=8)
        ax_curve.set_ylabel("top-10% anomaly", fontsize=8)
        ax_curve.tick_params(axis="both", labelsize=8)
        ax_curve.grid(alpha=0.2)

    fig.colorbar(im, ax=axes[:, 1], fraction=0.025, pad=0.02, label="robust anomaly")
    fig.suptitle("Representative patch-level anomaly maps on VideoFeedback", fontsize=12)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".svg"))
    fig.savefig(out_stem.with_suffix(".png"), dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / "results/paper_sensitivity/failure_case_candidates.csv",
    )
    parser.add_argument(
        "--patch-cache",
        type=Path,
        default=ROOT / "cache/patch_embeddings/videofeedback",
    )
    parser.add_argument(
        "--patch-params",
        type=Path,
        default=ROOT / "precomputed/patch_params_videofeedback_real_same_grid_second_order_region1_mean_v2.npz",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results/journal_experiments/case_visualizations",
    )
    parser.add_argument("--duration", type=int, default=2)
    parser.add_argument("--patch-temp-mode", default="same_grid_second_order")
    parser.add_argument("--patch-region-size", type=int, default=1)
    parser.add_argument("--score-device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    candidates = pd.read_csv(args.candidates)
    selected = select_cases(candidates, args.patch_cache, args.duration)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.out_dir / "selected_patch_cases.csv", index=False)

    scorer = FastPatchScorer(str(args.patch_params), device=args.score_device)
    scorer.validate(args.patch_temp_mode)
    fields = compute_anomaly_fields(selected, scorer, args.patch_temp_mode, args.patch_region_size)
    write_summary(selected, args.out_dir)
    plot_cases(selected, fields, args.out_dir / "patch_anomaly_cases")
    print(f"selected cases: {len(selected)}")
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
