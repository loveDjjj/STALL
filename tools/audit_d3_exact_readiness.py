#!/usr/bin/env python3
"""审计 D3-exact GenVideo 复现链路是否可运行。

该脚本不安装依赖、不下载模型、不抽帧、不跑 encoder。它只检查：

1. D3-exact CSV 协议文件是否存在、行数是否合理；
2. frame cache 中已抽取多少 real/fake smoke frame folders；
3. 当前 Python 环境是否能导入 D3-exact 需要的模块；
4. HuggingFace cache 中是否已有关键 encoder 权重；
5. 下一步应执行的最小命令。

目标是把 D3-exact 的阻塞点从口头描述变成机器可读审计结果。
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_DIR = (
    REPO_ROOT
    / "results/journal_experiments/global_second_order_volatility/genvideo/"
    / "d3_comparable_protocol/d3_exact_protocol"
)
DEFAULT_FRAME_ROOTS = [
    REPO_ROOT / "cache/d3_frames",
    REPO_ROOT.parent / "cache/d3_frames",
]

ENCODER_CACHE_HINTS = {
    "XCLIP-16": ["models--microsoft--xclip-base-patch16", "microsoft/xclip-base-patch16"],
    "XCLIP-32": ["models--microsoft--xclip-base-patch32", "microsoft/xclip-base-patch32"],
    "CLIP-16": ["models--openai--clip-vit-base-patch16", "openai/clip-vit-base-patch16"],
    "CLIP-32": ["models--openai--clip-vit-base-patch32", "openai/clip-vit-base-patch32"],
    "DINOv2-B": ["models--facebook--dinov2-base", "facebook/dinov2-base"],
    "DINOv2-L": ["models--facebook--dinov2-large", "facebook/dinov2-large"],
}

WEIGHT_SUFFIXES = (".bin", ".safetensors")


def cache_dir_loadable(path: Path) -> bool:
    """粗略判断 HuggingFace model cache 是否包含可离线加载的 snapshot。"""
    if not path.exists() or not path.is_dir():
        return False
    snapshot_roots = [path]
    snapshots_dir = path / "snapshots"
    if snapshots_dir.exists():
        snapshot_roots.extend([p for p in snapshots_dir.iterdir() if p.is_dir()])
    for root in snapshot_roots:
        has_config = (root / "config.json").exists()
        has_weight = any(p.name.endswith(WEIGHT_SUFFIXES) for p in root.rglob("*") if p.is_file() or p.is_symlink())
        if has_config and has_weight:
            return True
    return False


def module_status() -> pd.DataFrame:
    rows = []
    for module in ["torch", "torchvision", "transformers", "PIL", "sklearn", "cv2", "moviepy", "albumentations"]:
        spec = importlib.util.find_spec(module)
        rows.append({"module": module, "available": bool(spec), "origin": spec.origin if spec else ""})
    return pd.DataFrame(rows)


def hf_cache_roots() -> list[Path]:
    roots: list[Path] = []
    candidates = [
        Path.home() / ".cache/huggingface/hub",
        Path("/home/ubuntu/.cache/huggingface/hub"),
        Path("/tmp/huggingface/hub"),
    ]
    for path in candidates:
        if path.exists() and path not in roots:
            roots.append(path)
    return roots


def encoder_cache_status() -> pd.DataFrame:
    roots = hf_cache_roots()
    rows = []
    for encoder, hints in ENCODER_CACHE_HINTS.items():
        matches: list[str] = []
        loadable_matches: list[str] = []
        for root in roots:
            for hint in hints:
                direct = root / hint
                if direct.exists():
                    matches.append(str(direct))
                    if cache_dir_loadable(direct):
                        loadable_matches.append(str(direct))
            # HuggingFace hub uses models--org--name.
            for child in root.glob("models--*"):
                if any(hint.replace("/", "--") in child.name for hint in hints):
                    matches.append(str(child))
                    if cache_dir_loadable(child):
                        loadable_matches.append(str(child))
        rows.append(
            {
                "encoder": encoder,
                "cache_dir_present": bool(matches),
                "cache_present": bool(loadable_matches),
                "matches": ";".join(sorted(set(matches))),
                "loadable_matches": ";".join(sorted(set(loadable_matches))),
            }
        )
    return pd.DataFrame(rows)


def csv_status(protocol_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    csv_dir = protocol_dir / "csv"
    rows = []
    real_csv = csv_dir / "real_MSRVTT_head1000.csv"
    paths = [real_csv] + sorted(p for p in csv_dir.glob("*_head1000.csv") if p.name != real_csv.name)
    for path in paths:
        if not path.exists():
            rows.append({"csv": str(path), "exists": False, "rows": 0, "labels": "", "source_models": ""})
            continue
        df = pd.read_csv(path)
        rows.append(
            {
                "csv": str(path),
                "exists": True,
                "rows": int(len(df)),
                "labels": ",".join(str(x) for x in sorted(df["label"].unique())) if "label" in df else "",
                "source_models": ",".join(str(x) for x in sorted(df["source_model"].astype(str).unique())) if "source_model" in df else "",
            }
        )

    manifest_rows = []
    for name in [
        "d3_frame_extraction_runlist.csv",
        "d3_exact_video_plan.csv",
        "d3_exact_source_summary.csv",
    ]:
        path = protocol_dir / name
        if path.exists():
            df = pd.read_csv(path)
            manifest_rows.append({"file": str(path), "exists": True, "rows": int(len(df))})
        else:
            manifest_rows.append({"file": str(path), "exists": False, "rows": 0})
    return pd.DataFrame(rows), pd.DataFrame(manifest_rows)


def _frame_folders_for_row(content_path: str) -> tuple[bool, int]:
    path = Path(str(content_path))
    if not path.exists() or not path.is_dir():
        return False, 0
    n = sum(1 for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return True, n


def frame_cache_status(protocol_dir: Path, max_rows_per_csv: int | None = None) -> pd.DataFrame:
    csv_dir = protocol_dir / "csv"
    real_csv = csv_dir / "real_MSRVTT_head1000.csv"
    paths = [real_csv] + sorted(p for p in csv_dir.glob("*_head1000.csv") if p.name != real_csv.name)
    rows = []
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if max_rows_per_csv is not None:
            df = df.head(max_rows_per_csv)
        exists_count = 0
        eligible_count = 0
        frame_counts: list[int] = []
        for content_path in df.get("content_path", []):
            exists, n_frames = _frame_folders_for_row(str(content_path))
            exists_count += int(exists)
            eligible_count += int(n_frames >= 8)
            if exists:
                frame_counts.append(n_frames)
        rows.append(
            {
                "csv": path.name,
                "rows_checked": int(len(df)),
                "frame_folders_present": int(exists_count),
                "d3_read_eligible_ge8": int(eligible_count),
                "min_frames_present": min(frame_counts) if frame_counts else 0,
                "median_frames_present": float(pd.Series(frame_counts).median()) if frame_counts else 0.0,
                "max_frames_present": max(frame_counts) if frame_counts else 0,
            }
        )
    return pd.DataFrame(rows)


def readiness_summary(
    modules: pd.DataFrame,
    encoders: pd.DataFrame,
    csvs: pd.DataFrame,
    manifests: pd.DataFrame,
    frames: pd.DataFrame,
) -> pd.DataFrame:
    required_modules = {"torch", "torchvision", "transformers", "PIL", "sklearn"}
    module_ok = bool(
        modules[modules["module"].isin(required_modules)]["available"].all()
        and len(modules[modules["module"].isin(required_modules)]) == len(required_modules)
    )
    xclip_cached = bool(encoders[(encoders["encoder"] == "XCLIP-16") & (encoders["cache_present"])].shape[0])
    csv_ok = bool(csvs["exists"].all() and (csvs["rows"] > 0).all())
    manifests_ok = bool(manifests["exists"].all() and (manifests["rows"] > 0).all())
    smoke_ok = bool((frames["d3_read_eligible_ge8"] > 0).any()) if not frames.empty else False
    full_frames_ok = bool((frames["d3_read_eligible_ge8"] >= frames["rows_checked"]).all()) if not frames.empty else False

    rows = [
        {"item": "required_python_modules", "ok": module_ok, "detail": "torch/torchvision/transformers/PIL/sklearn"},
        {"item": "xclip_16_cache", "ok": xclip_cached, "detail": "microsoft/xclip-base-patch16"},
        {"item": "d3_csv_protocol", "ok": csv_ok, "detail": "real_MSRVTT_head1000 + 10 fake head1000 CSVs"},
        {"item": "d3_manifests", "ok": manifests_ok, "detail": "runlist/video_plan/source_summary"},
        {"item": "frame_smoke_present", "ok": smoke_ok, "detail": "至少已有部分 >=8 帧目录可跑 smoke"},
        {"item": "full_frame_cache_present", "ok": full_frames_ok, "detail": "所有 CSV 行均已有 >=8 帧目录"},
    ]
    return pd.DataFrame(rows)


def write_markdown(
    out_dir: Path,
    modules: pd.DataFrame,
    encoders: pd.DataFrame,
    csvs: pd.DataFrame,
    manifests: pd.DataFrame,
    frames: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    def yes_no(value: bool) -> str:
        return "是" if bool(value) else "否"

    lines = [
        "# D3-exact GenVideo readiness 审计",
        "",
        "本审计不安装依赖、不下载模型、不抽帧、不运行 encoder；只检查当前机器是否具备运行 D3-exact 的条件。",
        "",
        "## 总结",
        "",
        "| 项 | 是否就绪 | 说明 |",
        "|---|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"| `{row.item}` | {yes_no(row.ok)} | {row.detail} |")

    lines.extend(["", "## Python 模块", "", "| module | available | origin |", "|---|---:|---|"])
    for row in modules.itertuples(index=False):
        lines.append(f"| `{row.module}` | {yes_no(row.available)} | `{row.origin}` |")

    lines.extend(["", "## Encoder cache", "", "| encoder | cache dir | loadable cache | matches | loadable matches |", "|---|---:|---:|---|---|"])
    for row in encoders.itertuples(index=False):
        lines.append(
            f"| `{row.encoder}` | {yes_no(row.cache_dir_present)} | {yes_no(row.cache_present)} | "
            f"`{row.matches}` | `{row.loadable_matches}` |"
        )

    lines.extend(["", "## CSV 协议文件", "", "| csv | exists | rows | labels | source_models |", "|---|---:|---:|---|---|"])
    for row in csvs.itertuples(index=False):
        lines.append(f"| `{Path(row.csv).name}` | {yes_no(row.exists)} | {row.rows} | `{row.labels}` | `{row.source_models}` |")

    lines.extend(["", "## manifest 文件", "", "| file | exists | rows |", "|---|---:|---:|"])
    for row in manifests.itertuples(index=False):
        lines.append(f"| `{Path(row.file).name}` | {yes_no(row.exists)} | {row.rows} |")

    lines.extend(["", "## frame cache 覆盖", "", "| csv | rows checked | frame folders | >=8 frames | min/median/max frames |", "|---|---:|---:|---:|---:|"])
    for row in frames.itertuples(index=False):
        lines.append(
            f"| `{row.csv}` | {row.rows_checked} | {row.frame_folders_present} | "
            f"{row.d3_read_eligible_ge8} | {row.min_frames_present}/{row.median_frames_present:.1f}/{row.max_frames_present} |"
        )

    missing_modules = modules[(modules["module"].isin(["transformers"])) & (~modules["available"])]
    xclip_missing = encoders[(encoders["encoder"] == "XCLIP-16") & (~encoders["cache_present"])]
    clip16_cached = bool(encoders[(encoders["encoder"] == "CLIP-16") & (encoders["cache_present"])].shape[0])
    lines.extend(["", "## 下一步命令建议", ""])
    if not missing_modules.empty or not xclip_missing.empty:
        lines.extend(["当前不能运行 D3-exact XCLIP smoke test。需要先完成：", ""])
        steps = []
        if not missing_modules.empty:
            steps.append("安装 `transformers`")
        if not xclip_missing.empty:
            steps.append("下载或缓存 `microsoft/xclip-base-patch16`")
        steps.append("然后运行 XCLIP smoke evaluator")
        for i, step in enumerate(steps, start=1):
            lines.append(f"{i}. {step}；" if i < len(steps) else f"{i}. {step}。")
        if clip16_cached and not missing_modules.empty:
            lines.extend(
                [
                    "",
                    "补充：本机已有 `openai/clip-vit-base-patch16` cache。若先只安装 `transformers`，可以先用 `--encoder CLIP-16` 跑 evaluator 链路 smoke；这不能替代 D3 论文主结果的 XCLIP-B/16，只用于验证读取 frames、score 方向和 metrics 落盘。",
                ]
            )
    else:
        lines.extend(
            [
                "依赖与 XCLIP cache 已满足。可先运行 Crafter 20/20 smoke test：",
                "",
                "```bash",
                "conda run --no-capture-output -n stall python tools/eval_d3_exact_from_frames.py \\",
                "  --encoder XCLIP-16 --loss l2 \\",
                "  --real-csv results/journal_experiments/global_second_order_volatility/genvideo/d3_comparable_protocol/d3_exact_protocol/csv/real_MSRVTT_head1000.csv \\",
                "  --fake-csv results/journal_experiments/global_second_order_volatility/genvideo/d3_comparable_protocol/d3_exact_protocol/csv/Crafter_head1000.csv \\",
                "  --max-real 20 --max-fake 20",
                "```",
            ]
        )

    (out_dir / "d3_exact_readiness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PROTOCOL_DIR / "d3_exact_eval")
    parser.add_argument("--frame-check-head", type=int, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    modules = module_status()
    encoders = encoder_cache_status()
    csvs, manifests = csv_status(args.protocol_dir)
    frames = frame_cache_status(args.protocol_dir, max_rows_per_csv=args.frame_check_head)
    summary = readiness_summary(modules, encoders, csvs, manifests, frames)

    modules.to_csv(args.out_dir / "d3_exact_readiness_modules.csv", index=False)
    encoders.to_csv(args.out_dir / "d3_exact_readiness_encoder_cache.csv", index=False)
    csvs.to_csv(args.out_dir / "d3_exact_readiness_csvs.csv", index=False)
    manifests.to_csv(args.out_dir / "d3_exact_readiness_manifests.csv", index=False)
    frames.to_csv(args.out_dir / "d3_exact_readiness_frame_cache.csv", index=False)
    summary.to_csv(args.out_dir / "d3_exact_readiness_summary.csv", index=False)
    write_markdown(args.out_dir, modules, encoders, csvs, manifests, frames, summary)

    print(f"saved -> {args.out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    main()
