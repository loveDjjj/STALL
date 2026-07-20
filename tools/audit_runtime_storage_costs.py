"""审计 Alpha-STALLED 当前可追溯的存储与运行代价证据。

该脚本只读取本地文件系统和已有 run log，不运行模型、不提取特征。
输出用于期刊稿的 cost / practicality 小节：

1. cache、precomputed、results 等目录的文件数和字节数；
2. 已有补充实验 log 中可解析的 tqdm elapsed time；
3. 哪些 runtime 指标当前缺少可信实测，不能在论文中直接声称。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


STORAGE_TARGETS = (
    ("index_csv", "cache/indexes"),
    ("global_embedding_cache_comgenvid", "cache/embeddings/comgenvid"),
    ("global_embedding_cache_videofeedback", "cache/embeddings/videofeedback"),
    ("global_embedding_cache_genvideo", "cache/embeddings/genvideo"),
    ("compact_patch_cache_comgenvid", "cache/patch_embeddings/comgenvid"),
    ("compact_patch_cache_videofeedback", "cache/patch_embeddings/videofeedback"),
    ("compact_patch_cache_genvideo", "cache/patch_embeddings/genvideo"),
    ("precomputed_calibration_params", "precomputed"),
    ("paper_scores", "results/paper_scores"),
    ("paper_tables", "results/paper_tables"),
    ("paper_sensitivity", "results/paper_sensitivity"),
    ("paper_sweeps", "results/paper_sweeps"),
    ("journal_experiments", "results/journal_experiments"),
    ("paper_figures", "results/paper_figures"),
)

LOG_GLOBS = (
    "results/journal_experiments/*/*.log",
    "results/journal_experiments/**/*run.log",
)

ELAPSED_RE = re.compile(r"100%\|[^\[]*\[(?P<elapsed>\d{2}:\d{2}(?::\d{2})?)<00:00")


def _iter_files(path: Path):
    if path.is_file():
        yield path
        return
    if not path.exists():
        return
    for child in path.rglob("*"):
        if child.is_file() and not child.is_symlink():
            yield child


def _size_row(root: Path, name: str, rel: str) -> dict:
    path = root / rel
    files = list(_iter_files(path))
    total = sum(f.stat().st_size for f in files)
    return {
        "name": name,
        "path": rel,
        "exists": path.exists(),
        "n_files": len(files),
        "bytes": total,
        "size_mib": total / (1024**2),
        "size_gib": total / (1024**3),
    }


def build_storage_audit(root: Path) -> pd.DataFrame:
    rows = [_size_row(root, name, rel) for name, rel in STORAGE_TARGETS]
    return pd.DataFrame(rows)


def _last_nonempty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped[:240]
    return ""


def build_runtime_log_audit(root: Path) -> pd.DataFrame:
    paths: set[Path] = set()
    for pattern in LOG_GLOBS:
        paths.update(root.glob(pattern))
    rows = []
    for path in sorted(paths):
        text = path.read_text(encoding="utf-8", errors="replace")
        elapsed = [m.group("elapsed") for m in ELAPSED_RE.finditer(text)]
        rows.append(
            {
                "log_path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "has_tqdm_elapsed": bool(elapsed),
                "last_tqdm_elapsed": elapsed[-1] if elapsed else "",
                "last_nonempty_line": _last_nonempty_line(text),
            }
        )
    return pd.DataFrame(rows)


def write_markdown(storage: pd.DataFrame, logs: pd.DataFrame, out_path: Path) -> None:
    video_stage_csv = (
        out_path.parent.parent
        / "video_stage_runtime_benchmark"
        / "video_stage_runtime_benchmark.csv"
    )
    video_stage_md = video_stage_csv.with_suffix(".md")
    video_stage = pd.read_csv(video_stage_csv) if video_stage_csv.exists() else pd.DataFrame()
    lines = [
        "# Runtime / storage 代价审计",
        "",
        "本审计读取本地文件系统、已有日志和已生成的固定 benchmark 结果。",
        "目录体积是当前状态的直接证据；运行时间分为历史补充实验日志、CSV-stage benchmark 和小样本 clean-cache video-stage benchmark。",
        "",
        "## 存储规模",
        "",
        "| item | path | files | size MiB | size GiB |",
        "|---|---|---:|---:|---:|",
    ]
    for row in storage.itertuples(index=False):
        lines.append(
            f"| {row.name} | `{row.path}` | {row.n_files} | {row.size_mib:.2f} | {row.size_gib:.3f} |"
        )

    patch = storage[storage["name"].str.startswith("compact_patch_cache")]
    global_cache = storage[storage["name"].str.startswith("global_embedding_cache")]
    results = storage[storage["path"].str.startswith("results/")]
    lines.extend(
        [
            "",
            "## 汇总判断",
            "",
            f"- compact patch cache 当前合计 {patch['size_gib'].sum():.2f} GiB，"
            f"global embedding cache 当前合计 {global_cache['size_gib'].sum():.2f} GiB。",
            f"- `results/` 下当前论文资产合计 {results['size_mib'].sum():.2f} MiB；"
            "提交范围仍应以 `.gitignore` 与 `git status` 为准，优先提交 CSV/Markdown/SVG/必要 PNG。",
            "- cache、precomputed 参数和新增运行日志不应提交。",
            "- duration/window sweep 若扩展到 1s/3s/4s，主要新增成本会落在 compact patch cache prefill，而不是 CSV 级融合分析。",
            "- video-stage runtime benchmark 已补充原始视频解码、DINOv3 embedding、patch prefill 和 patch cache scoring 的代表性小样本 clean-cache 计时。",
            "",
            "## 已有日志中可解析的运行片段",
            "",
            "| log | parsed tqdm elapsed | last line |",
            "|---|---:|---|",
        ]
    )
    if logs.empty:
        lines.append("| n/a | n/a | 当前没有可解析日志 |")
    else:
        for row in logs.itertuples(index=False):
            elapsed = row.last_tqdm_elapsed or "not parsed"
            lines.append(f"| `{row.log_path}` | {elapsed} | {row.last_nonempty_line} |")

    lines.extend(
        [
            "",
            "## 固定 video-stage benchmark",
            "",
        ]
    )
    if video_stage.empty:
        lines.append("当前尚未生成 `results/journal_experiments/video_stage_runtime_benchmark/`。")
    else:
        lines.extend(
            [
                f"结果文件：`{video_stage_md.relative_to(out_path.parents[2])}`。",
                "",
                "| stage | elapsed sec | cache files | cache MiB | score rows |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in video_stage.itertuples(index=False):
            lines.append(
                f"| `{row.stage}` | {row.elapsed_sec:.3f} | "
                f"{row.cache_files_after_stage} | {row.cache_size_mib_after_stage:.3f} | "
                f"{row.score_rows} |"
            )

    lines.extend(
        [
            "",
            "## 尚未覆盖的 runtime 边界",
            "",
            "- 尚未清空三数据集全部 cache 后重跑全量端到端总耗时；这会产生大量重复计算和 I/O，不建议作为默认补充任务。",
            "- 当前可以报告 storage footprint、CSV-stage 后处理成本、代表性 video-stage clean-cache 阶段成本，以及已有全量补充实验日志片段。",
            "",
            "论文中应明确区分小样本阶段成本、已有全量实验片段耗时和全数据集总耗时，避免把小样本 clean-cache benchmark 线性外推为完整 benchmark。",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results/journal_experiments/runtime_storage_audit",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    storage = build_storage_audit(root)
    logs = build_runtime_log_audit(root)
    storage.to_csv(out_dir / "storage_audit.csv", index=False)
    logs.to_csv(out_dir / "runtime_log_audit.csv", index=False)
    write_markdown(storage, logs, out_dir / "runtime_storage_audit.md")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
