"""Benchmark Alpha-STALLED 的轻量 CSV-stage runtime。

该脚本不读取原始视频、不提取 DINOv3 特征、不重建 patch cache。
它只对当前 release 中可由 CSV 直接复现的阶段做固定命令计时：

1. global + patch 融合和主指标计算；
2. 单个 score CSV 的 pairwise balanced metrics；
3. alpha sweep；
4. 已有补充审计脚本。

长任务（global embedding、patch prefill、全量 patch eval）不在默认计时范围内，
但会在报告中列为 pending benchmark 项。
"""

from __future__ import annotations

import argparse
import platform
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd


DATASETS = ("comgenvid", "videofeedback", "genvideo")


def _run_text(cmd: list[str], cwd: Path, timeout: int = 20) -> str:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return proc.stdout.strip()
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"unavailable: {exc}"


def collect_environment(root: Path) -> str:
    lines = [
        "# Runtime benchmark environment",
        "",
        f"- platform: `{platform.platform()}`",
        f"- python: `{sys.version.split()[0]}`",
        f"- executable: `{sys.executable}`",
        "",
        "## conda",
        "",
        "```text",
        _run_text(["conda", "info", "--envs"], root, timeout=20),
        "```",
        "",
        "## GPU",
        "",
        "```text",
        _run_text(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            root,
            timeout=20,
        ),
        "```",
        "",
        "## key packages",
        "",
        "```text",
        _run_text(
            [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                "stall",
                "python",
                "-c",
                "import pandas, numpy, sklearn; "
                "print('pandas', pandas.__version__); "
                "print('numpy', numpy.__version__); "
                "print('sklearn', sklearn.__version__)",
            ],
            root,
            timeout=30,
        ),
        "```",
    ]
    return "\n".join(lines) + "\n"


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _timed_run(name: str, cmd: list[str], root: Path, timeout: int) -> dict:
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    elapsed = time.perf_counter() - started
    return {
        "stage": name,
        "command": _format_cmd(cmd),
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-8:]),
    }


def build_commands(root: Path, work_dir: Path, alpha_sweep: bool) -> list[tuple[str, list[str], int]]:
    py = ["conda", "run", "--no-capture-output", "-n", "stall", "python"]
    commands: list[tuple[str, list[str], int]] = []
    for dataset in DATASETS:
        commands.append(
            (
                f"fuse_main_{dataset}",
                py
                + [
                    "tools/eval_alpha_stalled.py",
                    "--global-csv",
                    f"results/paper_scores/{dataset}_global.csv",
                    "--patch-csv",
                    f"results/paper_scores/{dataset}_patch_second_order.csv",
                    "--patch-score-col",
                    "patch_final_score",
                    "--alpha",
                    "0.60",
                    "--output-csv",
                    str(work_dir / "tmp_scores" / f"{dataset}_alpha_stalled.csv"),
                    "--metrics-csv",
                    str(work_dir / "tmp_metrics" / f"{dataset}_alpha_stalled_metrics.csv"),
                ],
                120,
            )
        )
        for kind, score_col in (
            ("global", "final_score"),
            ("patch_second_order", "patch_final_score"),
            ("alpha_stalled", "final_score"),
        ):
            commands.append(
                (
                    f"metrics_{dataset}_{kind}",
                    py
                    + [
                        "tools/eval_score_csv.py",
                        "--csv",
                        f"results/paper_scores/{dataset}_{kind}.csv",
                        "--score-col",
                        score_col,
                        "--output-csv",
                        str(work_dir / "tmp_metrics" / f"{dataset}_{kind}_metrics.csv"),
                    ],
                    120,
                )
            )
        if alpha_sweep:
            commands.append(
                (
                    f"alpha_sweep_{dataset}",
                    py
                    + [
                        "tools/fuse_scores.py",
                        "--dataset",
                        dataset,
                        "--tag",
                        "runtime_benchmark",
                        "--global-csv",
                        f"results/paper_scores/{dataset}_global.csv",
                        "--patch-csv",
                        f"results/paper_scores/{dataset}_patch_second_order.csv",
                        "--patch-score-col",
                        "patch_final_score",
                        "--alphas",
                        "0:1:0.05",
                        "--output-dir",
                        str(work_dir / "tmp_sweeps" / f"{dataset}_alpha"),
                    ],
                    180,
                )
            )

    for script in (
        "tools/analyze_duration_window_feasibility.py",
        "tools/audit_runtime_storage_costs.py",
        "tools/audit_reference_experiment_alignment.py",
    ):
        commands.append(
            (
                f"run_{Path(script).stem}",
                py + [script, "--out-dir", str(work_dir / "tmp_audits" / Path(script).stem)],
                120,
            )
        )
    return commands


def write_markdown(results: pd.DataFrame, env_path: Path, out_path: Path) -> None:
    ok = results[results["returncode"] == 0].copy()
    lines = [
        "# CSV-stage runtime benchmark",
        "",
        "本 benchmark 只覆盖不需要原始视频、不重提 DINOv3 特征、不重建 patch cache 的阶段。",
        "因此它不能替代完整端到端 runtime 表；它用于量化论文资产重建、融合和 CSV 级分析的实际成本。",
        "",
        f"环境记录见 `{env_path.name}`。",
        "",
        "## Timed commands",
        "",
        "| stage | elapsed sec | return code |",
        "|---|---:|---:|",
    ]
    for row in results.itertuples(index=False):
        lines.append(f"| `{row.stage}` | {row.elapsed_sec:.3f} | {row.returncode} |")

    lines.extend(
        [
            "",
            "## Summary",
            "",
        ]
    )
    if ok.empty:
        lines.append("- No successful commands.")
    else:
        lines.append(f"- successful commands: {len(ok)} / {len(results)}")
        lines.append(f"- total successful elapsed time: {ok['elapsed_sec'].sum():.3f} sec")
        fuse = ok[ok["stage"].str.startswith("fuse_main_")]
        metrics = ok[ok["stage"].str.startswith("metrics_")]
        sweeps = ok[ok["stage"].str.startswith("alpha_sweep_")]
        audits = ok[ok["stage"].str.startswith("run_")]
        if not fuse.empty:
            lines.append(f"- main fusion + metrics total: {fuse['elapsed_sec'].sum():.3f} sec")
        if not metrics.empty:
            lines.append(f"- score metrics total: {metrics['elapsed_sec'].sum():.3f} sec")
        if not sweeps.empty:
            lines.append(f"- alpha sweep total: {sweeps['elapsed_sec'].sum():.3f} sec")
        if not audits.empty:
            lines.append(f"- audit scripts total: {audits['elapsed_sec'].sum():.3f} sec")

    failed = results[results["returncode"] != 0]
    if not failed.empty:
        lines.extend(["", "## Failed command tails", ""])
        for row in failed.itertuples(index=False):
            lines.extend([f"### {row.stage}", "", "```text", row.stdout_tail, "```", ""])

    lines.extend(
        [
            "",
            "## Video-stage benchmark status",
            "",
            "| item | status | where to look |",
            "|---|---|---|",
            "| global embedding extraction | covered by representative clean-cache video-stage benchmark | `results/journal_experiments/video_stage_runtime_benchmark/video_stage_runtime_benchmark.md` |",
            "| compact patch cache prefill | covered by representative clean-cache video-stage benchmark | `results/journal_experiments/video_stage_runtime_benchmark/video_stage_runtime_benchmark.md` |",
            "| patch cached scoring | covered by representative clean-cache video-stage benchmark | `results/journal_experiments/video_stage_runtime_benchmark/video_stage_runtime_benchmark.md` |",
            "| full-dataset clean-cache total runtime | intentionally not run by this CSV-stage script | only run if reviewers require a full end-to-end wall-clock table |",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results/journal_experiments/runtime_benchmark",
    )
    parser.add_argument("--skip-alpha-sweep", action="store_true")
    parser.add_argument(
        "--keep-temporary-artifacts",
        action="store_true",
        help="保留中间 fused CSV / metrics / sweep 产物；默认使用 /tmp 并自动清理。",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    env_path = out_dir / "runtime_benchmark_environment.md"
    env_path.write_text(collect_environment(root), encoding="utf-8")

    if args.keep_temporary_artifacts:
        work_dir = out_dir / "temporary_artifacts"
        work_dir.mkdir(parents=True, exist_ok=True)
        temp_context = None
    else:
        temp_context = tempfile.TemporaryDirectory(prefix="alpha_stalled_runtime_", dir="/tmp")
        work_dir = Path(temp_context.name)

    rows = []
    try:
        for name, cmd, timeout in build_commands(root, work_dir, alpha_sweep=not args.skip_alpha_sweep):
            print(f"[benchmark] {name}")
            rows.append(_timed_run(name, cmd, root, timeout=timeout))
    finally:
        if temp_context is not None:
            temp_context.cleanup()
    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "csv_stage_runtime_benchmark.csv", index=False)
    write_markdown(results, env_path, out_dir / "csv_stage_runtime_benchmark.md")
    if (results["returncode"] != 0).any():
        raise SystemExit("some benchmark commands failed")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
