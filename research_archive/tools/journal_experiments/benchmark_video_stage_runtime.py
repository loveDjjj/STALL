#!/usr/bin/env python3
"""Benchmark representative video-stage runtime for Alpha-STALLED.

Unlike ``benchmark_csv_stage_runtime.py``, this script touches raw videos and
loads DINOv3.  It intentionally uses a small, fixed, clean-cache subset so that
the benchmark is reproducible and safe to keep in the release results:

1. global STALL compact embedding extraction + scoring;
2. patch compact cache prefill;
3. patch-only scoring from the freshly created patch cache.

The default uses the main ComGenVid 2s patch configuration so that the script
does not depend on ignored local calibration files.  The temporary embedding
caches are removed by default.  Only small score CSVs, stage timing tables and
environment notes are written under ``results/``.
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


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _run_text(cmd: list[str], cwd: Path, timeout: int = 30) -> str:
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


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return 1
    return sum(1 for child in path.rglob("*") if child.is_file())


def _read_score_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return len(pd.read_csv(path))
    except Exception:
        return 0


def _display_path(path: Path) -> str:
    repo_root = Path(__file__).resolve().parents[3]
    try:
        return str(path.resolve().relative_to(repo_root))
    except Exception:
        return str(path)


def collect_environment(repo_root: Path, project_root: Path, args: argparse.Namespace) -> str:
    gpu_query = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader",
    ]
    package_query = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        args.conda_env,
        "python",
        "-c",
        (
            "import torch, pandas, numpy; "
            "print('torch', torch.__version__); "
            "print('pandas', pandas.__version__); "
            "print('numpy', numpy.__version__); "
            "print('cuda_available', torch.cuda.is_available()); "
            "print('cuda_device_count', torch.cuda.device_count())"
        ),
    ]
    lines = [
        "# Video-stage runtime benchmark environment",
        "",
        f"- platform: `{platform.platform()}`",
        f"- python: `{sys.version.split()[0]}`",
        f"- executable: `{sys.executable}`",
        f"- repo root: `{repo_root}`",
        f"- subprocess cwd: `{project_root}`",
        f"- dataset: `{args.dataset}`",
        f"- duration_sec: `{args.duration}`",
        f"- debug_n per `(subset, source_model)`: `{args.debug_n}`",
        f"- num_workers: `{args.num_workers}`",
        f"- video_batch: `{args.video_batch}`",
        f"- frame_batch: `{args.frame_batch}`",
        f"- score_batch: `{args.score_batch}`",
        f"- device: `{args.device}`",
        "",
        "## GPU",
        "",
        "```text",
        _run_text(gpu_query, repo_root),
        "```",
        "",
        "## key packages",
        "",
        "```text",
        _run_text(package_query, repo_root, timeout=60),
        "```",
    ]
    return "\n".join(lines) + "\n"


def _timed_stage(
    stage: str,
    cmd: list[str],
    cwd: Path,
    timeout: int,
    cache_path: Path | None,
    score_path: Path | None,
) -> dict:
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    elapsed = time.perf_counter() - started
    cache_files = _count_files(cache_path) if cache_path is not None else 0
    cache_bytes = _dir_size_bytes(cache_path) if cache_path is not None else 0
    score_rows = _read_score_rows(score_path) if score_path is not None else 0
    return {
        "stage": stage,
        "command": _format_cmd(cmd),
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "cache_files_after_stage": cache_files,
        "cache_size_mib_after_stage": cache_bytes / (1024**2),
        "score_rows": score_rows,
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-12:]),
    }


def build_commands(
    repo_name: str,
    work_dir: Path,
    out_dir_in_repo: str,
    args: argparse.Namespace,
) -> list[dict]:
    py = ["conda", "run", "--no-capture-output", "-n", args.conda_env, "python"]
    dataset = args.dataset
    csv = f"{repo_name}/cache/indexes/{dataset}.csv"
    duration = str(args.duration)
    global_cache = work_dir / "global_embeddings" / dataset
    patch_cache = work_dir / "patch_embeddings" / dataset
    global_scores = (
        f"{out_dir_in_repo}/{dataset}_{args.duration}s_debug{args.debug_n}_global_scores.csv"
    )
    patch_prefill = (
        f"{out_dir_in_repo}/{dataset}_{args.duration}s_debug{args.debug_n}_patch_prefill.csv"
    )
    patch_scores = (
        f"{out_dir_in_repo}/{dataset}_{args.duration}s_debug{args.debug_n}_patch_scores.csv"
    )
    patch_params = (
        f"{repo_name}/precomputed/"
        "patch_params_comgenvid_real_second_order_region3_bottomk0p20_v2.npz"
    )
    global_params = f"{repo_name}/precomputed/stall_params_vatex_dino_v3.npz"

    return [
        {
            "stage": "global_compact_embedding_and_scoring",
            "cmd": py
            + [
                f"{repo_name}/src/eval.py",
                "--csv",
                csv,
                "--emb-cache",
                str(global_cache),
                "--params",
                global_params,
                "--duration",
                duration,
                "--compact",
                "--workers",
                str(args.num_workers),
                "--video-batch",
                str(args.video_batch),
                "--debug",
                str(args.debug_n),
                "--output-csv",
                global_scores,
            ],
            "timeout": args.timeout,
            "cache_path": global_cache,
            "score_path": Path(global_scores),
        },
        {
            "stage": "patch_compact_cache_prefill",
            "cmd": py
            + [
                f"{repo_name}/tools/prefill_patch_cache.py",
                "--csv",
                csv,
                "--patch-emb-cache",
                str(patch_cache),
                "--duration",
                duration,
                "--compact",
                "--debug-n",
                str(args.debug_n),
                "--num-workers",
                str(args.num_workers),
                "--video-batch",
                str(args.video_batch),
                "--frame-batch",
                str(args.frame_batch),
                "--device",
                args.device,
                "--execute",
                "--output-summary-csv",
                patch_prefill,
            ],
            "timeout": args.timeout,
            "cache_path": patch_cache,
            "score_path": Path(patch_prefill),
        },
        {
            "stage": "patch_cached_scoring",
            "cmd": py
            + [
                f"{repo_name}/src/eval_patch_fast.py",
                "--csv",
                csv,
                "--patch-emb-cache",
                str(patch_cache),
                "--patch-params",
                patch_params,
                "--output-csv",
                patch_scores,
                "--duration",
                duration,
                "--compact",
                "--debug-n",
                str(args.debug_n),
                "--score-device",
                args.device,
                "--score-batch-size",
                str(args.score_batch),
                "--patch-spat-weight",
                "0.10",
                "--patch-temp-weight",
                "0.90",
                "--patch-temp-mode",
                "same_grid_second_order",
                "--patch-region-size",
                "3",
                "--aggregation",
                "bottomk_mean",
                "--bottomk-ratio",
                "0.20",
            ],
            "timeout": args.timeout,
            "cache_path": patch_cache,
            "score_path": Path(patch_scores),
        },
    ]


def write_markdown(results: pd.DataFrame, env_path: Path, out_path: Path, args: argparse.Namespace) -> None:
    global_score_path = (
        args.output_dir / f"{args.dataset}_{args.duration}s_debug{args.debug_n}_global_scores.csv"
    )
    patch_score_path = (
        args.output_dir / f"{args.dataset}_{args.duration}s_debug{args.debug_n}_patch_scores.csv"
    )
    patch_prefill_path = (
        args.output_dir / f"{args.dataset}_{args.duration}s_debug{args.debug_n}_patch_prefill.csv"
    )
    lines = [
        "# Video-stage runtime benchmark",
        "",
        (
            "本 benchmark 使用原始视频和干净临时 cache，计时 Alpha-STALLED 中无法由 CSV "
            "直接复现的视频级阶段。默认设置为 ComGenVid 2s，每个 `(subset, source_model)` "
            f"取 {args.debug_n} 个视频，因此共 6 个视频；该设置用于端到端阶段成本审计，"
            "不作为性能指标。"
        ),
        "",
        f"环境记录见 `{env_path.name}`。",
        "",
        "## Timed stages",
        "",
        "| stage | elapsed sec | return code | cache files | cache MiB | score rows |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in results.itertuples(index=False):
        lines.append(
            f"| `{row.stage}` | {row.elapsed_sec:.3f} | {row.returncode} | "
            f"{row.cache_files_after_stage} | {row.cache_size_mib_after_stage:.3f} | "
            f"{row.score_rows} |"
        )

    ok = results[results["returncode"] == 0]
    lines.extend(["", "## Summary", ""])
    if ok.empty:
        lines.append("- No successful stages.")
    else:
        lines.append(f"- successful stages: {len(ok)} / {len(results)}")
        lines.append(f"- total successful elapsed time: {ok['elapsed_sec'].sum():.3f} sec")
        for row in ok.itertuples(index=False):
            if row.stage == "patch_compact_cache_prefill":
                denom = max(row.cache_files_after_stage, 1)
                unit = "cache file"
            else:
                denom = max(row.score_rows, 1)
                unit = "score row"
            lines.append(f"- `{row.stage}`: {row.elapsed_sec / denom:.3f} sec per {unit}")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "该结果补足 CSV-stage benchmark 不能覆盖的 raw-video 阶段：视频解码、DINOv3 "
                "embedding 写 cache、patch cache prefill 以及从 patch cache 读取后的快速评分。"
                "由于这是小样本 clean-cache benchmark，论文中应把它表述为代表性阶段成本，"
                "不要外推为全数据集总耗时。全数据集总耗时仍应结合已有日志和 cache/storage 审计说明。"
            ),
            "",
            "## Output files",
            "",
            f"- stage table: `{_display_path(out_path.with_suffix('.csv'))}`",
            f"- environment: `{_display_path(env_path)}`",
            (
                f"- score CSVs: `{_display_path(global_score_path)}`, "
                f"`{_display_path(patch_score_path)}`"
            ),
            f"- patch prefill summary: `{_display_path(patch_prefill_path)}`",
        ]
    )

    failed = results[results["returncode"] != 0]
    if not failed.empty:
        lines.extend(["", "## Failed command tails", ""])
        for row in failed.itertuples(index=False):
            lines.extend([f"### {row.stage}", "", "```text", row.stdout_tail, "```", ""])

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a small clean-cache video-stage runtime benchmark."
    )
    parser.add_argument("--dataset", default="comgenvid", choices=["comgenvid"])
    parser.add_argument("--duration", type=int, default=2, choices=[2])
    parser.add_argument("--debug-n", type=int, default=2)
    parser.add_argument("--conda-env", default="stall")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--video-batch", type=int, default=2)
    parser.add_argument("--frame-batch", type=int, default=16)
    parser.add_argument("--score-batch", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "results/journal_experiments/video_stage_runtime_benchmark",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Optional cache work directory. If omitted, a /tmp TemporaryDirectory is used and removed.",
    )
    parser.add_argument("--keep-work-dir", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    project_root = repo_root.parent
    repo_name = repo_root.name
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    env_path = out_dir / "video_stage_runtime_environment.md"
    env_path.write_text(collect_environment(repo_root, project_root, args), encoding="utf-8")

    temp_context: tempfile.TemporaryDirectory[str] | None = None
    if args.work_dir is None:
        temp_context = tempfile.TemporaryDirectory(prefix="alpha_stalled_video_runtime_", dir="/tmp")
        work_dir = Path(temp_context.name)
    else:
        work_dir = args.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)

    out_dir_in_repo = f"{repo_name}/{out_dir.relative_to(repo_root)}"
    commands = build_commands(repo_name, work_dir, out_dir_in_repo, args)

    records = []
    try:
        for spec in commands:
            records.append(
                _timed_stage(
                    stage=spec["stage"],
                    cmd=spec["cmd"],
                    cwd=project_root,
                    timeout=spec["timeout"],
                    cache_path=spec["cache_path"],
                    score_path=project_root / spec["score_path"],
                )
            )
    finally:
        if temp_context is not None and not args.keep_work_dir:
            temp_context.cleanup()

    results = pd.DataFrame(records)
    csv_path = out_dir / "video_stage_runtime_benchmark.csv"
    md_path = out_dir / "video_stage_runtime_benchmark.md"
    results.to_csv(csv_path, index=False)
    write_markdown(results, env_path, md_path, args)

    failures = results[results["returncode"] != 0]
    print(f"saved -> {out_dir}")
    print(results[["stage", "elapsed_sec", "returncode", "score_rows"]].to_string(index=False))
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
