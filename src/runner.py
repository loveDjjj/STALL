"""Alpha STALL 唯一的配置驱动执行路径。"""

from __future__ import annotations

import contextlib
import json
import shlex
import signal
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from artifacts import (
    create_run_directory,
    run_directory,
    write_csv,
    write_progress,
    write_run_manifest,
)
from config import config_digest
from pipeline import build_bootstrap, run_from_cache
from evaluation.tables import build_metric_tables, normalize_scores


class RunTerminated(RuntimeError):
    """将可捕获的外部终止信号转换为可落盘的运行失败记录。"""


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class _Tee:
    """把 runner 的终端信息同时保留到运行目录日志。"""

    def __init__(self, terminal, log) -> None:
        self.terminal = terminal
        self.log = log

    def write(self, value: str) -> int:
        self.terminal.write(value)
        self.log.write(value)
        return len(value)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()


def run(
    repository_root: Path,
    run_name: str,
    config: dict,
    *,
    dry_run: bool,
    overwrite: bool,
    scores_csv: Path | None,
    command: list[str],
) -> Path:
    if dry_run:
        print("dry-run：配置验证通过；不会创建结果目录或写入任何文件。")
        print(f"运行名：{run_name}")
        print(f"数据集：{', '.join(config['data']['datasets'])}")
        print(f"设备：{config['runtime'].get('devices') or [config['runtime']['device']]}")
        return repository_root / "results" / "runs" / run_name

    output_dir = create_run_directory(repository_root, run_name, overwrite=overwrite)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    (output_dir / "command.txt").write_text(
        "命令：" + " ".join(shlex.quote(item) for item in command) + "\n"
        + "开始时间（UTC）：" + datetime.now(timezone.utc).isoformat() + "\n",
        encoding="utf-8",
    )
    write_run_manifest(output_dir, repository_root, run_name, config, status="running")
    progress = {"status": "running", "run_name": run_name, "current_dataset": None, "phase": "initializing"}
    write_progress(output_dir, progress)
    received_signal: int | None = None
    previous_handlers: dict[int, object] = {}

    def on_signal(signum, _frame) -> None:
        nonlocal received_signal
        received_signal = int(signum)
        raise RunTerminated(f"运行器收到外部信号 {signal.Signals(signum).name}")

    for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, on_signal)

    def report(event: dict) -> None:
        progress.update(event)
        progress["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_progress(output_dir, progress)
        message = event.get("message")
        if message:
            print(message, flush=True)

    with (logs_dir / "run.log").open("a", encoding="utf-8", buffering=1) as log, \
         contextlib.redirect_stdout(_Tee(sys.stdout, log)), \
         contextlib.redirect_stderr(_Tee(sys.stderr, log)):
        print("[命令] " + " ".join(shlex.quote(item) for item in command), flush=True)
        print(f"[开始] run={run_name}，数据集={','.join(config['data']['datasets'])}", flush=True)
        windows = None
        pipeline_metadata = None
        try:
            if scores_csv is None:
                windows, scores, pipeline_metadata = run_from_cache(repository_root, config, report=report)
            else:
                report({"phase": "loading_scores", "message": "[输入] 读取外部分数 CSV"})
                scores = normalize_scores(pd.read_csv(scores_csv))
                selected = config["data"]["datasets"]
                scores = scores[scores["dataset"].isin(selected)].copy()
                if scores.empty:
                    raise ValueError("分数 CSV 不包含当前配置数据集的任何记录")
            report({"phase": "metrics", "message": "[汇总] 计算数据集与生成器指标"})
            dataset_metrics, generator_metrics = build_metric_tables(scores, run_name)
            write_csv(output_dir, "video_scores.csv", scores)
            write_csv(output_dir, "dataset_metrics.csv", dataset_metrics)
            write_csv(output_dir, "generator_metrics.csv", generator_metrics)
            if windows is not None:
                write_csv(output_dir, "window_scores.csv", windows)
                report({"phase": "bootstrap", "message": "[汇总] 计算 paired bootstrap"})
                bootstrap = build_bootstrap(scores, config)
                if not bootstrap.empty:
                    write_csv(output_dir, "bootstrap_metrics.csv", bootstrap)
            progress.update({"status": "completed", "phase": "completed"})
            write_progress(output_dir, progress)
            write_run_manifest(output_dir, repository_root, run_name, config, status="completed", input_scores=str(scores_csv) if scores_csv is not None else None, pipeline_metadata=pipeline_metadata)
            print(f"[完成] 结果已写入 {output_dir}", flush=True)
        except BaseException as error:
            failure = {
                "exception_type": type(error).__name__,
                "message": str(error),
                "received_signal": signal.Signals(received_signal).name if received_signal else None,
                "occurred_at_utc": datetime.now(timezone.utc).isoformat(),
                "traceback": traceback.format_exc(),
            }
            _write_json(output_dir / "failure.json", failure)
            if received_signal:
                _write_json(output_dir / "termination.json", failure)
            print(f"[中断] {failure['exception_type']}: {failure['message']}", flush=True)
            progress.update({"status": "interrupted", "phase": "interrupted"})
            write_progress(output_dir, progress)
            write_run_manifest(output_dir, repository_root, run_name, config, status="interrupted", input_scores=str(scores_csv) if scores_csv is not None else None, pipeline_metadata=pipeline_metadata)
            raise
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
    return output_dir


def resume_bootstrap(
    repository_root: Path,
    run_name: str,
    config: dict,
    *,
    command: list[str],
) -> Path:
    """仅补齐中断运行的 bootstrap，绝不重新读取特征缓存或评分。"""

    output_dir = run_directory(repository_root, run_name)
    manifest_path = output_dir / "run_manifest.json"
    scores_path = output_dir / "video_scores.csv"
    if not manifest_path.is_file() or not scores_path.is_file():
        raise FileNotFoundError("补跑 bootstrap 需要已有 run manifest 和 video_scores.csv")
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    if previous.get("status") != "interrupted":
        raise ValueError("仅允许为状态为 interrupted 的运行补跑 bootstrap")
    if previous.get("config_hash") != config_digest(config):
        raise ValueError("当前配置与中断运行不一致，拒绝将不同实验的 bootstrap 混入同一结果目录")

    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    with (logs_dir / "run.log").open("a", encoding="utf-8", buffering=1) as log, \
         contextlib.redirect_stdout(_Tee(sys.stdout, log)), \
         contextlib.redirect_stderr(_Tee(sys.stderr, log)):
        print("[恢复] 仅补跑 paired bootstrap，不重读缓存或重新评分", flush=True)
        progress = {
            "status": "running", "run_name": run_name,
            "phase": "bootstrap", "message": "[恢复] 计算 paired bootstrap",
        }
        write_progress(output_dir, progress)
        try:
            scores = normalize_scores(pd.read_csv(scores_path))
            bootstrap = build_bootstrap(scores, config)
            if not bootstrap.empty:
                write_csv(output_dir, "bootstrap_metrics.csv", bootstrap)
            progress.update({"status": "completed", "phase": "completed"})
            write_progress(output_dir, progress)
            write_run_manifest(
                output_dir, repository_root, run_name, config, status="completed",
                pipeline_metadata=previous.get("pipeline"),
            )
            with (output_dir / "command.txt").open("a", encoding="utf-8") as command_log:
                command_log.write("恢复命令：" + " ".join(shlex.quote(item) for item in command) + "\n")
            print(f"[完成] bootstrap 已补齐：{output_dir}", flush=True)
        except BaseException:
            progress.update({"status": "interrupted", "phase": "interrupted"})
            write_progress(output_dir, progress)
            write_run_manifest(
                output_dir, repository_root, run_name, config, status="interrupted",
                pipeline_metadata=previous.get("pipeline"),
            )
            raise
    return output_dir
