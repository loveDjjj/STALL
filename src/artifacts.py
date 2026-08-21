"""写入可复现实验产物，默认禁止覆盖已有结果。"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config import config_digest, dump_config


_RUN_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
REQUIRED_RUN_ARTIFACTS = (
    "resolved_config.yaml", "run_manifest.json", "video_scores.csv",
    "dataset_metrics.csv", "generator_metrics.csv", "pairwise_metrics.csv",
)
OPTIONAL_RUN_ARTIFACTS = (
    "window_scores.csv",
    "bootstrap_metrics.csv",
)


def run_directory(repository_root: Path, run_name: str) -> Path:
    """返回并校验一个运行结果目录。"""

    if not _RUN_NAME.fullmatch(run_name):
        raise ValueError("run 名称只能使用小写字母、数字、下划线或连字符")
    return repository_root / "results" / "runs" / run_name


def git_commit(repository_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def create_run_directory(repository_root: Path, run_name: str, *, overwrite: bool) -> Path:
    directory = run_directory(repository_root, run_name)
    if directory.exists() and not overwrite:
        raise FileExistsError(f"run 结果目录已存在：{directory}")
    if directory.exists() and overwrite:
        # --overwrite 表示开始一条全新的候选运行，不能让旧 CSV、进度或日志混入。
        for child in directory.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_run_manifest(
    output_dir: Path,
    repository_root: Path,
    run_name: str,
    config: dict[str, Any],
    *,
    status: str,
    input_scores: str | None = None,
    pipeline_metadata: dict[str, Any] | None = None,
) -> None:
    dump_config(output_dir / "resolved_config.yaml", config)
    manifest = {
        "run_name": run_name,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(repository_root),
        "config_hash": config_digest(config),
        "method": config["method"]["name"],
        "datasets": config["data"]["datasets"],
        "sampling": config["sampling"],
        "calibration": config["calibration"],
        "score_direction": config["metrics"]["score_direction"],
        "positive_class": config["metrics"]["positive_class"],
        "input_scores": input_scores,
        "pipeline": pipeline_metadata,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_csv(output_dir: Path, name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(output_dir / name, index=False)


def write_progress(output_dir: Path, payload: dict[str, Any]) -> None:
    """原子写入可观察的运行进度，避免监控端读到半个 JSON。"""

    destination = output_dir / "progress.json"
    temporary = destination.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def freeze_run(repository_root: Path, run_name: str, release_name: str) -> Path:
    """将完成的 run 复制为不可覆盖的冻结发布版本。"""

    source = run_directory(repository_root, run_name)
    manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("只能冻结状态为 completed 的 run")
    missing = [name for name in REQUIRED_RUN_ARTIFACTS if not (source / name).is_file()]
    if missing:
        raise ValueError(f"待冻结 run 缺少标准产物：{missing}")
    destination = repository_root / "release" / release_name
    if destination.exists():
        raise FileExistsError(f"发布目录已存在：{destination}")
    destination.mkdir(parents=True)
    for name in (*REQUIRED_RUN_ARTIFACTS, *OPTIONAL_RUN_ARTIFACTS):
        if not (source / name).is_file():
            continue
        shutil.copy2(source / name, destination / name)
    (destination / "release_manifest.json").write_text(
        json.dumps({
            "release_name": release_name, "source_run": run_name,
            "source_config_hash": manifest["config_hash"],
            "source_git_commit": manifest["git_commit"],
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination
