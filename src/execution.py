"""固定视频分片的多GPU编排；子进程只产检查点，父进程验收并归并。"""

import copy
import ctypes
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import uuid

from artifacts import PaperLog, paper_json


def devices_for(config):
    return config["runtime"].get("devices") or [config["runtime"]["device"]]


def row_numbers(frame, indices=None):
    values = list(range(1, len(frame) + 1)) if indices is None else list(indices)
    if (
        len(values) != len(frame)
        or len(set(values)) != len(values)
        or any(type(i) is not int or i < 1 for i in values)
    ):
        raise ValueError("检查点行号须为互异正整数且与清单长度相同")
    return values


def build_jobs(frame, config, common):
    devices = devices_for(config)
    jobs = []
    for rank, device in enumerate(devices):
        positions = list(range(rank, len(frame), len(devices)))
        if not positions:
            continue
        worker_config = copy.deepcopy(config)
        worker_config["runtime"].update(device=device, devices=[])
        jobs.append(
            dict(
                common,
                frame=frame.iloc[positions].copy(),
                config=worker_config,
                row_indices=[i + 1 for i in positions],
                device=device,
                total=len(positions),
            )
        )
    return jobs


def _worker_entry(target, job, directory, ticket, parent_pid):
    # Linux父进程异常退出时终止自己的子进程；不让旧worker在新恢复期间继续写同一检查点。
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM) != 0:
            raise OSError(ctypes.get_errno(), "无法设置父进程退出信号")
        if os.getppid() != parent_pid:
            raise RuntimeError("父进程已退出")
    directory = Path(directory)
    with (directory / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with PaperLog(directory):
            total = job["total"]
            started = time.monotonic()

            def report(stage, completed, reported_total, video_id):
                if reported_total != total or not 0 <= completed <= total:
                    raise ValueError("worker进度越界")
                paper_json(
                    directory / "status.json",
                    dict(
                        status="running",
                        ticket=ticket,
                        pid=os.getpid(),
                        stage=stage,
                        completed=completed,
                        total=total,
                        video_id=video_id,
                        device=job.get("device"),
                        seconds=time.monotonic() - started,
                    ),
                )

            try:
                report("starting", 0, total, None)
                target(job, report)
                paper_json(
                    directory / "status.json",
                    dict(
                        status="completed",
                        ticket=ticket,
                        pid=os.getpid(),
                        completed=total,
                        total=total,
                        device=job.get("device"),
                        seconds=time.monotonic() - started,
                    ),
                )
            except BaseException as exc:
                print(traceback.format_exc(), file=sys.stderr)
                paper_json(
                    directory / "status.json",
                    dict(status="failed", ticket=ticket, pid=os.getpid(), error=str(exc)),
                )
                raise


def run_workers(target, jobs, directory, stage, progress=None):
    """无进程池隐式重试；每个已启动句柄必须回收，状态文件不替代exitcode。"""
    if not jobs:
        return
    directory = Path(directory) / "workers" / stage
    directory.mkdir(parents=True, exist_ok=True)
    context = multiprocessing.get_context("spawn")
    processes = []
    ticket = uuid.uuid4().hex
    total = sum(job["total"] for job in jobs)
    last_completed = -1
    last_report = 0.0
    try:
        for rank, job in enumerate(jobs):
            worker_dir = directory / f"rank_{rank}"
            worker_dir.mkdir(exist_ok=True)
            paper_json(
                worker_dir / "status.json",
                dict(status="preparing", ticket=ticket, completed=0, total=job["total"]),
            )
            process = context.Process(
                target=_worker_entry,
                args=(target, job, worker_dir, ticket, os.getpid()),
                daemon=True,
            )
            process.start()
            processes.append((process, worker_dir))
        while True:
            completed = 0
            for process, path in processes:
                if process.exitcode not in (None, 0):
                    raise RuntimeError(
                        f"{stage}工作进程{process.pid}失败，见{path}/logs/console.log"
                    )
                state = json.loads((path / "status.json").read_text())
                if state.get("ticket") != ticket:
                    raise ValueError("worker状态来自另一次调用")
                completed += state.get("completed", 0)
            now = time.monotonic()
            if progress and (completed != last_completed or now - last_report >= 10):
                progress(stage, completed, total, "workers")
                last_completed = completed
                last_report = now
            if all(process.exitcode is not None for process, _ in processes):
                break
            for process, _ in processes:
                process.join(timeout=0.1)
        for process, path in processes:
            state = json.loads((path / "status.json").read_text())
            if process.exitcode != 0 or state.get("status") != "completed":
                raise RuntimeError("worker退出但未正常完成")
    finally:
        for process, _ in processes:
            if process.is_alive():
                process.terminate()
        for process, _ in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()


def gpu_job(job, report):
    """spawn可导入入口；不加载/执行scripts，设备只由父进程分片计划决定。"""
    import torch

    torch.set_num_threads(4)
    root, config, frame, directory, identity = (
        job[k] for k in ("root", "config", "frame", "directory", "identity")
    )
    numbers = job["row_indices"]
    total = len(frame)
    if job["kind"] == "score":
        from workflow import score_manifest_checkpointed

        score_manifest_checkpointed(
            root,
            config,
            frame,
            job["reference"],
            job["reference_hash"],
            directory,
            identity,
            job["states"],
            lambda done, vid: report("score", done, total, vid),
            row_indices=numbers,
        )
    elif job["kind"] == "fit_features":
        from reference_fit import prepare_fit_assets

        order = {vid: i + 1 for i, vid in enumerate(frame.video_id)}
        prepare_fit_assets(
            root,
            config,
            frame,
            job["states"],
            directory,
            identity,
            lambda stage, done, n, vid: report(stage, order[vid], total, vid),
            row_indices=numbers,
            write_manifest=False,
        )
    elif job["kind"] == "cdf":
        from reference_fit import score_cdf_checkpointed, _load_gaussians

        params, _ = _load_gaussians(Path(directory) / "gaussians.npz", identity)
        score_cdf_checkpointed(
            root,
            config,
            frame,
            params,
            job["states"],
            directory,
            identity,
            progress=report,
            row_indices=numbers,
        )
    else:
        raise ValueError("未知GPU工作阶段")
