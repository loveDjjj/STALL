"""固定行号分片、spawn工作进程失败回收及父进程不隐式补算。"""

import json
import multiprocessing
from pathlib import Path
import sys
import time

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from artifacts import paper_json
from config import load_paper_config
from execution import build_jobs, devices_for, row_numbers, run_workers
from workflow import score_manifest_checkpointed


def example_worker(job, report):
    paper_json(Path(job["output"]) / f"{job['name']}.json", dict(started=True))
    if job.get("fail"):
        raise RuntimeError("测试工作进程失败")
    time.sleep(job.get("delay", 0.0))
    report("fixture", job["total"], job["total"], job["name"])


def test_shard_identity_and_config():
    frame = pd.DataFrame(dict(video_id=list("abcde")))
    config = load_paper_config(ROOT / "configs/paper.yaml", ["runtime.devices=[cuda:0,cuda:1]"])
    jobs = build_jobs(frame, config, dict(identity="fixed"))
    assert jobs[0]["row_indices"] == [1, 3, 5] and jobs[1]["row_indices"] == [2, 4]
    assert jobs[0]["frame"].video_id.tolist() == list("ace")
    assert jobs[1]["config"]["runtime"]["device"] == "cuda:1"
    assert jobs[1]["config"]["runtime"]["devices"] == []
    assert devices_for(config) == ["cuda:0", "cuda:1"]
    for override in (
        "runtime.devices=[cuda:0,cuda:0]",
        "runtime.devices=[cpu,cuda:0]",
        "runtime.devices=[cuda:3]",
    ):
        with pytest.raises(ValueError):
            load_paper_config(ROOT / "configs/paper.yaml", [override])
    with pytest.raises(ValueError):
        row_numbers(frame, [1, 2])
    with pytest.raises(ValueError):
        row_numbers(frame, [1, 2, 3, 4, 4])


def test_workers_wait_for_exit_and_reap_failure(tmp_path):
    before = {p.pid for p in multiprocessing.active_children()}
    jobs = [dict(name=str(i), total=1, output=str(tmp_path), delay=0.1) for i in range(2)]
    progress = []
    run_workers(
        example_worker, jobs, tmp_path, "success", lambda stage, n, total, vid: progress.append(n)
    )
    assert progress[-1] == 2
    assert all(
        json.loads(p.read_text())["status"] == "completed"
        for p in (tmp_path / "workers/success").glob("*/status.json")
    )
    jobs[0]["fail"] = True
    jobs[1]["delay"] = 10.0
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="失败"):
        run_workers(example_worker, jobs, tmp_path, "failure")
    assert time.monotonic() - started < 10.0
    assert {p.pid for p in multiprocessing.active_children()} == before
    assert (tmp_path / "0.json").is_file()  # 已完成载荷没有被失败清理删除。


def test_missing_worker_checkpoint_is_not_silently_rescored(tmp_path):
    frame = pd.DataFrame([dict(video_id="a")])
    with pytest.raises(ValueError, match="缺失评分检查点"):
        score_manifest_checkpointed(
            tmp_path, {}, frame, None, "ref", tmp_path, "id", {}, require_cached=True
        )
