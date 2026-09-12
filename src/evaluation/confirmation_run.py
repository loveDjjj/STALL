"""新参考与编码对照调度；按完整域提前出表，不重复启动已有任务。"""

import argparse, ctypes, fcntl, json, os, select, subprocess, sys, time
from pathlib import Path
from artifacts import paper_json
from evaluation.confirmation_data import context


def wait_pid(pid):
    if not Path(f"/proc/{pid}").exists():
        return
    if hasattr(os, "pidfd_open"):
        fd = os.pidfd_open(pid)
    else:
        libc = ctypes.CDLL(None, use_errno=True)
        fd = libc.pidfd_open(pid, 0) if hasattr(libc, "pidfd_open") else libc.syscall(434, pid, 0)
        if fd < 0:
            raise OSError(ctypes.get_errno(), "不能绑定已有任务")
    try:
        while not select.select([fd], [], [], 30)[0]:
            pass
    finally:
        os.close(fd)


def run(root, wait_fit=()):
    c, out, spec = context(root)
    lock = (out / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def stage(module, arguments, parallel=False, preview=False):
        cmds = [[sys.executable, "-m", module, *arguments]]
        if parallel:
            cmds = [cmds[0] + ["--rank", str(i), "--world", "2"] for i in (0, 1)]
        ps = [subprocess.Popen(cmd, cwd=root) for cmd in cmds]
        paper_json(
            out / "status.json",
            dict(
                status="running",
                stage=module + " " + " ".join(arguments),
                pid=os.getpid(),
                children=[p.pid for p in ps],
            ),
        )
        last = time.monotonic()
        try:
            while any(p.poll() is None for p in ps):
                if any(p.poll() not in (None, 0) for p in ps):
                    raise RuntimeError("分片失败，终止同阶段其它分片")
                if preview and time.monotonic() - last >= 60:
                    from evaluation.confirmation_results import export

                    export(root, partial=True)
                    last = time.monotonic()
                time.sleep(2)
            if any(p.returncode for p in ps):
                raise RuntimeError("阶段退出失败")
        finally:
            for p in ps:
                if p.poll() is None:
                    p.terminate()
            for p in ps:
                p.wait()

    try:
        if wait_fit:
            paper_json(
                out / "status.json",
                dict(status="waiting_existing_fit", pid=os.getpid(), children=list(wait_fit)),
            )
            for pid in wait_fit:
                wait_pid(pid)
        stage("evaluation.confirmation_data", [])
        stage("evaluation.confirmation_fit", ["features"], parallel=True)
        stage("evaluation.confirmation_fit", ["fit"])
        stage("evaluation.confirmation_engine", ["cdf", "--limit", "1"], parallel=True)
        stage("evaluation.confirmation_engine", ["evaluation", "--limit", "1"])
        stage("evaluation.confirmation_engine", ["cdf"], parallel=True)
        stage("evaluation.confirmation_engine", ["evaluation"], preview=True)
        stage("evaluation.confirmation_results", ["export"])
        stage("evaluation.confirmation_results", ["analyze"])
        paper_json(
            out / "status.json",
            dict(
                status="reference_complete",
                pid=os.getpid(),
                next="encoding contrast and independent audit",
            ),
        )
    except BaseException as e:
        paper_json(out / "status.json", dict(status="failed", pid=os.getpid(), error=str(e)))
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wait-fit", type=int, nargs="*", default=[])
    a = p.parse_args()
    run(Path.cwd(), a.wait_fit)
