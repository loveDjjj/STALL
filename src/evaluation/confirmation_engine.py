"""新参考评分：CDF共享提取，评价单读取流供双卡，轻量结果分块原子保存。"""

import argparse, json, multiprocessing as mp, queue, time, traceback
from pathlib import Path
import numpy as np, torch, yaml
from artifacts import paper_json
from config import config_digest
from data.prefetch import bounded_map, frame_reservation
from data.video import decode_bounded
from evaluation.confirmation_data import context, DOMAINS
from evaluation.direction_study import parameter
from features import AlphaStallFeatureExtractor
from branches.global_branch import score_global_raw
from math_utils import GaussianMeanCandidateScorerFloat64, l2_normalized_second_order
from reference import file_digest


class Chunks:
    """每32条提交一次，最多重算31条；避免每视频fsync争抢机械盘。"""

    def __init__(self, path, identity):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.records = {}
        self.pending = []
        self.next = 0
        for p in sorted(self.path.glob("*.json")):
            item = json.loads(p.read_text())
            if item["identity"] != identity or config_digest(item["records"]) != item["sha256"]:
                raise ValueError("结果分块身份或内容错误")
            for r in item["records"]:
                if r["key"] in self.records:
                    raise ValueError("分块重复视频")
                self.records[r["key"]] = r
            self.next = max(self.next, int(p.stem) + 1)

    def add(self, value):
        if value["key"] in self.records:
            raise ValueError("重复发布视频")
        self.records[value["key"]] = value
        self.pending.append(value)
        if len(self.pending) >= 32:
            self.flush()

    def flush(self):
        if not self.pending:
            return

        # 正无穷先规范化，否则JSON表示改变会导致payload hash不一致。
        def safe(x):
            if isinstance(x, dict):
                return {k: safe(v) for k, v in x.items()}
            if isinstance(x, list):
                return [safe(v) for v in x]
            if isinstance(x, float) and x == float("inf"):
                return "Infinity"
            return x

        rows = safe(self.pending)
        paper_json(
            self.path / f"{self.next:06d}.json",
            dict(identity=self.identity, records=rows, sha256=config_digest(rows)),
        )
        self.next += 1
        self.pending = []


def setup(root):
    c, out, data = context(root)
    models = {}
    for d in DOMAINS:
        for seed in c["seeds"]:
            p = out / "models" / f"{d}_s{seed}.npz"
            models[str(p.relative_to(root))] = file_digest(p)
        p = out / "models" / f"{d}_original_matched.npz"
        models[str(p.relative_to(root))] = file_digest(p)
    code = {
        p: file_digest(root / p)
        for p in (
            "src/evaluation/confirmation_engine.py",
            "src/math_utils.py",
            "src/features.py",
            "src/branches/global_branch.py",
            "src/reference.py",
        )
    }
    spec = dict(data=config_digest(data), models=models, code=code)
    p = out / "score_identity.json"
    if p.exists() and json.loads(p.read_text()) != spec:
        raise ValueError("评分模型或代码改变，拒绝混合恢复")
    if not p.exists():
        paper_json(p, spec)
    return c, out, config_digest(spec)


def globals_from_cache(root, j):
    p = root / "results/runs/mainline_experts/global" / (j["key"] + ".npz")
    with np.load(p, allow_pickle=False) as z:
        g = z["global_features"].copy()
        u = z["uniform_global"].copy()
    return g, (u if u.size else None)


class Scorer:
    def __init__(self, root, c, out, device):
        self.root = root
        self.c = c
        self.device = device
        self.models = {}
        self.groups = {}
        self.extractor = None
        self.center = parameter(root / "results/runs/paper_fit_source/gaussians.npz", "lt").mean
        for d in DOMAINS:
            orig = root / f"results/runs/paper_fit_{d}/gaussians.npz"
            m = dict(
                global_models={"anchor": (parameter(orig, "gs"), parameter(orig, "gt"))},
                local={"anchor": parameter(orig, "lt")},
                first={},
            )
            m["local"]["matched_anchor"] = parameter(
                out / "models" / f"{d}_original_matched.npz", "matched"
            )
            for seed in c["seeds"]:
                p = out / "models" / f"{d}_s{seed}.npz"
                tag = f"s{seed}"
                m["global_models"][tag] = (parameter(p, "gs"), parameter(p, "gt"))
                for name in ("d2", "source_equal", "matched"):
                    m["local"][tag + "_" + name] = parameter(p, name)
                m["first"][tag + "_d1"] = parameter(p, "d1")
            self.models[d] = m

    def score(self, j, g, p, ug=None):
        g = np.ascontiguousarray(g)
        p = np.ascontiguousarray(p)
        x = torch.from_numpy(p)
        units = dict(
            local=l2_normalized_second_order(x),
            first=torch.nn.functional.normalize(x[:, 1:] - x[:, :-1], dim=-1, eps=1e-12),
        )
        domains = tuple(sorted(j["targets"]))
        locals = {d: {} for d in domains}
        for rep, u in units.items():
            tag = (domains, rep)
            if tag not in self.groups:
                names = [(d, n) for d in domains for n in self.models[d][rep]]
                self.groups[tag] = (
                    names,
                    GaussianMeanCandidateScorerFloat64(
                        [self.models[d][rep][n] for d, n in names], self.center, self.device
                    ),
                )
            names, s = self.groups[tag]
            values = s.score(u)
            for i, (d, n) in enumerate(names):
                locals[d][n] = values[:, i].tolist()
        result = {}
        error = 0.0
        for d, meta in j["targets"].items():
            global_values = {}
            for name, (gs, gt) in self.models[d]["global_models"].items():
                v = score_global_raw(g, gs, gt, device=self.device)
                uniform = None
                if ug is not None:
                    uv = score_global_raw(ug, gs, gt, device=self.device)
                    uniform = dict(gs=float(uv.spatial[0]), gt=float(uv.temporal_t1[0]))
                global_values[name] = dict(
                    gs=v.spatial.tolist(), gt=v.temporal_t1.tolist(), uniform=uniform
                )
            expected = np.array(
                [[float(w[k]) for k in ("gs", "gt", "lt")] for w in meta["expected"]]
            )
            actual = np.stack(
                [global_values["anchor"]["gs"], global_values["anchor"]["gt"], locals[d]["anchor"]],
                axis=1,
            )
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8)
            valid = np.isfinite(expected)
            error = max(error, float(np.abs(actual[valid] - expected[valid]).max()))
            global_values["anchor"]["gs"] = expected[:, 0].tolist()
            global_values["anchor"]["gt"] = expected[:, 1].tolist()
            locals[d]["anchor"] = expected[:, 2].tolist()
            if ug is not None:
                a = global_values["anchor"]["uniform"]
                e = meta["uniform"]
                np.testing.assert_allclose(
                    [a["gs"], a["gt"]], [float(e["gs"]), float(e["gt"])], rtol=0, atol=1e-8
                )
                global_values["anchor"]["uniform"] = {k: float(e[k]) for k in ("gs", "gt")}
            result[d] = dict(global_values=global_values, local=locals[d])
        return dict(key=j["key"], targets=result, max_anchor_error=error)


def cdf(root, rank=0, world=2, limit=0):
    c, out, identity = setup(root)
    store = Chunks(out / f"cdf_rank{rank}", identity)
    jobs = [j for j in json.loads((root / c["plans"]).read_text())["jobs"] if j["role"] == "cdf"]
    jobs = jobs[rank::world]
    if limit:
        jobs = jobs[:limit]
    pending = [j for j in jobs if j["key"] not in store.records]
    if not pending:
        return
    scorer = Scorer(root, c, out, c["devices"][rank])
    e = yaml.safe_load((root / "configs/paper.yaml").read_text())["encoder"]
    if file_digest(root / e["weights"]) != e["weights_sha256"]:
        raise ValueError("DINO权重改变")
    extractor = AlphaStallFeatureExtractor(
        c["devices"][rank],
        dino_repo=str(root / e["repo"]),
        dino_weights=str(root / e["weights"]),
        pad_tail_batch=True,
    )

    def load(j):
        path = root / j["video_path"]
        s = path.stat()
        if (s.st_size, s.st_mtime_ns) != (j["source_bytes"], j["source_mtime_ns"]):
            raise ValueError("CDF原视频改变")
        union = sorted({x for w in j["windows"] for x in w})
        return extractor.prepare_frames(decode_bounded(path, union))

    stream = bounded_map(
        pending,
        load,
        lambda j: frame_reservation(
            root / j["video_path"], len({x for w in j["windows"] for x in w})
        ),
        workers=c["decode_workers"],
        depth=4,
        budget=4 * 2**30,
    )
    start = time.perf_counter()
    for n, (j, frames) in enumerate(stream, 1):
        f = extractor.frames_to_global_patch_embeddings([frames], batch_size=8)[0]
        union = sorted({x for w in j["windows"] for x in w})
        where = {x: i for i, x in enumerate(union)}
        picks = [[where[x] for x in w] for w in j["windows"]]
        g, ug = globals_from_cache(root, j)
        np.testing.assert_array_equal(np.stack([f["global"][ix] for ix in picks]), g)
        p = np.stack([f["patch"][ix] for ix in picks])
        store.add(scorer.score(j, g, p, ug))
        if n % 32 == 0 or n == len(pending):
            elapsed = time.perf_counter() - start
            paper_json(
                out / f"cdf_progress_{rank}.json",
                dict(
                    completed=n,
                    total=len(pending),
                    seconds=elapsed,
                    eta_seconds=(len(pending) - n) * elapsed / n,
                ),
            )
            print("new reference CDF", rank, n, len(pending), flush=True)
    store.flush()
    paper_json(
        out / f"cdf_done_{rank}.json", dict(status="completed", identity=identity, limit=limit)
    )


def _worker(root, device, raw, gg, slots, request, response):
    torch.set_num_threads(2)
    root = Path(root)
    try:
        c, out, identity = setup(root)
        scorer = Scorer(root, c, out, device)
        patches = np.frombuffer(raw, dtype=np.float32).reshape(slots, 3, 16, 196, 1024)
        global_features = np.frombuffer(gg, dtype=np.float32).reshape(slots, 3, 16, 1024)
        while True:
            item = request.get()
            if item is None:
                break
            slot, j = item
            k = len(j["windows"])
            t = len(j["windows"][0])
            result = scorer.score(j, global_features[slot, :k, :t], patches[slot, :k, :t])
            response.put(("ok", slot, result))
    except BaseException:
        response.put(("error", -1, traceback.format_exc()))


def evaluate_stream(root, limit=0):
    c, out, identity = setup(root)
    store = Chunks(out / "evaluation_chunks", identity)
    jobs = [
        j for j in json.loads((root / c["plans"]).read_text())["jobs"] if j["role"] == "evaluation"
    ]
    if limit:
        from collections import Counter

        counts = Counter()
        selected = []
        for j in jobs:
            tag = (
                tuple(j["targets"]),
                len(j["windows"][0]),
                next(iter(j["targets"].values()))["subset"],
            )
            if counts[tag] < limit:
                selected.append(j)
                counts[tag] += 1
        jobs = selected
    pending = [j for j in jobs if j["key"] not in store.records]
    if not pending:
        return
    cache = root / c["patch_cache"]
    manifest = json.loads((cache / "manifest.json").read_text())
    receipts = manifest["records"]
    if manifest["status"] not in ("verified", "ready_for_training"):
        raise ValueError("Patch生产未完成")
    ctx = mp.get_context("spawn")
    slots = 4
    raw = ctx.RawArray("f", slots * 3 * 16 * 196 * 1024)
    gg = ctx.RawArray("f", slots * 3 * 16 * 1024)
    patch = np.frombuffer(raw, dtype=np.float32).reshape(slots, 3, 16, 196, 1024)
    glob = np.frombuffer(gg, dtype=np.float32).reshape(slots, 3, 16, 1024)
    queues = [ctx.Queue(maxsize=slots) for _ in c["devices"]]
    responses = ctx.Queue(maxsize=slots * 2)
    processes = [
        ctx.Process(target=_worker, args=(str(root), device, raw, gg, slots, queues[i], responses))
        for i, device in enumerate(c["devices"])
    ]
    for p in processes:
        p.start()
    paper_json(
        out / "evaluation_workers.json",
        dict(
            pids=[p.pid for p in processes],
            transport="single reader / shared FP32 slots / one stream for both GPUs",
        ),
    )
    free = list(range(slots))
    completed = 0
    inflight = 0
    start = time.perf_counter()
    read_time = 0.0

    def accept(block):
        nonlocal completed, inflight
        try:
            item = responses.get(timeout=30) if block else responses.get_nowait()
        except queue.Empty:
            if any(not p.is_alive() for p in processes):
                raise RuntimeError("评分worker退出但没有结果")
            return False
        kind, slot, result = item
        if kind != "ok":
            raise RuntimeError(result)
        store.add(result)
        free.append(slot)
        completed += 1
        inflight -= 1
        if completed % 32 == 0 or completed == len(pending):
            elapsed = time.perf_counter() - start
            paper_json(
                out / "evaluation_progress.json",
                dict(
                    completed=completed,
                    total=len(pending),
                    seconds=elapsed,
                    read_seconds=read_time,
                    eta_seconds=(len(pending) - completed) * elapsed / completed,
                ),
            )
            print(
                "new reference evaluation",
                completed,
                len(pending),
                round(completed / elapsed, 2),
                "v/s",
                flush=True,
            )
        return True

    try:
        for j in pending:
            while not free:
                accept(True)
            slot = free.pop(0)
            r = receipts[j["key"]]
            path = cache / (j["key"] + ".npy")
            s = path.stat()
            if (s.st_size, s.st_mtime_ns) != (r["bytes"], r["mtime_ns"]):
                raise ValueError("Patch stat改变")
            source = (root / j["video_path"]).stat()
            if (source.st_size, source.st_mtime_ns) != (j["source_bytes"], j["source_mtime_ns"]):
                raise ValueError("原视频改变")
            indices = sorted({x for w in j["windows"] for x in w})
            picks = [[indices.index(x) for x in w] for w in j["windows"]]
            if picks != r["window_positions"] or indices != r["frame_indices"]:
                raise ValueError("Patch索引不符")
            tick = time.perf_counter()
            x = np.load(path, allow_pickle=False)
            if x.dtype != np.float32 or list(x.shape) != r["shape"]:
                raise ValueError("Patch格式不符")
            t = len(j["windows"][0])
            k = len(j["windows"])
            for i, ix in enumerate(picks):
                np.take(x, np.asarray(ix), axis=0, out=patch[slot, i, :t], mode="clip")
            g, _ = globals_from_cache(root, j)
            glob[slot, :k, :t] = g
            read_time += time.perf_counter() - tick
            queues[slot % len(queues)].put((slot, j))
            inflight += 1
            while accept(False):
                pass
        while inflight:
            accept(True)
        for q in queues:
            q.put(None)
        for p in processes:
            p.join()
        if any(p.exitcode for p in processes):
            raise RuntimeError("评分worker失败")
        store.flush()
        paper_json(
            out / "evaluation_done.json",
            dict(
                status="completed",
                identity=identity,
                limit=limit,
                seconds=time.perf_counter() - start,
            ),
        )
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join()
        for q in queues:
            q.close()
        responses.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["cdf", "evaluation"])
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    torch.set_num_threads(2)
    if a.action == "cdf":
        cdf(Path.cwd(), a.rank, a.world, a.limit)
    else:
        evaluate_stream(Path.cwd(), a.limit)
