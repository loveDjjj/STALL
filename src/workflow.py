"""论文主线的原视频评分流程，参数与路径显式注入，不依赖历史实验脚本。"""

from pathlib import Path
import numpy as np
import torch

from features import AlphaStallFeatureExtractor
from data.video import decode_bounded
from selection import validate_indices, uniform_windows, feature_change_windows
from reference import ReferenceBundle
from branches.global_branch import score_global_raw
from math_utils import GaussianMeanCandidateScorerFloat64, l2_normalized_second_order


class RawVideoScorer:
    """原始证据与CDF分离，拟合参考时不构造虚假的占位CDF。"""

    def __init__(self, gs, gt, lt, device="cuda:0", *, dino_repo=None, dino_weights=None):
        self.gs, self.gt = gs, gt
        self.device = device
        self.extractor = AlphaStallFeatureExtractor(
            device, dino_repo=dino_repo, dino_weights=dino_weights, pad_tail_batch=True
        )
        self.local_scorer = GaussianMeanCandidateScorerFloat64([lt], lt.mean, device)

    def score_global_first_window(self, path, indices):
        """Global窗口CDF沿用独立Uniform首窗的Global-only前向。"""
        window = uniform_windows(validate_indices(indices), 1)[0]
        frames = self.extractor.prepare_frames(decode_bounded(Path(path), window.frame_indices))
        return self.global_from_prepared(window, frames)

    def global_from_prepared(self, window, frames):
        features = self.extractor.frames_to_global_embeddings([frames], batch_size=8)[0][None]
        raw = score_global_raw(features, self.gs, self.gt, device=self.device)
        return dict(
            selector="uniform",
            windows=[
                dict(
                    rank=0,
                    frame_indices=list(window.frame_indices),
                    global_spatial_raw=float(raw.spatial[0]),
                    global_temporal_raw=float(raw.temporal_t1[0]),
                )
            ],
        )

    def prepare_coarse(
        self,
        path,
        indices,
        *,
        selector="feature_change",
        k=3,
        include_uniform=False,
        materialize=True,
    ):
        """只做CPU工作；同步路径以生成器维持原来逐8帧的有限内存读取。"""
        path = Path(path)
        indices = validate_indices(indices)
        if len(indices) < 16 or k not in (1, 2, 3):
            raise ValueError("需要至少16个互异dense帧，支持窗口预算1/2/3")
        if selector not in ("uniform", "feature_change"):
            raise ValueError("未知选择器")
        uniform = None
        if include_uniform:
            window = uniform_windows(indices, 1)[0]
            uniform = (
                window,
                self.extractor.prepare_frames(decode_bounded(path, window.frame_indices)),
            )
        coarse_indices = indices[::8] if selector == "feature_change" else []

        def batches():
            for start in range(0, len(coarse_indices), 8):
                yield self.extractor.prepare_frames(
                    decode_bounded(path, coarse_indices[start : start + 8])
                )

        return dict(
            uniform=uniform,
            coarse=list(batches()) if materialize else batches(),
            coarse_count=len(coarse_indices),
        )

    def plan_from_prepared(self, indices, prepared, *, selector="feature_change", k=3):
        """只有消费线程调用，保留每视频/每粗批的GPU前向定义。"""
        uniform = (
            self.global_from_prepared(*prepared["uniform"])
            if prepared["uniform"] is not None
            else None
        )
        if selector == "uniform":
            windows = uniform_windows(indices, k)
        elif selector == "feature_change":
            coarse = []
            for frames in prepared["coarse"]:
                coarse.append(self.extractor.frames_to_global_embeddings([frames], batch_size=8)[0])
            windows = feature_change_windows(indices, np.concatenate(coarse), k)
        else:
            raise ValueError("只支持已定义的uniform/feature_change选择器")
        union = sorted({i for w in windows for i in w.frame_indices})
        return dict(
            windows=windows,
            union=union,
            coarse_count=prepared["coarse_count"],
            selector=selector,
            uniform=uniform,
        )

    def prepare_dense(self, path, plan):
        return self.extractor.prepare_frames(decode_bounded(Path(path), plan["union"]))

    def score_dense(self, plan, frames):
        windows = plan["windows"]
        union = plan["union"]
        features = self.extractor.frames_to_global_patch_embeddings([frames], batch_size=8)[0]
        positions = {frame: i for i, frame in enumerate(union)}
        offsets = [[positions[i] for i in w.frame_indices] for w in windows]
        global_features = np.stack([features["global"][idx] for idx in offsets])
        patches = torch.from_numpy(np.stack([features["patch"][idx] for idx in offsets]))
        local = self.local_scorer.score(l2_normalized_second_order(patches))[:, 0]
        global_raw = score_global_raw(global_features, self.gs, self.gt, device=self.device)
        return dict(
            coarse_frames=plan["coarse_count"],
            dense_unique_frames=len(union),
            selector=plan["selector"],
            windows=[
                dict(
                    rank=i,
                    start_position=w.start_position,
                    frame_indices=list(w.frame_indices),
                    change=w.change,
                    global_spatial_raw=float(global_raw.spatial[i]),
                    global_temporal_raw=float(global_raw.temporal_t1[i]),
                    local_raw=float(local[i]),
                )
                for i, w in enumerate(windows)
            ],
        )

    def score_raw(self, path, indices, *, selector="feature_change", k=3):
        prepared = self.prepare_coarse(path, indices, selector=selector, k=k, materialize=False)
        plan = self.plan_from_prepared(indices, prepared, selector=selector, k=k)
        return self.score_dense(plan, self.prepare_dense(path, plan))


class VideoScorer(RawVideoScorer):
    """推理只接受完整参考包；拟合Gaussian不等于已有可部署检测器。"""

    def __init__(
        self, reference: ReferenceBundle, device="cuda:0", *, dino_repo=None, dino_weights=None
    ):
        reference.validate()
        self.reference = reference
        super().__init__(
            reference.global_spatial,
            reference.global_temporal,
            reference.local_temporal,
            device,
            dino_repo=dino_repo,
            dino_weights=dino_weights,
        )

    def score(self, path, indices, *, selector="feature_change", k=3, method=None):
        self.check_selection(selector, k)
        return self.finalize(self.score_raw(path, indices, selector=selector, k=k), method)

    def check_selection(self, selector, k):
        if selector != self.reference.metadata.get(
            "selector", "feature_change"
        ) or k != self.reference.metadata.get("requested_k", 3):
            raise ValueError("观察规则与参考包不匹配，需先建立该selector/K对应的CDF")

    def finalize(self, raw, method=None):
        windows = raw["windows"]
        result = self.reference.score_video(
            *(
                [w[name] for w in windows]
                for name in ("global_spatial_raw", "global_temporal_raw", "local_raw")
            ),
            **score_options(method),
        )
        return dict(**result, **raw)


def prefetched_raw_scores(root, config, scorer, items, *, include_uniform=False):
    """items为稳定(检查点编号,清单行)；所有CUDA调用仍在当前消费线程。"""
    import json
    from data.prefetch import staged_map, frame_reservation

    root = Path(root)
    runtime = config["runtime"]
    selector = config["selection"]["name"]
    k = config["selection"]["k"]

    def indices(item):
        return validate_indices(json.loads(item[1].downsample_idxs))

    def prepare(item):
        return scorer.prepare_coarse(
            root / item[1].video_path,
            indices(item),
            selector=selector,
            k=k,
            include_uniform=include_uniform,
        )

    def advance(item, prepared):
        return scorer.plan_from_prepared(indices(item), prepared, selector=selector, k=k)

    def finish(item, plan):
        return scorer.prepare_dense(root / item[1].video_path, plan)

    def estimate(item):
        count = len(indices(item))
        coarse = len(range(0, count, 8)) if selector == "feature_change" else 0
        return frame_reservation(
            root / item[1].video_path, coarse + min(count, 16 * k) + (16 if include_uniform else 0)
        )

    stream = staged_map(
        items,
        prepare,
        advance,
        finish,
        estimate,
        workers=runtime.get("decode_workers", 4),
        depth=runtime.get("prefetch_depth", 4),
        budget=runtime.get("prefetch_memory_mb", 4096) * 2**20,
    )
    try:
        for item, plan, frames in stream:
            raw = scorer.score_dense(plan, frames)
            yield item, (dict(uniform=plan["uniform"], selected=raw) if include_uniform else raw)
            del frames, raw, plan
    finally:
        stream.close()


def score_options(method=None):
    method = method or {}
    return dict(
        use_global=method.get("global_enabled", True),
        use_local=method.get("local_enabled", True),
        spatial_weight=method.get("spatial_weight", 0.5),
        global_weight=method.get("global_weight", 0.5),
    )


def replay_scores(windows, references, reference_hashes, method=None):
    """标准raw只允许匹配参考包；改变融合不重新提特征，改变Gaussian必须另有raw。"""
    import pandas as pd

    required = {
        "video_id",
        "dataset",
        "subset",
        "source_model",
        "video_path",
        "window_id",
        "reference_sha256",
        "global_spatial_raw",
        "global_temporal_raw",
        "local_raw",
    }
    if required - set(windows):
        raise ValueError(f"窗口表缺字段：{sorted(required - set(windows))}")
    if windows.duplicated(["video_id", "window_id"]).any():
        raise ValueError("窗口身份重复")
    output = []
    for vid, group in windows.groupby("video_id", sort=False):
        for name in ("dataset", "subset", "source_model", "video_path", "reference_sha256"):
            if group[name].nunique(dropna=False) != 1:
                raise ValueError("同视频元信息不一致")
        group = group.sort_values("window_id")
        first = group.iloc[0]
        domain = first.dataset
        if domain not in references or first.reference_sha256 != reference_hashes[domain]:
            raise ValueError("raw与参考包身份不匹配")
        if group.window_id.tolist() != list(range(len(group))):
            raise ValueError("窗口排名必须从0连续编号")
        result = references[domain].score_video(
            group.global_spatial_raw,
            group.global_temporal_raw,
            group.local_raw,
            **score_options(method),
        )
        output.append(
            dict(
                video_id=vid,
                **{k: first[k] for k in ("dataset", "subset", "source_model", "video_path")},
                **result,
            )
        )
    if not output:
        raise ValueError("窗口表为空")
    return pd.DataFrame(output)


def video_file_states(root, frame):
    """运行开始/恢复时冻结物理文件stat；明确是stat合同而非完整内容去重。"""
    result = {}
    for row in frame.itertuples():
        path = Path(root) / row.video_path
        stat = path.stat()
        result[row.video_id] = dict(
            path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns
        )
    return result


def replay_checkpointed(frame, references, hashes, method, directory, identity, progress=None):
    import pandas as pd
    from artifacts import checkpoint_read, checkpoint_write

    rows = []
    for index, (vid, group) in enumerate(frame.groupby("video_id", sort=False), 1):
        path = Path(directory) / "checkpoints" / f"{index:06d}.json"
        if path.exists():
            result = checkpoint_read(path, identity, vid)
        else:
            result = replay_scores(group, references, hashes, method).iloc[0].to_dict()
            checkpoint_write(path, identity, result)
        rows.append(result)
        if progress:
            progress(index, vid)
    return pd.DataFrame(rows)


def score_manifest_checkpointed(
    root,
    config,
    frame,
    reference,
    reference_hash,
    directory,
    identity,
    states,
    progress=None,
    *,
    row_indices=None,
    require_cached=False,
):
    import pandas as pd
    from artifacts import checkpoint_read, checkpoint_write
    import json
    from execution import row_numbers

    root = Path(root)
    directory = Path(directory)
    scorer = None
    rows = []
    windows = []
    items = list(zip(row_numbers(frame, row_indices), frame.itertuples()))

    def publish(index, row, result):
        stat = (root / row.video_path).stat()
        expected = states[row.video_id]
        if stat.st_size != expected["bytes"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError("评分期间原视频文件改变")
        metadata = {
            name: getattr(row, name)
            for name in ("video_id", "video_path", "dataset", "subset", "source_model")
        }
        payload = dict(**metadata, reference_sha256=reference_hash, **result)
        checkpoint_write(directory / "raw" / f"{index:06d}.json", identity, payload)
        return payload

    prefetch = config.get("runtime", {}).get("prefetch_enabled", False)
    if prefetch and not require_cached:
        pending = []
        for item in items:
            path = directory / "raw" / f"{item[0]:06d}.json"
            if path.exists():
                checkpoint_read(path, identity, item[1].video_id)
            else:
                pending.append(item)
        if pending:
            scorer = VideoScorer(
                reference,
                config["runtime"]["device"],
                dino_repo=str(root / config["encoder"]["repo"]),
                dino_weights=str(root / config["encoder"]["weights"]),
            )
            scorer.check_selection(config["selection"]["name"], config["selection"]["k"])
            order = {row.video_id: i + 1 for i, (_, row) in enumerate(items)}
            stream = prefetched_raw_scores(root, config, scorer, pending)
            try:
                for (index, row), raw in stream:
                    publish(index, row, scorer.finalize(raw, config["method"]))
                    if progress:
                        progress(order[row.video_id], row.video_id)
            finally:
                stream.close()
        require_cached = True
    for completed, (index, row) in enumerate(items, 1):
        path = directory / "raw" / f"{index:06d}.json"
        if path.exists():
            payload = checkpoint_read(path, identity, row.video_id)
        else:
            if require_cached:
                raise ValueError("worker返回后仍有缺失评分检查点")
            if scorer is None:
                scorer = VideoScorer(
                    reference,
                    config["runtime"]["device"],
                    dino_repo=str(root / config["encoder"]["repo"]),
                    dino_weights=str(root / config["encoder"]["weights"]),
                )
            result = scorer.score(
                root / row.video_path,
                validate_indices(json.loads(row.downsample_idxs)),
                selector=config["selection"]["name"],
                k=config["selection"]["k"],
                method=config["method"],
            )
            payload = publish(index, row, result)
        metadata = {
            name: payload[name]
            for name in ("video_id", "video_path", "dataset", "subset", "source_model")
        }
        result = {
            name: value
            for name, value in payload.items()
            if name not in (*metadata, "reference_sha256", "windows")
        }
        for window in payload["windows"]:
            window = dict(window)
            window["window_id"] = window.pop("rank")
            for name in ("global_spatial_raw", "global_temporal_raw", "local_raw"):
                window[name] = float(window[name])
            window["frame_indices"] = json.dumps(window["frame_indices"])
            windows.append(dict(**metadata, reference_sha256=reference_hash, **window))
        rows.append(dict(**metadata, **result))
        if progress and not prefetch:
            progress(completed, row.video_id)
    if progress and prefetch and items:
        progress(len(items), items[-1][1].video_id)
    return pd.DataFrame(rows), pd.DataFrame(windows)


def score_manifest_distributed(
    root, config, frame, reference, reference_hash, directory, identity, states, progress=None
):
    from execution import devices_for, build_jobs, run_workers, gpu_job

    if len(devices_for(config)) == 1:
        import copy

        selected = copy.deepcopy(config)
        selected["runtime"]["device"] = devices_for(config)[0]
        return score_manifest_checkpointed(
            root, selected, frame, reference, reference_hash, directory, identity, states, progress
        )
    jobs = build_jobs(
        frame,
        config,
        dict(
            kind="score",
            root=root,
            directory=directory,
            identity=identity,
            reference=reference,
            reference_hash=reference_hash,
            states=states,
        ),
    )
    run_workers(
        gpu_job,
        jobs,
        directory,
        "score",
        lambda stage, done, total, vid: progress(done, vid) if progress else None,
    )
    return score_manifest_checkpointed(
        root,
        config,
        frame,
        reference,
        reference_hash,
        directory,
        identity,
        states,
        require_cached=True,
    )


def evaluate_run(root, config, directory, pairs_path, command, *, resume=False, dry_run=False):
    """在原run追加可恢复评价，不改评分文件；发布过程与数值计算分别记录。"""
    import fcntl
    import importlib.metadata
    import json
    import shutil
    import sys
    import traceback
    import pandas as pd
    from artifacts import (
        read_paper_scores,
        paper_json,
        atomic_csv,
        publish_stage_file,
        finish_paper_stage,
        PaperLog,
    )
    from config import config_digest
    from reference import file_digest
    from evaluation.tables import evaluate_fixed_pairs

    root = Path(root)
    directory = Path(directory).resolve()
    pairs_path = Path(pairs_path).resolve()
    stage = directory / "stages/evaluate"
    plan_path = stage / "plan.json"
    outputs = (
        "generator_metrics.csv",
        "dataset_metrics.csv",
        "macro_metrics.csv",
        "full_population_metrics.csv",
        "pair_ids.csv",
    )
    lock_path = directory / ".run.lock"
    lock = lock_path.open("r" if dry_run else "a") if lock_path.exists() or not dry_run else None
    try:
        if lock:
            fcntl.flock(
                lock, fcntl.LOCK_SH | fcntl.LOCK_NB if dry_run else fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        scores, source = read_paper_scores(directory)
        manifest = json.loads((directory / "run_manifest.json").read_text())
        if manifest["config_sha256"] != config_digest(config):
            raise ValueError("评价配置与原评分不同")
        names = (
            "src/workflow.py",
            "src/artifacts.py",
            "src/config.py",
            "src/evaluation/tables.py",
            "src/evaluation/metrics.py",
            "scripts/run.py",
        )
        code = {name: file_digest(root / name) for name in names}
        pair_identity = dict(path=str(pairs_path), sha256=file_digest(pairs_path))
        identity = dict(
            config_sha256=manifest["config_sha256"],
            scores_sha256=source["scores_sha256"],
            pairs=pair_identity,
            references=manifest["inputs"].get("references"),
            code_sha256=code,
            python=sys.version,
            packages={
                name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "scikit-learn")
            },
        )
        plan = json.loads(plan_path.read_text()) if plan_path.exists() else None
        if plan is not None:
            if not resume:
                raise FileExistsError("评价阶段已存在，请显式--resume")
            if plan.get("schema") != "paper_evaluation_stage_v1" or plan["identity"] != identity:
                raise ValueError("评价恢复的输入/配置/源码身份改变")
        elif any((directory / name).exists() for name in outputs):
            raise FileExistsError("已有无当前阶段身份的评价文件，拒绝覆盖")
        if dry_run:
            # 严格配对校验也执行，避免dry-run把缺视频的错误留到正式评价。
            evaluate_fixed_pairs(scores, pd.read_csv(pairs_path, dtype={"video_id": str}))
            print(
                json.dumps(
                    dict(action="evaluate", run=str(directory), resume=resume, identity=identity),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        stage.mkdir(parents=True, exist_ok=True)
        if plan is None:
            plan = dict(
                schema="paper_evaluation_stage_v1",
                identity=identity,
                status="preparing",
                command=command,
            )
            paper_json(plan_path, plan)
        with PaperLog(directory):
            try:
                for name, digest in code.items():
                    target = stage / "code" / name
                    if target.exists():
                        if file_digest(target) != digest:
                            raise ValueError("评价源码快照改变")
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(root / name, target)
                    if file_digest(target) != digest or file_digest(root / name) != digest:
                        raise ValueError("评价源码复制期间改变")
                if plan["status"] == "completed":
                    status = json.loads((directory / "status.json").read_text())
                    artifacts = json.loads((directory / "artifact_manifest.json").read_text())
                    if status.get("status") == "completed" and artifacts.get("stage") == "evaluate":
                        for name, digest in artifacts["artifacts"].items():
                            if file_digest(directory / name) != digest:
                                raise ValueError("已完成评价的产物改变")
                        print(f"[已经完成:evaluate] {directory}", flush=True)
                        return
                print("[开始:evaluate] 固定配对身份与阶段产物", flush=True)
                paper_json(
                    directory / "status.json",
                    dict(
                        status="running",
                        stage="evaluate",
                        completed_steps=manifest["completed_steps"],
                    ),
                )
                if plan["status"] == "preparing":
                    pairs = pd.read_csv(pairs_path, dtype={"video_id": str})
                    tables = evaluate_fixed_pairs(scores, pairs)
                    for name, table in tables.items():
                        atomic_csv(stage / "payload" / f"{name}.csv", table)
                    atomic_csv(stage / "payload/pair_ids.csv", pairs)
                    plan.update(
                        status="prepared",
                        outputs={name: file_digest(stage / "payload" / name) for name in outputs},
                    )
                    paper_json(plan_path, plan)
                if (
                    file_digest(pairs_path) != pair_identity["sha256"]
                    or file_digest(directory / "video_scores.csv") != identity["scores_sha256"]
                ):
                    raise ValueError("评价期间输入改变")
                for name, digest in plan["outputs"].items():
                    publish_stage_file(stage / "payload" / name, directory / name, digest)
                # 保留其他元信息，但不接受评价期间偷偷更换评分身份。
                manifest = json.loads((directory / "run_manifest.json").read_text())
                if (
                    manifest["config_sha256"] != identity["config_sha256"]
                    or manifest["inputs"].get("references") != identity["references"]
                ):
                    raise ValueError("评价期间run身份改变")
                manifest["inputs"]["evaluation_pairs"] = pair_identity
                manifest.setdefault("stage_commands", {})["evaluate"] = plan["command"]
                manifest.setdefault("stage_inputs", {})["evaluate"] = identity
                if resume:
                    manifest.setdefault("stage_resume_commands", {}).setdefault(
                        "evaluate", []
                    ).append(command)
                paper_json(directory / "run_manifest.json", manifest)
                plan["status"] = "completed"
                paper_json(plan_path, plan)
                finish_paper_stage(directory, "evaluate")
                print(pd.read_csv(directory / "macro_metrics.csv").to_string(index=False))
            except BaseException as exc:
                print(traceback.format_exc(), file=sys.stderr)
                paper_json(
                    directory / "status.json",
                    dict(
                        status="failed",
                        stage="evaluate",
                        error=str(exc),
                        completed_steps=manifest["completed_steps"],
                    ),
                )
                raise
    finally:
        if lock:
            lock.close()
