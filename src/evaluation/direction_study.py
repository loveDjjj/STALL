"""真实参考Local方向研究：共享Patch、独立CDF、原子恢复和完整身份评价。"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

from artifacts import atomic_csv, checkpoint_read, checkpoint_write, paper_json
from config import config_digest
from evaluation.direction_features import direction_features, pooled_score, fit_source_balanced
from evaluation.tables import evaluate_fixed_pairs
from evaluation.bootstrap import paired_source_contrast
from math_utils import (
    GaussianMeanCandidateScorerFloat64,
    StableGaussianParams,
    l2_normalized_second_order,
)
from reference import (
    file_digest,
    fit_gaussian,
    local_video_cdfs,
    percentile,
    publish_arrays,
    window_mean,
)
from reference_fit import sample_d2_positions
from selection import uniform_windows

DOMAINS = ("comgenvid", "videofeedback", "genvideo")
CODE = (
    "src/evaluation/direction_study.py",
    "src/evaluation/direction_features.py",
    "src/features.py",
    "src/math_utils.py",
    "src/reference.py",
    "src/reference_fit.py",
    "src/data/video.py",
    "src/selection.py",
)


def configuration(root):
    c = yaml.safe_load((root / "configs/local_direction.yaml").read_text())
    keys = {
        "protocol",
        "output",
        "plans",
        "baseline",
        "patch_cache",
        "reference_seeds",
        "devices",
        "decode_workers",
        "prefetch_depth",
        "prefetch_gib",
        "disk_floor_gib",
        "fit_clips",
        "ridge",
        "split_epsilon",
        "split_gamma",
        "bootstrap_iterations",
        "bootstrap_seed",
    }
    if set(c) != keys or c["fit_clips"] != 200 or c["ridge"] != 1e-5:
        raise ValueError("研究配置未知字段或改变冻结预算/ridge")
    if c["split_epsilon"] != 1e-8 or c["split_gamma"] != 8.0:
        raise ValueError("不搜索SPLIT超参数")
    if len(set(c["reference_seeds"])) != 5:
        raise ValueError("需要五个预定源重抽样seed")
    return c


def context(root):
    root = Path(root).resolve()
    c = configuration(root)
    out = root / c["output"]
    p = root / c["plans"]
    source = json.loads((p.parent / "data_manifest.json").read_text())
    if source["files"]["plans.json"] != file_digest(p):
        raise ValueError("冻结窗口计划改变")
    inputs = {str(p.relative_to(root)): file_digest(p)}
    for name in ("evaluation.csv", "pairs.csv", "scores_fc3.csv.gz", "scores_uniform3.csv.gz"):
        path = root / c["baseline"] / name
        inputs[str(path.relative_to(root))] = file_digest(path)
    for d in DOMAINS:
        for name in (
            f"data/manifests/active/{d}/fit.csv",
            f"results/runs/paper_fit_{d}/gaussians.npz",
            f"results/runs/paper_representation_fit_{d}/gaussians.npz",
        ):
            inputs[name] = file_digest(root / name)
    inputs["results/runs/paper_fit_source/gaussians.npz"] = file_digest(
        root / "results/runs/paper_fit_source/gaussians.npz"
    )
    cfg = yaml.safe_load((root / "configs/paper.yaml").read_text())
    spec = dict(
        config=c,
        encoder=cfg["encoder"],
        inputs=inputs,
        code={p: file_digest(root / p) for p in CODE},
    )
    identity = config_digest(spec)
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "identity.json"
    if marker.exists() and json.loads(marker.read_text()) != spec:
        raise ValueError("研究输入/数值代码改变，拒绝混合恢复")
    if not marker.exists():
        paper_json(marker, spec)
    return c, out, spec, identity


def plans(root, c):
    jobs = json.loads((root / c["plans"]).read_text())["jobs"]
    expected = pd.read_csv(root / c["baseline"] / "evaluation.csv", keep_default_na=False)
    ids = [m["video_id"] for j in jobs if j["role"] == "evaluation" for m in j["targets"].values()]
    if len(ids) != len(set(ids)) or set(ids) != set(expected.video_id):
        raise ValueError("计划不是完整23单元身份")
    return jobs


def parameter(path, name):
    with np.load(path, allow_pickle=False) as z:
        return StableGaussianParams(
            z[name + "_mean"].copy(), z[name + "_whitening"].copy(), np.empty(0)
        )


def fit_jobs(root, c):
    jobs = []
    blocked = pd.read_csv(root / c["baseline"] / "evaluation.csv", keep_default_na=False)
    forbidden = set(blocked.source_group)
    for d in DOMAINS:
        frame = pd.read_csv(root / f"data/manifests/active/{d}/fit.csv", keep_default_na=False)
        if len(frame) != c["fit_clips"] or not frame.subset.eq("real").all():
            raise ValueError("fit预算/标签错误")
        if set(frame.source_group) & forbidden:
            raise ValueError("fit/evaluation源组交叉")
        for r in frame.to_dict("records"):
            windows = [
                list(w.frame_indices) for w in uniform_windows(json.loads(r["downsample_idxs"]), 3)
            ]
            spec = dict(video_path=r["video_path"], windows=windows, role="fit")
            st = (root / r["video_path"]).stat()
            jobs.append(
                dict(
                    **spec,
                    key=config_digest(spec),
                    dataset=d,
                    row=r,
                    source_bytes=st.st_size,
                    source_mtime_ns=st.st_mtime_ns,
                )
            )
    return jobs


def extract_fit(root, args):
    from features import AlphaStallFeatureExtractor
    from data.prefetch import bounded_map, frame_reservation
    from data.video import decode_bounded

    c, out, spec, identity = context(root)
    jobs = fit_jobs(root, c)
    jobs = [j for i, j in enumerate(jobs) if i % args.world_size == args.rank]
    if args.limit:
        jobs = jobs[: args.limit]
    pending = []
    for j in jobs:
        p = out / "fit_features" / (j["key"] + ".json")
        if p.exists():
            s = checkpoint_read(p, identity, j["key"])
            if file_digest(out / "fit_features" / (j["key"] + ".npz")) != s["sha256"]:
                raise ValueError("fit检查点改变")
        else:
            pending.append(j)
    if not pending:
        return
    e = spec["encoder"]
    if file_digest(root / e["weights"]) != e["weights_sha256"]:
        raise ValueError("DINO权重改变")
    extractor = AlphaStallFeatureExtractor(
        c["devices"][args.rank],
        dino_repo=str(root / e["repo"]),
        dino_weights=str(root / e["weights"]),
        pad_tail_batch=True,
    )

    def load(j):
        st = (root / j["video_path"]).stat()
        if (st.st_size, st.st_mtime_ns) != (j["source_bytes"], j["source_mtime_ns"]):
            raise ValueError("fit视频改变")
        union = sorted({x for w in j["windows"] for x in w})
        return extractor.prepare_frames(decode_bounded(root / j["video_path"], union))

    stream = bounded_map(
        pending,
        load,
        lambda j: frame_reservation(
            root / j["video_path"], len({x for w in j["windows"] for x in w})
        ),
        workers=c["decode_workers"],
        depth=c["prefetch_depth"],
        budget=c["prefetch_gib"] * 2**30,
    )
    start = time.perf_counter()
    for n, (j, frames) in enumerate(stream, 1):
        f = extractor.frames_to_global_patch_embeddings([frames], batch_size=8)[0]
        union = sorted({x for w in j["windows"] for x in w})
        where = {x: i for i, x in enumerate(union)}
        pick = [[where[x] for x in w] for w in j["windows"]]
        p = np.stack([f["patch"][ix] for ix in pick])
        g = np.stack([f["global"][ix] for ix in pick])
        r = j["row"]
        asset = root / r["feature_asset"]
        if file_digest(asset) != r["feature_asset_sha256"]:
            raise ValueError("原fit轻资产改变")
        with np.load(asset, allow_pickle=False) as z:
            np.testing.assert_array_equal(g, z["global_windows"])
            x = torch.from_numpy(p)
            raw = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
            sample = sample_d2_positions(
                raw.reshape(-1, 1024), r["legacy_path"] or r["video_path"]
            ).numpy()
            np.testing.assert_array_equal(sample, z["raw_d2"])
        pooled, scalar = direction_features(p)
        path = out / "fit_features" / (j["key"] + ".npz")
        publish_arrays(
            path,
            dict(
                pooled=pooled,
                scalar=scalar,
                frame_indices=np.asarray(j["windows"]),
                video_id=np.asarray(r["video_id"]),
            ),
        )
        checkpoint_write(
            path.with_suffix(".json"),
            identity,
            dict(
                video_id=j["key"], sha256=file_digest(path), source_sha256=r["feature_asset_sha256"]
            ),
        )
        if n % 10 == 0 or n == len(pending):
            paper_json(
                out / f"fit_progress_{args.rank}.json",
                dict(completed=n, pending=len(pending), seconds=time.perf_counter() - start),
            )
            print("fit features", args.rank, n, len(pending), flush=True)
    paper_json(
        out / f"fit_complete_{args.rank}.json",
        dict(status="completed", limit=args.limit, identity=identity),
    )


def fit_models(root, args):
    c, out, spec, identity = context(root)
    for d in DOMAINS:
        dest = out / "models" / f"{d}.npz"
        marker = dest.with_suffix(".json")
        if marker.exists():
            m = checkpoint_read(marker, identity, d)
            if file_digest(dest) != m["sha256"]:
                raise ValueError("模型改变")
            continue
        jobs = [j for j in fit_jobs(root, c) if j["dataset"] == d]
        pooled = []
        d1 = []
        d2 = []
        groups = []
        sources = []
        for j in jobs:
            r = j["row"]
            p = out / "fit_features" / (j["key"] + ".npz")
            record = checkpoint_read(p.with_suffix(".json"), identity, j["key"])
            if file_digest(p) != record["sha256"]:
                raise ValueError("pooled fit改变")
            with np.load(p) as z:
                pooled.append(z["pooled"].reshape(-1, 1024))
            asset = root / r["feature_asset"]
            if file_digest(asset) != r["feature_asset_sha256"]:
                raise ValueError("原fit资产改变")
            with np.load(asset) as z:
                d1.append(z["unit_d1"].copy())
                d2.append(
                    torch.nn.functional.normalize(
                        torch.from_numpy(z["raw_d2"].copy()), dim=-1, eps=1e-12
                    ).numpy()
                )
            groups.append(r["source_group"])
            sources.append(
                dict(
                    video_id=r["video_id"],
                    source_group=r["source_group"],
                    sha256=r["feature_asset_sha256"],
                )
            )
        models = {
            "pooled_d2": fit_gaussian(pooled),
            "d2_anchor": parameter(root / f"results/runs/paper_fit_{d}/gaussians.npz", "lt"),
        }
        models["d1_anchor"] = parameter(
            root / f"results/runs/paper_representation_fit_{d}/gaussians.npz", "local_d1"
        )
        models["d2_diagonal"] = parameter(
            root / f"results/runs/paper_representation_fit_{d}/gaussians.npz", "local_diagonal_d2"
        )
        models["d2_source_balanced"] = fit_source_balanced(d2, groups)
        unique = sorted(set(groups))
        draws = {}
        for seed in c["reference_seeds"]:
            counts = Counter(np.random.default_rng(seed).choice(unique, len(unique), replace=True))
            ids = [i for i, g in enumerate(groups) for _ in range(counts[g])]
            draws[str(seed)] = {g: int(counts[g]) for g in unique}
            models[f"d2_boot_{seed}"] = fit_gaussian([d2[i] for i in ids])
            models[f"d1_boot_{seed}"] = fit_gaussian([d1[i] for i in ids])
            models[f"d2_diag_boot_{seed}"] = fit_gaussian([d2[i] for i in ids], diagonal=True)
            print("reference bootstrap", d, seed, flush=True)
        arrays = {}
        for name, m in models.items():
            arrays[name + "_mean"] = m.mean
            arrays[name + "_whitening"] = m.whitening
        publish_arrays(dest, arrays)
        checkpoint_write(
            marker,
            identity,
            dict(
                video_id=d,
                sha256=file_digest(dest),
                fit=sources,
                draws=draws,
                uncertainty="original-fit-pool source bootstrap; fixed Global; not new independent real pools",
                pooled_observations=sum(len(x) for x in pooled),
            ),
        )


class Scorer:
    def __init__(self, root, c, out, device):
        self.root = root
        self.c = c
        self.out = out
        self.device = device
        self.extractor = None
        source = parameter(root / "results/runs/paper_fit_source/gaussians.npz", "lt")
        self.models = {}
        self.scorers = {}
        self.center = source.mean
        for d in DOMAINS:
            with np.load(out / "models" / f"{d}.npz") as z:
                ms = {
                    k[:-5]: StableGaussianParams(
                        z[k].copy(), z[k[:-5] + "_whitening"].copy(), np.empty(0)
                    )
                    for k in z.files
                    if k.endswith("_mean")
                }
            self.models[d] = dict(pooled=ms["pooled_d2"])
            for rep in ("d1", "d2"):
                names = [k for k in ms if k.startswith(rep + "_")]
                self.models[d][rep] = {name: ms[name] for name in names}
        self.cache = json.loads((root / c["patch_cache"] / "manifest.json").read_text())
        if self.cache["status"] not in ("ready_for_training", "verified"):
            raise ValueError("Patch生产未验收")
        cache_spec = json.loads((root / c["patch_cache"] / "identity.json").read_text())
        paper = yaml.safe_load((root / "configs/paper.yaml").read_text())
        if (
            cache_spec["encoder"] != paper["encoder"]
            or cache_spec["batch"] != 8
            or cache_spec["dtype"] != "float32"
            or not cache_spec["pad_tail"]
        ):
            raise ValueError("Patch数值合同不匹配")
        self.accepted = {config_digest(cache_spec)}
        compat = root / c["patch_cache"] / "producer_compatibility.json"
        if compat.exists():
            proof = json.loads(compat.read_text())
            if proof["current_identity"] not in self.accepted:
                raise ValueError("Patch兼容链断开")
            self.accepted.add(proof["previous_identity"])

    def load(self, j):
        p = self.root / j["video_path"]
        st = p.stat()
        if (st.st_size, st.st_mtime_ns) != (j["source_bytes"], j["source_mtime_ns"]):
            raise ValueError("原视频stat改变")
        if j["role"] == "evaluation":
            r = self.cache["records"][j["key"]]
            path = self.root / self.c["patch_cache"] / (j["key"] + ".npy")
            s = path.stat()
            if r["identity"] not in self.accepted:
                raise ValueError("未知Patch生产身份")
            if (s.st_size, s.st_mtime_ns) != (r["bytes"], r["mtime_ns"]):
                raise ValueError("Patch stat改变")
            indices = sorted({x for w in j["windows"] for x in w})
            pick = [[indices.index(x) for x in w] for w in j["windows"]]
            if r["frame_indices"] != indices or r["window_positions"] != pick:
                raise ValueError("Patch索引改变")
            x = np.load(path, allow_pickle=False)
            if x.dtype != np.float32 or list(x.shape) != r["shape"]:
                raise ValueError("Patch形状或dtype改变")
            return np.stack([x[ix] for ix in pick]), None
        from data.video import decode_bounded

        union = sorted({x for w in j["windows"] for x in w})
        return None, self.extractor.prepare_frames(decode_bounded(p, union))

    def score(self, j, data):
        p, frames = data
        if p is None:
            f = self.extractor.frames_to_global_patch_embeddings([frames], batch_size=8)[0]
            union = sorted({x for w in j["windows"] for x in w})
            where = {x: i for i, x in enumerate(union)}
            pick = [[where[x] for x in w] for w in j["windows"]]
            p = np.stack([f["patch"][ix] for ix in pick])
            # CDF新前向必须复现该窗口原Global，不需要重新拟合或改写Global。
            path = self.root / "results/runs/mainline_experts/global" / (j["key"] + ".npz")
            with np.load(path) as z:
                np.testing.assert_array_equal(
                    np.stack([f["global"][ix] for ix in pick]), z["global_features"]
                )
        pooled, scalar = direction_features(
            p, epsilon=self.c["split_epsilon"], gamma=self.c["split_gamma"]
        )
        x = torch.from_numpy(p)
        units = {
            "d2": l2_normalized_second_order(x),
            "d1": torch.nn.functional.normalize(x[:, 1:] - x[:, :-1], dim=-1, eps=1e-12),
        }
        result = {}
        error = 0.0
        domains = tuple(sorted(j["targets"]))
        # 三域共用CDF视频时，同一表示只计算一次方向二阶矩。
        shared = {}
        for rep in ("d1", "d2"):
            key = (domains, rep)
            if key not in self.scorers:
                names = [(d, name) for d in domains for name in self.models[d][rep]]
                ms = [self.models[d][rep][name] for d, name in names]
                self.scorers[key] = (
                    names,
                    GaussianMeanCandidateScorerFloat64(ms, self.center, self.device),
                )
            names, scorer = self.scorers[key]
            actual = scorer.score(units[rep])
            shared.update({pair: actual[:, i].tolist() for i, pair in enumerate(names)})
        for d, meta in j["targets"].items():
            m = self.models[d]
            values = {"pooled_d2": pooled_score(pooled, m["pooled"]).tolist()}
            values.update({name: v for (domain, name), v in shared.items() if domain == d})
            expected = np.asarray([float(w["lt"]) for w in meta["expected"]])
            actual = np.asarray(values["d2_anchor"])
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8)
            error = max(error, float(np.abs(actual - expected).max()))
            # 固定回归锚点使用原raw，微小GEMM舍入不改变原CDF并列关系。
            values["d2_anchor"] = expected.tolist()
            result[d] = values
        return dict(
            video_id=j["key"],
            key=j["key"],
            targets=result,
            scalars=scalar.astype(float).tolist(),
            max_baseline_error=error,
            unique_frames=len({x for w in j["windows"] for x in w}),
        )


def dense(root, args):
    from data.prefetch import bounded_map, frame_reservation

    c, out, spec, identity = context(root)
    jobs = plans(root, c)
    if args.role:
        jobs = [j for j in jobs if j["role"] == args.role]
    jobs = [j for i, j in enumerate(jobs) if i % args.world_size == args.rank]
    if args.limit:
        counts = Counter()
        subset = []
        for j in jobs:
            tag = (j["role"], tuple(j["targets"]), len(j["windows"][0]))
            if counts[tag] < args.limit:
                subset.append(j)
                counts[tag] += 1
        jobs = subset
    model_hashes = {d: file_digest(out / "models" / f"{d}.npz") for d in DOMAINS}
    dense_identity = config_digest(
        dict(identity=identity, models=model_hashes, runner=file_digest(Path(__file__)))
    )
    marker = out / "dense_identity.json"
    ms = dict(identity=dense_identity, models=model_hashes)
    if marker.exists() and json.loads(marker.read_text()) != ms:
        raise ValueError("密集评分实现/模型改变")
    if not marker.exists():
        paper_json(marker, ms)
    pending = []
    for j in jobs:
        path = out / "raw" / (j["key"] + ".json")
        if path.exists():
            checkpoint_read(path, dense_identity, j["key"])
        else:
            pending.append(j)
    if not pending:
        return
    scorer = Scorer(root, c, out, c["devices"][args.rank])
    if any(j["role"] == "cdf" for j in pending):
        from features import AlphaStallFeatureExtractor

        e = spec["encoder"]
        if file_digest(root / e["weights"]) != e["weights_sha256"]:
            raise ValueError("DINO权重改变")
        scorer.extractor = AlphaStallFeatureExtractor(
            c["devices"][args.rank],
            dino_repo=str(root / e["repo"]),
            dino_weights=str(root / e["weights"]),
            pad_tail_batch=True,
        )

    def estimate(j):
        n = len({x for w in j["windows"] for x in w})
        return (
            max(256 * 2**20, n * 196 * 1024 * 4 * 5)
            if j["role"] == "evaluation"
            else frame_reservation(root / j["video_path"], n)
        )

    start = time.perf_counter()
    error = 0.0
    stream = bounded_map(
        pending,
        scorer.load,
        estimate,
        workers=c["decode_workers"],
        depth=c["prefetch_depth"],
        budget=c["prefetch_gib"] * 2**30,
    )
    for n, (j, data) in enumerate(stream, 1):
        import shutil

        if shutil.disk_usage(out).free < c["disk_floor_gib"] * 2**30:
            raise RuntimeError("数据盘达到保留边界")
        raw = scorer.score(j, data)
        error = max(error, raw["max_baseline_error"])
        checkpoint_write(out / "raw" / (j["key"] + ".json"), dense_identity, raw)
        if n % 16 == 0 or n == len(pending):
            elapsed = time.perf_counter() - start
            paper_json(
                out / f"dense_progress_{args.rank}.json",
                dict(
                    completed=n,
                    pending=len(pending),
                    seconds=elapsed,
                    role=j["role"],
                    max_baseline_error=error,
                    eta_seconds=(len(pending) - n) * elapsed / n,
                ),
            )
            print(
                "direction dense",
                args.rank,
                n,
                len(pending),
                j["role"],
                round(n / elapsed, 3),
                "v/s",
                flush=True,
            )
    paper_json(
        out / f"dense_complete_{args.rank}.json",
        dict(status="completed", identity=dense_identity, role=args.role, limit=args.limit),
    )


def evaluate(root, args):
    c, out, spec, identity = context(root)
    jobs = plans(root, c)
    did = json.loads((out / "dense_identity.json").read_text())["identity"]
    refs = {}
    records = {}
    for j in jobs:
        raw = checkpoint_read(out / "raw" / (j["key"] + ".json"), did, j["key"])
        records[j["key"]] = raw
        if j["role"] != "cdf":
            continue
        length = len(j["windows"][0])
        for d, values in raw["targets"].items():
            all_values = dict(values)
            for i, name in enumerate(("ttr", "lsmi", "split")):
                all_values[name] = (-np.asarray(raw["scalars"])[:, i]).tolist()
            for name, values in all_values.items():
                refs.setdefault((d, length, name), []).append(values)
    cdfs = {}
    for key, values in refs.items():
        if len(values) != 2000:
            raise ValueError("每个CDF必须2000真实视频")
        cdfs[key] = (
            local_video_cdfs(values) if key[1] == 16 else {1: np.sort(np.asarray(values)[:, 0])}
        )
    base = pd.read_csv(root / c["baseline"] / "scores_fc3.csv.gz", float_precision="round_trip")
    g = base[base.variant.eq("global_only")].set_index("video_id").final_score
    full = base[base.variant.eq("full")].set_index("video_id").final_score
    rows = []
    regression = []
    for j in jobs:
        if j["role"] != "evaluation":
            continue
        raw = records[j["key"]]
        length = len(j["windows"][0])
        k = len(j["windows"])
        for d, meta in j["targets"].items():
            vals = dict(raw["targets"][d])
            for i, name in enumerate(("ttr", "lsmi", "split")):
                vals[name] = (-np.asarray(raw["scalars"])[:, i]).tolist()
            m = {
                key: meta[key]
                for key in ("video_id", "dataset", "subset", "source_model", "source_group")
            }
            m.update(video_path=j["video_path"], effective_k=k, window_frames=length)
            for name, values in vals.items():
                q = window_mean(values)
                l = float(percentile([q], cdfs[d, length, name][k])[0])
                s = 0.5 * float(g[meta["video_id"]]) + 0.5 * l
                for suffix, v in (("", s), ("_only", l), ("_raw", q)):
                    rows.append(dict(**m, variant=name + suffix, final_score=v))
                if name == "d2_anchor":
                    err = abs(s - float(full[meta["video_id"]]))
                    if err > 1e-12:
                        raise ValueError("原D2完整CDF/Final不能回归")
                    regression.append(
                        dict(
                            video_id=meta["video_id"],
                            final_error=err,
                            raw_error=raw["max_baseline_error"],
                        )
                    )
            rows.append(dict(**m, variant="global_only", final_score=float(g[meta["video_id"]])))
    scores = pd.DataFrame(rows)
    pairs = pd.read_csv(root / c["baseline"] / "pairs.csv")
    tables = {}
    if scores.video_id.nunique() != 15569:
        raise ValueError("23单元分数缺失")
    path = out / "video_scores.csv.gz"
    tmp = out / "video_scores.tmp.csv.gz"
    scores.to_csv(tmp, index=False, compression="gzip")
    tmp.replace(path)
    atomic_csv(out / "regression.csv", pd.DataFrame(regression))
    for name, part in scores.groupby("variant", sort=False):
        for key, table in evaluate_fixed_pairs(part, pairs).items():
            tables.setdefault(key, []).append(table.assign(variant=name))
    for key, items in tables.items():
        atomic_csv(
            out / (key + ".csv"), pd.concat(items).replace({"scope": {"Macro-3": "Average"}})
        )
    publish_arrays(
        out / "cdf_arrays.npz",
        {f"{d}__T{t}__{name}__K{k}": a for (d, t, name), v in cdfs.items() for k, a in v.items()},
    )
    paper_json(
        out / "evaluation.json",
        dict(
            status="point_estimates_complete",
            identity=did,
            clips=15569,
            cells=23,
            files={
                p.name: file_digest(p)
                for p in [
                    path,
                    out / "regression.csv",
                    out / "cdf_arrays.npz",
                    out / "macro_metrics.csv",
                    out / "dataset_metrics.csv",
                    out / "generator_metrics.csv",
                ]
            },
        ),
    )
    table = pd.read_csv(out / "macro_metrics.csv")
    print(
        table[table.variant.isin(["global_only", "d2_anchor", "pooled_d2", "ttr", "split"])][
            ["variant", "auc", "real_positive_ap"]
        ].to_string(index=False),
        flush=True,
    )


def pipeline(root, args):
    import fcntl

    c, out, spec, identity = context(root)
    lock = (out / "pipeline.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("已有方向研究调度器，拒绝重复启动")

    def stage(action, parallel=False, extra=()):
        commands = [
            [sys.executable, "-m", "evaluation.direction_study", action, "--root", str(root)]
        ]
        if parallel:
            commands = [commands[0] + ["--rank", str(i), "--world-size", "2"] for i in (0, 1)]
        commands = [cmd + list(extra) for cmd in commands]
        processes = [subprocess.Popen(cmd, cwd=root) for cmd in commands]
        paper_json(
            out / "status.json",
            dict(
                status="running", phase=action, pid=os.getpid(), children=[p.pid for p in processes]
            ),
        )
        try:
            codes = [p.wait() for p in processes]
        except BaseException:
            for p in processes:
                if p.poll() is None:
                    p.terminate()
            for p in processes:
                p.wait()
            raise
        if any(codes):
            raise RuntimeError(f"{action}失败：{codes}；保留已完成检查点")

    try:
        stage("fit_features", True)
        stage("fit")
        stage("dense", extra=("--limit", "2"))
        stage("dense", True)
        stage("evaluate")
        paper_json(
            out / "status.json",
            dict(
                status="scores_complete",
                pid=os.getpid(),
                next="paired intervals, independent audit and manuscript update",
            ),
        )
    except BaseException as exc:
        paper_json(out / "status.json", dict(status="failed", pid=os.getpid(), error=str(exc)))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["fit_features", "fit", "dense", "evaluate", "pipeline"])
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--role", choices=["cdf", "evaluation"])
    a = p.parse_args()
    if not 0 <= a.rank < a.world_size <= 2 or a.limit < 0:
        raise ValueError("分片/limit非法")
    torch.set_num_threads(2)
    globals()[{"fit_features": "extract_fit", "fit": "fit_models"}.get(a.action, a.action)](
        a.root.resolve(), a
    )


if __name__ == "__main__":
    main()
