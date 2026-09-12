"""新真实池及重编码子集的确定性清单；不读取检测分数来选择样本。"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from artifacts import atomic_csv, paper_json
from config import config_digest
from data.video import video_metadata, downsample_indices, decode_bounded
from reference import file_digest
from selection import uniform_windows

DOMAINS = ("comgenvid", "videofeedback", "genvideo")


def configuration(root):
    c = yaml.safe_load((root / "configs/reference_confirmation.yaml").read_text())
    if (
        c["fit_clips"] != 200
        or c["seeds"] != [17, 29, 43, 59, 71]
        or c["positions_per_clip"] != 256
    ):
        raise ValueError("预定参考预算/种子改变")
    if c["crf"] != [23, 35] or c["robust_observation"] != "fixed_original_windows":
        raise ValueError("预定扰动规则改变")
    return c


def source_group(domain, path):
    stem = Path(path).stem
    return domain + ":real-content:" + (stem[:11] if domain == "comgenvid" else stem)


def media_key(domain, path):
    stem = Path(path).stem
    if domain in ("comgenvid", "vatex"):
        return "youtube:" + (stem[:11] if domain == "comgenvid" else stem)
    return source_group(domain, path)


def order(seed, domain, path):
    return hashlib.sha256(f"{seed}:{domain}:{path}".encode()).hexdigest()


def context(root):
    c = configuration(root)
    out = root / c["output"]
    out.mkdir(parents=True, exist_ok=True)
    paths = [
        root / "configs/reference_confirmation.yaml",
        root / c["baseline"] / "evaluation.csv",
        root / c["baseline"] / "pairs.csv",
        root / c["plans"],
    ]
    paths += [root / f"data/manifests/active/{d}/fit.csv" for d in DOMAINS]
    paths += [root / f"data/manifests/active/vatex/{s}.csv" for s in ("cdf", "threshold")]
    spec = dict(
        config=c,
        inputs={str(p.relative_to(root)): file_digest(p) for p in paths},
        data_code=file_digest(Path(__file__)),
    )
    path = out / "data_identity.json"
    if path.exists() and json.loads(path.read_text()) != spec:
        previous = json.loads(path.read_text())
        if (
            (out / "prepared.json").exists()
            or previous["config"] != spec["config"]
            or previous["inputs"] != spec["inputs"]
        ):
            raise ValueError("数据规则或源清单改变，拒绝混合恢复")
        paper_json(
            out / "preparation_repair.json",
            dict(
                previous=previous,
                current=spec,
                reason="仅在清单尚未发布且输入/配置不变时修复准备代码",
            ),
        )
        paper_json(path, spec)
    if not path.exists():
        paper_json(path, spec)
    return c, out, spec


def prepare(root):
    c, out, spec = context(root)
    if (out / "prepared.json").exists():
        m = json.loads((out / "prepared.json").read_text())
        for p, h in m["files"].items():
            if file_digest(out / p) != h:
                raise ValueError("新冻结清单改变")
        return
    original = pd.read_csv(root / c["baseline"] / "evaluation.csv", keep_default_na=False)
    old_fit = [
        pd.read_csv(root / f"data/manifests/active/{d}/fit.csv", keep_default_na=False)
        for d in DOMAINS
    ]
    refs = [
        pd.read_csv(root / f"data/manifests/active/vatex/{s}.csv", keep_default_na=False)
        for s in ("cdf", "threshold")
    ]
    forbidden = pd.concat([original, *old_fit, *refs], ignore_index=True).fillna("")
    groups = set(forbidden.source_group) - {""}
    physical = {str((root / p).resolve()) for p in forbidden.video_path}
    media = {
        media_key(r.dataset, r.video_path) for r in forbidden.itertuples() if r.subset == "real"
    }
    candidates = {}
    excluded = []
    for d in DOMAINS:
        rows = []
        for p in sorted((root / f"datasets/{d}/real").rglob("*.mp4")):
            rel = str(p.relative_to(root))
            g = source_group(d, rel)
            if str(p.resolve()) in physical or g in groups or media_key(d, rel) in media:
                continue
            st = p.stat()
            rows.append(
                dict(
                    dataset=d,
                    video_path=rel,
                    source_group=g,
                    source_bytes=st.st_size,
                    source_mtime_ns=st.st_mtime_ns,
                )
            )
        if len(rows) < 200:
            raise ValueError(f"{d}剩余真实候选不足200")
        candidates[d] = rows
    # 各seed先独立按身份排列；无效视频按此顺序跳过，失败原因明确记录。
    inspected = {}
    selection = []

    def inspect(r):
        p = root / r["video_path"]
        key = config_digest(r)
        dest = out / "metadata" / (key + ".json")
        if dest.exists():
            return json.loads(dest.read_text())
        try:
            v = video_metadata(p)
            indices = downsample_indices(v["num_frames"], v["fps"])
            if len(indices) < 16:
                raise ValueError("不足16个互异8FPS帧")
            # 验证首/中/尾真实可解码；完整Uniform帧仍在提取时严格检查。
            decode_bounded(p, [indices[0], indices[len(indices) // 2], indices[-1]])
            value = dict(
                **r,
                **v,
                status="eligible",
                downsample_idxs=json.dumps(indices),
                subset="real",
                split="fit",
                source_model=p.parent.name,
                real_source=p.parent.name,
                video_id=r["dataset"] + ":" + r["video_path"],
                legacy_path=r["video_path"],
            )
        except Exception as e:
            value = dict(**r, status="excluded", reason=type(e).__name__ + ": " + str(e))
        paper_json(dest, value)
        return value

    with ThreadPoolExecutor(max_workers=c["metadata_workers"]) as pool:
        for d in DOMAINS:
            for seed in c["seeds"]:
                ranked = sorted(candidates[d], key=lambda r: order(seed, d, r["video_path"]))
                chosen = []
                for offset in range(0, len(ranked), 64):
                    batch = ranked[offset : offset + 64]
                    pending = [r for r in batch if r["video_path"] not in inspected]
                    for v in pool.map(inspect, pending):
                        inspected[v["video_path"]] = v
                        if v["status"] != "eligible":
                            excluded.append(v)
                    for r in batch:
                        v = inspected[r["video_path"]]
                        if v["status"] == "eligible" and len(chosen) < c["fit_clips"]:
                            chosen.append(v)
                    if len(chosen) == c["fit_clips"]:
                        break
                if len(chosen) != 200:
                    raise ValueError(f"{d}有效真实视频不足")
                selection.extend([dict(**v, fit_seed=seed) for v in chosen])
                print(
                    "new fit selection",
                    d,
                    seed,
                    len(chosen),
                    "clips",
                    len({v["source_group"] for v in chosen}),
                    "sources",
                    flush=True,
                )
    selected = pd.DataFrame(selection)
    unique = selected.drop_duplicates("video_id").drop(columns="fit_seed")
    if set(selected.source_group) & groups:
        raise ValueError("新fit与旧角色源组交叉")
    overlap = []
    for d in DOMAINS:
        for a in c["seeds"]:
            for b in c["seeds"]:
                x = selected[(selected.dataset == d) & (selected.fit_seed == a)]
                y = selected[(selected.dataset == d) & (selected.fit_seed == b)]
                overlap.append(
                    dict(
                        dataset=d,
                        seed_a=a,
                        seed_b=b,
                        shared_clips=len(set(x.video_id) & set(y.video_id)),
                        shared_sources=len(set(x.source_group) & set(y.source_group)),
                    )
                )
    atomic_csv(out / "fit_selections.csv", selected)
    atomic_csv(out / "fit_unique.csv", unique)
    atomic_csv(out / "fit_overlap.csv", pd.DataFrame(overlap))
    paper_json(out / "fit_exclusions.json", dict(rows=excluded))
    # 预定每生成器最多50对，按身份hash取样，重复使用的real仍共享source_group。
    pairs = pd.read_csv(root / c["baseline"] / "pairs.csv", keep_default_na=False)
    parts = []
    for (d, g), p in pairs.groupby(["dataset", "generator"]):
        count = min(
            c["robust_pairs_per_cell"],
            int(p.subset.eq("real").sum()),
            int(p.subset.eq("annotated").sum()),
        )
        for role in ("real", "annotated"):
            q = p[p.subset.eq(role)].copy()
            q["sort_key"] = q.video_id.map(lambda v: order(c["robust_seed"], d, v))
            parts.append(q.sort_values("sort_key").head(count).drop(columns="sort_key"))
    robust_pairs = pd.concat(parts)
    wanted = set(robust_pairs.video_id)
    atomic_csv(out / "robust_pairs.csv", robust_pairs)
    atomic_csv(out / "robust_evaluation.csv", original[original.video_id.isin(wanted)])
    jobs = json.loads((root / c["plans"]).read_text())["jobs"]
    robust = [
        j
        for j in jobs
        if j["role"] == "evaluation" and any(m["video_id"] in wanted for m in j["targets"].values())
    ]
    paper_json(out / "robust_jobs.json", dict(jobs=robust))
    summary = [
        dict(dataset=d, seed=int(s), clips=len(f), sources=int(f.source_group.nunique()))
        for (d, s), f in selected.groupby(["dataset", "fit_seed"])
    ]
    paper_json(
        out / "prepared.json",
        dict(
            status="prepared",
            identity=config_digest(spec),
            fit=summary,
            unique_fit_clips=len(unique),
            forbidden_source_intersection=0,
            independence="old fit/evaluation/CDF/threshold source IDs and physical paths; known YouTube IDs; not semantic deduplication",
            pools_mutually_disjoint=False,
            robust_cells=robust_pairs[["dataset", "generator"]].drop_duplicates().shape[0],
            robust_unique_clips=len(wanted),
            files={
                p: file_digest(out / p)
                for p in (
                    "fit_selections.csv",
                    "fit_unique.csv",
                    "fit_overlap.csv",
                    "fit_exclusions.json",
                    "robust_pairs.csv",
                    "robust_evaluation.csv",
                    "robust_jobs.json",
                )
            },
        ),
    )


if __name__ == "__main__":
    prepare(Path.cwd())
