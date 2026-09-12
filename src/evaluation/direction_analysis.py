"""当前方向研究的配对分析；共享源权重，预排序精确处理AUC/AP并列。"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from artifacts import atomic_csv, paper_json
from config import config_digest
from reference import file_digest
from evaluation.tables import evaluate_fixed_pairs


class RankedMetrics:
    """预排序后仅改变源权重，与sklearn完整阈值AUC/AP定义一致。"""

    def __init__(self, y, score):
        y = np.asarray(y, dtype=bool)
        score = np.asarray(score, dtype=np.float64)
        if y.shape != score.shape or not np.isfinite(score).all():
            raise ValueError("标签或分数非法")
        self.order = np.argsort(-score, kind="stable")
        self.y = y[self.order]
        ordered = score[self.order]
        self.ends = np.r_[np.where(np.diff(ordered) != 0)[0], len(ordered) - 1]

    def __call__(self, weights):
        w = np.asarray(weights, dtype=np.float64)[self.order]
        tp = np.cumsum(w * self.y)[self.ends]
        fp = np.cumsum(w * ~self.y)[self.ends]
        if tp[-1] <= 0 or fp[-1] <= 0:
            raise ValueError("加权后缺少类别")
        dt = np.diff(np.r_[0, tp])
        df = np.diff(np.r_[0, fp])
        auc = np.sum(df * (2 * tp - dt)) / (2 * tp[-1] * fp[-1])
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        ap = np.sum(dt * precision) / tp[-1]
        return np.array([auc, ap])


def shared_contrasts(frames, pairs, groups, contrasts, iterations=1000, seed=17):
    """每个contrast为{variant:系数}；四格交互在同一次源抽样内计算。"""
    names = list(frames)
    indexed = {k: v.set_index("video_id", verify_integrity=True) for k, v in frames.items()}
    anchor = indexed[names[0]]
    for name, frame in indexed.items():
        if set(frame.index) != set(anchor.index):
            raise ValueError("候选不能自动取身份交集")
        for col in ("dataset", "subset", "source_model"):
            if not frame.loc[anchor.index, col].equals(anchor[col]):
                raise ValueError("候选来源或标签不同")
    coefficient = np.array([[c.get(n, 0) for n in names] for c in contrasts.values()])
    if not np.allclose(coefficient.sum(1), 0):
        raise ValueError("差值系数应和为零")
    point_rows = []
    draws_all = {}
    point_all = {}
    n_cells = {}
    for d, part in pairs.groupby("dataset"):
        ids = anchor.index[anchor.dataset.eq(d)]
        gs = groups.loc[ids]
        if gs.isna().any() or gs.eq("").any():
            raise ValueError("缺少源组")
        mapping = {g: i for i, g in enumerate(sorted(set(gs)))}
        cells = []
        for _, p in part.groupby("generator"):
            y = anchor.loc[p.video_id, "subset"].eq("real").to_numpy()
            if y.sum() != len(y) // 2 or y.sum() * 2 != len(y):
                raise ValueError("单元不平衡")
            if not (anchor.loc[p.video_id, "subset"].to_numpy() == p.subset.to_numpy()).all():
                raise ValueError("配对标签改变")
            idx = np.array([mapping[g] for g in groups.loc[p.video_id]])
            funcs = [RankedMetrics(y, indexed[n].loc[p.video_id, "final_score"]) for n in names]
            cells.append((idx, y, funcs))
        raw_point = np.mean(
            [[f(np.ones(len(idx))) for f in funcs] for idx, y, funcs in cells], axis=0
        )
        point = coefficient @ raw_point
        point_all[d] = point
        n_cells[d] = len(cells)
        tag = int.from_bytes(hashlib.sha256(d.encode()).digest()[:4], "little")
        rng = np.random.default_rng(np.random.SeedSequence([seed, tag]))
        draws = []
        attempts = 0
        while len(draws) < iterations:
            attempts += 1
            if attempts > iterations * 100 + 100:
                raise RuntimeError("有效重抽样不足")
            w = rng.poisson(1, len(mapping))
            values = []
            for idx, y, funcs in cells:
                weight = w[idx]
                if not weight[y].sum() or not weight[~y].sum():
                    break
                values.append([f(weight) for f in funcs])
            if len(values) == len(cells):
                draws.append(coefficient @ np.mean(values, axis=0))
        draws_all[d] = np.array(draws)
        print("shared bootstrap", d, iterations, "draws", flush=True)
    dev = ("comgenvid", "genvideo", "videofeedback")
    if set(dev).issubset(draws_all):
        draws_all["Average"] = np.mean([draws_all[d] for d in dev], axis=0)
        point_all["Average"] = np.mean([point_all[d] for d in dev], axis=0)
    for d, draws in draws_all.items():
        for i, name in enumerate(contrasts):
            for k, metric in enumerate(("auc", "ap_real")):
                point_rows.append(
                    dict(
                        dataset=d,
                        contrast=name,
                        metric=metric,
                        delta=point_all[d][i, k],
                        ci95_low=np.quantile(draws[:, i, k], 0.025),
                        ci95_high=np.quantile(draws[:, i, k], 0.975),
                    )
                )
    return pd.DataFrame(point_rows)


def observation(root):
    base = root / "results/paper_complete"
    out = root / "results/runs/local_direction_evidence/observation"
    names = ["scores_fc3.csv.gz", "scores_uniform3.csv.gz", "evaluation.csv", "pairs.csv"]
    identity = dict(
        inputs={n: file_digest(base / n) for n in names},
        code=file_digest(Path(__file__)),
        iterations=1000,
        seed=17,
    )
    marker = out / "manifest.json"
    if marker.exists():
        m = json.loads(marker.read_text())
        if m["identity"] != identity:
            raise ValueError("观察分析来源改变")
        return
    frames = {}
    tables = []
    pairs = pd.read_csv(base / "pairs.csv")
    meta = pd.read_csv(base / "evaluation.csv", keep_default_na=False)
    for mode in ("fc3", "uniform3"):
        s = pd.read_csv(base / f"scores_{mode}.csv.gz", float_precision="round_trip")
        for v in ("full", "global_only"):
            name = mode + "__" + v
            frames[name] = s[s.variant.eq(v)]
            t = evaluate_fixed_pairs(frames[name], pairs)
            tables.append(
                pd.concat(
                    [t["dataset_metrics"], t["macro_metrics"].rename(columns={"scope": "dataset"})]
                ).assign(variant=name)
            )
    contrasts = {
        "Local_increment_Uniform": {"uniform3__full": 1, "uniform3__global_only": -1},
        "Local_increment_FC": {"fc3__full": 1, "fc3__global_only": -1},
        "observation_Local_interaction": {
            "fc3__full": 1,
            "fc3__global_only": -1,
            "uniform3__full": -1,
            "uniform3__global_only": 1,
        },
    }
    start = time.perf_counter()
    result = shared_contrasts(frames, pairs, meta.set_index("video_id").source_group, contrasts)
    atomic_csv(out / "metrics.csv", pd.concat(tables).replace({"dataset": {"Macro-3": "Average"}}))
    atomic_csv(out / "contrasts.csv", result)
    paper_json(
        marker,
        dict(
            status="completed",
            identity=identity,
            seconds=time.perf_counter() - start,
            files={p: file_digest(out / p) for p in ("metrics.csv", "contrasts.csv")},
        ),
    )
    print(result[result.dataset.eq("Average")].to_string(index=False), flush=True)


def representations(root):
    out = root / "results/runs/local_direction_evidence"
    m = json.loads((out / "evaluation.json").read_text())
    for name, h in m["files"].items():
        if file_digest(out / name) != h:
            raise ValueError("点结果改变")
    s = pd.read_csv(out / "video_scores.csv.gz", float_precision="round_trip")
    base = root / "results/paper_complete"
    pairs = pd.read_csv(base / "pairs.csv")
    meta = pd.read_csv(base / "evaluation.csv", keep_default_na=False)
    contrast = {
        "D2_vs_SPLIT": {"d2_anchor": 1, "split": -1},
        "D2_vs_TTR": {"d2_anchor": 1, "ttr": -1},
        "D2_vs_pooled": {"d2_anchor": 1, "pooled_d2": -1},
        "D2_vs_Global": {"d2_anchor": 1, "global_only": -1},
        "D2raw_vs_SPLITraw": {"d2_anchor_raw": 1, "split_raw": -1},
        "D2raw_vs_TTRraw": {"d2_anchor_raw": 1, "ttr_raw": -1},
        "D2raw_vs_pooledraw": {"d2_anchor_raw": 1, "pooled_d2_raw": -1},
        "source_balanced_vs_clip": {"d2_source_balanced": 1, "d2_anchor": -1},
    }
    names = {n for c in contrast.values() for n in c}
    frames = {n: s[s.variant.eq(n)] for n in sorted(names)}
    result = shared_contrasts(frames, pairs, meta.set_index("video_id").source_group, contrast)
    atomic_csv(out / "contrasts.csv", result)
    paper_json(
        out / "analysis.json",
        dict(
            status="completed",
            source=file_digest(out / "video_scores.csv.gz"),
            code=file_digest(Path(__file__)),
            iterations=1000,
            seed=17,
            contrasts=contrast,
            sha256=file_digest(out / "contrasts.csv"),
        ),
    )
    print(result[result.dataset.eq("Average")].to_string(index=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["observation", "representations"])
    p.add_argument("--root", type=Path, default=Path.cwd())
    a = p.parse_args()
    globals()[a.action](a.root.resolve())
