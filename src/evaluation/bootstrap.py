"""基于视频样本的成对 bootstrap 统计入口。"""

from __future__ import annotations

import pandas as pd
import hashlib
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def paired_source_contrast(
    candidate,
    baseline,
    pairs,
    source_groups,
    *,
    seed=17,
    iterations=1000,
    progress=None,
    include_generator_average=False,
):
    """固定配对、域内源组Poisson权重；共享real及ViF内容组在各生成器同步加权。"""
    required = {"video_id", "dataset", "subset", "source_model", "final_score"}
    for frame in (candidate, baseline):
        if required - set(frame) or frame.video_id.duplicated().any():
            raise ValueError("评分字段缺失或身份重复")
        if not set(frame.subset).issubset({"real", "annotated"}):
            raise ValueError("评分标签无效")
        if not np.isfinite(frame.final_score.to_numpy(dtype=float)).all():
            raise ValueError("分数非有限")
    if set(candidate.video_id) != set(baseline.video_id):
        raise ValueError("候选与基线视频集合不同，禁止自动取交集")
    candidate = candidate.set_index("video_id", verify_integrity=True)
    baseline = baseline.set_index("video_id", verify_integrity=True).loc[candidate.index]
    for column in ("dataset", "subset", "source_model"):
        if not candidate[column].equals(baseline[column]):
            raise ValueError("候选与基线标签/来源不一致")
    if iterations < 1 or seed < 0:
        raise ValueError("bootstrap次数/seed无效")
    if source_groups.index.duplicated().any() or not set(candidate.index).issubset(
        source_groups.index
    ):
        raise ValueError("源组身份缺失或重复")
    required_pairs = {"video_id", "dataset", "subset", "generator"}
    if required_pairs - set(pairs) or pairs.duplicated(["dataset", "generator", "video_id"]).any():
        raise ValueError("配对表字段或身份错误")
    if not set(pairs.video_id).issubset(candidate.index):
        raise ValueError("配对视频没有评分")
    points = []
    intervals = []
    draws_by_domain = {}

    def metric(y, s, w=None):
        return np.array(
            [roc_auc_score(y, s, sample_weight=w), average_precision_score(y, s, sample_weight=w)]
        )

    for domain, domain_pairs in pairs.groupby("dataset"):
        ids = candidate.index[candidate.dataset == domain]
        domain_groups = source_groups.loc[ids]
        if domain_groups.isna().any() or domain_groups.astype(str).str.len().eq(0).any():
            raise ValueError("源组为空")
        mapping = {g: i for i, g in enumerate(sorted(set(domain_groups)))}
        prepared = []
        for generator, pair in domain_pairs.groupby("generator"):
            c = candidate.loc[pair.video_id]
            b = baseline.loc[pair.video_id]
            if (
                not c.dataset.eq(domain).all()
                or not (c.subset.to_numpy() == pair.subset.to_numpy()).all()
            ):
                raise ValueError("配对标签/域改变")
            y = c.subset.eq("real").to_numpy()
            if not y.any() or not (~y).any() or y.sum() != (~y).sum():
                raise ValueError("配对必须平衡且包含两类")
            if not c.loc[~y, "source_model"].eq(generator).all():
                raise ValueError("配对生成器错误")
            idx = np.asarray([mapping[g] for g in source_groups.loc[pair.video_id]])
            prepared.append((y, c.final_score.to_numpy(), b.final_score.to_numpy(), idx))
        point = np.mean([metric(y, c) - metric(y, b) for y, c, b, _ in prepared], axis=0)
        tag = int.from_bytes(hashlib.sha256(domain.encode()).digest()[:4], "little")
        rng = np.random.default_rng(np.random.SeedSequence([seed, tag]))
        draws = []
        attempts = 0
        while len(draws) < iterations:
            attempts += 1
            if attempts > iterations * 100 + 100:
                raise RuntimeError("可用bootstrap抽样不足")
            weights = rng.poisson(1, len(mapping))
            values = []
            for y, c, b, idx in prepared:
                w = weights[idx]
                if not w[y].sum() or not w[~y].sum():
                    break
                values.append(metric(y, c, w) - metric(y, b, w))
            if len(values) == len(prepared):
                draws.append(np.mean(values, axis=0))
        draws = np.asarray(draws)
        draws_by_domain[domain] = draws
        for i, name in enumerate(("auc", "ap_real")):
            points.append(
                dict(
                    dataset=domain,
                    metric=name,
                    delta=float(point[i]),
                    source_groups=len(mapping),
                    generator_cells=len(prepared),
                )
            )
            intervals.append(
                dict(
                    dataset=domain,
                    metric=name,
                    ci95_low=float(np.quantile(draws[:, i], 0.025)),
                    ci95_high=float(np.quantile(draws[:, i], 0.975)),
                )
            )
        if progress:
            progress(domain)
    dev = ("comgenvid", "videofeedback", "genvideo")
    if set(dev).issubset(draws_by_domain):
        draws = np.mean([draws_by_domain[d] for d in dev], axis=0)
        for i, name in enumerate(("auc", "ap_real")):
            delta = np.mean(
                [r["delta"] for r in points if r["dataset"] in dev and r["metric"] == name]
            )
            points.append(dict(dataset="Macro-3", metric=name, delta=float(delta)))
            intervals.append(
                dict(
                    dataset="Macro-3",
                    metric=name,
                    ci95_low=float(np.quantile(draws[:, i], 0.025)),
                    ci95_high=float(np.quantile(draws[:, i], 0.975)),
                )
            )
    if include_generator_average and set(dev).issubset(draws_by_domain):
        counts = np.asarray([pairs.loc[pairs.dataset.eq(d), "generator"].nunique() for d in dev])
        draws = np.average(np.stack([draws_by_domain[d] for d in dev]), axis=0, weights=counts)
        for i, name in enumerate(("auc", "ap_real")):
            values = [
                next(r["delta"] for r in points if r["dataset"] == d and r["metric"] == name)
                for d in dev
            ]
            points.append(
                dict(
                    dataset="Average", metric=name, delta=float(np.average(values, weights=counts))
                )
            )
            intervals.append(
                dict(
                    dataset="Average",
                    metric=name,
                    ci95_low=float(np.quantile(draws[:, i], 0.025)),
                    ci95_high=float(np.quantile(draws[:, i], 0.975)),
                )
            )
    return pd.DataFrame(points), pd.DataFrame(intervals)
