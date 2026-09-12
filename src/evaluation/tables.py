"""将规范化逐视频分数转换为诊断表与论文口径指标表。"""

from __future__ import annotations

import pandas as pd
import numpy as np

from .metrics import binary_metrics


REQUIRED_SCORE_COLUMNS = {"video_id", "dataset", "subset", "source_model", "final_score"}


def component_ablation_tables(scores, pairs):
    """固定目标参考与窗口，只移除证据；不扫描权重，不重建任一CDF。"""
    scores = normalize_scores(scores)
    fields = [
        "global_score",
        "global_spatial_score",
        "global_temporal_score",
        "local_score",
        "final_score",
    ]
    if set(fields) - set(scores):
        raise ValueError("组件重算需要完整分支分数")
    values = scores[fields].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("组件分数必须有限且在[0,1]")
    expected = 0.5 * scores.global_score + 0.5 * scores.local_score
    if not np.array_equal(expected.to_numpy(), scores.final_score.to_numpy()):
        raise ValueError("源分数不是固定等权完整模型")
    variants = dict(
        full=scores.final_score,
        global_only=scores.global_score,
        local_only=scores.local_score,
        without_gs=0.5 * scores.global_temporal_score + 0.5 * scores.local_score,
        without_gt=0.5 * scores.global_spatial_score + 0.5 * scores.local_score,
    )
    tables = {}
    videos = []
    for name, final in variants.items():
        frame = scores.copy()
        frame["final_score"] = final
        videos.append(frame.assign(variant=name))
        for key, table in evaluate_fixed_pairs(frame, pairs).items():
            tables.setdefault(key, []).append(table.assign(variant=name))
    output = {key: pd.concat(parts, ignore_index=True) for key, parts in tables.items()}
    output["video_scores"] = pd.concat(videos, ignore_index=True)
    for key, keys in [
        ("generator_metrics", ["dataset", "generator"]),
        ("dataset_metrics", ["dataset"]),
        ("macro_metrics", ["scope"]),
    ]:
        table = output[key]
        if table.empty:
            continue
        metrics = [col for col in table if col not in (*keys, "variant", "n_real", "n_fake")]
        baseline = table[table.variant == "full"].set_index(keys)[metrics]
        rows = []
        for name, part in table.groupby("variant", sort=False):
            delta = part.set_index(keys)[metrics] - baseline
            rows.append(
                delta.rename(columns={metric: "delta_" + metric for metric in metrics})
                .reset_index()
                .assign(variant=name)
            )
        output[key.replace("_metrics", "_deltas")] = pd.concat(rows, ignore_index=True)
    return output


def evaluate_fixed_pairs(scores: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """新入口只消费冻结配对，不在评价阶段重新抽样或平衡类别。"""
    scores = normalize_scores(scores)
    required = {"video_id", "dataset", "subset", "generator"}
    if required - set(pairs):
        raise ValueError("配对表缺少必要字段")
    if pairs.duplicated(["dataset", "generator", "video_id"]).any():
        raise ValueError("单元内视频重复")
    indexed = scores.set_index("video_id", verify_integrity=True)
    rows = []
    for (domain, generator), pair in pairs.groupby(["dataset", "generator"]):
        if not set(pair.video_id).issubset(indexed.index):
            raise ValueError("配对视频缺少评分")
        selected = indexed.loc[pair.video_id].reset_index()
        if (
            not (selected.dataset.to_numpy() == pair.dataset.to_numpy()).all()
            or not (selected.subset.to_numpy() == pair.subset.to_numpy()).all()
        ):
            raise ValueError("配对域或标签漂移")
        fake = selected.subset.eq("annotated")
        if not selected.loc[fake, "source_model"].eq(generator).all():
            raise ValueError("生成器归属错误")
        real_count = int((~fake).sum())
        fake_count = int(fake.sum())
        if real_count != fake_count or not real_count:
            raise ValueError("主配对单元必须平衡且非空")
        rows.append(
            dict(
                dataset=domain,
                generator=generator,
                n_real=real_count,
                n_fake=fake_count,
                **binary_metrics(selected),
            )
        )
    if not rows:
        raise ValueError("配对表为空")
    generators = pd.DataFrame(rows)
    metrics = [c for c in generators if c not in ("dataset", "generator", "n_real", "n_fake")]
    datasets = generators.groupby("dataset", as_index=False)[metrics].mean()
    dev = ["comgenvid", "videofeedback", "genvideo"]
    macro = pd.DataFrame(columns=["scope", *metrics])
    if set(dev).issubset(datasets.dataset):
        macro = pd.DataFrame(
            [
                dict(
                    scope="Macro-3",
                    **datasets[datasets.dataset.isin(dev)][metrics].mean().to_dict(),
                )
            ]
        )
    population = []
    for domain, group in scores.groupby("dataset"):
        population.append(
            dict(
                dataset=domain,
                n_real=int(group.subset.eq("real").sum()),
                n_fake=int(group.subset.eq("annotated").sum()),
                real_prevalence=float(group.subset.eq("real").mean()),
                **binary_metrics(group),
            )
        )
    return dict(
        generator_metrics=generators,
        dataset_metrics=datasets,
        macro_metrics=macro,
        full_population_metrics=pd.DataFrame(population),
    )


def normalize_scores(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_SCORE_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"逐视频分数 CSV 缺少字段：{sorted(missing)}")
    result = frame.copy()
    result["video_id"] = result["video_id"].astype(str)
    if result["video_id"].duplicated().any():
        raise ValueError("逐视频分数 CSV 含有重复 video_id")
    if not set(result["subset"].unique()).issubset({"real", "annotated"}):
        raise ValueError("subset 字段只能使用 real 或 annotated")
    result["final_score"] = pd.to_numeric(result["final_score"], errors="raise")
    return result
