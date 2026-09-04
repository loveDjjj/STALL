"""CAES selector 间的严格视频对齐、配对 bootstrap 与预注册门槛。"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from evaluation.metrics import binary_metrics
from evaluation.tables import _balanced_real_pair


IDENTITY_COLUMNS = ("video_id", "dataset", "subset", "source_model", "video_path")


def align_selector_scores(
    scores: pd.DataFrame,
    *,
    baseline: str = "uniform",
) -> pd.DataFrame:
    """把 long-form selector 分数按完全相同的视频身份对齐成宽表。"""

    required = {*IDENTITY_COLUMNS, "selector", "final_score"}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"CAES逐视频分数缺少字段：{sorted(missing)}")
    selectors = list(dict.fromkeys(scores["selector"].astype(str)))
    if baseline not in selectors:
        raise ValueError(f"CAES分数缺少baseline selector：{baseline}")
    selectors = [baseline, *[item for item in selectors if item != baseline]]
    aligned = None
    expected_ids = None
    for selector in selectors:
        frame = scores[scores["selector"].eq(selector)].copy()
        if frame["video_id"].duplicated().any():
            raise ValueError(f"selector={selector}包含重复video_id")
        ids = set(frame["video_id"].astype(str))
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError(
                f"selector={selector}与baseline视频身份不一致："
                f"expected={len(expected_ids)}, actual={len(ids)}"
            )
        selected = frame[[*IDENTITY_COLUMNS, "final_score"]].rename(
            columns={"final_score": f"final_score__{selector}"}
        )
        aligned = selected if aligned is None else aligned.merge(
            selected,
            on=list(IDENTITY_COLUMNS),
            how="inner",
            validate="one_to_one",
        )
    if aligned is None or expected_ids is None or len(aligned) != len(expected_ids):
        raise ValueError("CAES selector视频身份对齐失败")
    return aligned


def build_matched_selector_pairwise_table(
    scores: pd.DataFrame,
    *,
    baseline: str = "uniform",
    seed: int = 42,
) -> pd.DataFrame:
    """仅从FS0构造一次配对身份，并对全部selector复用同一视频集合。"""

    aligned = align_selector_scores(scores, baseline=baseline)
    selectors = [
        column.removeprefix("final_score__")
        for column in aligned.columns
        if column.startswith("final_score__")
    ]
    rows: list[dict[str, object]] = []
    for dataset, dataset_frame in aligned.groupby("dataset", sort=True):
        real = dataset_frame[dataset_frame["subset"].eq("real")]
        pairs = [
            _balanced_real_pair(real, fake, seed)
            for _, fake in dataset_frame[
                dataset_frame["subset"].eq("annotated")
            ].groupby("source_model", sort=True)
        ]
        if not pairs:
            raise ValueError(f"{dataset}没有可用于CAES配对指标的生成器")
        for selector in selectors:
            metrics = [
                binary_metrics(pair, f"final_score__{selector}") for pair in pairs
            ]
            rows.append({
                "selector": selector,
                "run_name": selector,
                "scope": "generator_pairwise_dataset_macro",
                "dataset": dataset,
                "auc": float(np.mean([item["auc"] for item in metrics])),
                "ap_real": float(
                    np.mean([item["real_positive_ap"] for item in metrics])
                ),
                "n_generators": len(pairs),
                "n_pairwise_real": int(sum(
                    pair["subset"].eq("real").sum() for pair in pairs
                )),
                "n_pairwise_fake": int(sum(
                    pair["subset"].eq("annotated").sum() for pair in pairs
                )),
                "pairwise_seed": seed,
                "pair_identity_source": baseline,
            })
    table = pd.DataFrame(rows)
    macro_rows = []
    for selector in selectors:
        selected = table[table["selector"].eq(selector)]
        macro_rows.append({
            "selector": selector,
            "run_name": selector,
            "scope": "generator_pairwise_macro3",
            "dataset": "Macro-3",
            "auc": float(selected["auc"].mean()),
            "ap_real": float(selected["ap_real"].mean()),
            "n_generators": int(selected["n_generators"].sum()),
            "n_pairwise_real": int(selected["n_pairwise_real"].sum()),
            "n_pairwise_fake": int(selected["n_pairwise_fake"].sum()),
            "pairwise_seed": seed,
            "pair_identity_source": baseline,
        })
    return pd.concat([table, pd.DataFrame(macro_rows)], ignore_index=True)


def _seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256(
        "\0".join((str(seed), *parts)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def paired_selector_bootstrap(
    scores: pd.DataFrame,
    *,
    baseline: str = "uniform",
    seed: int = 42,
    iterations: int = 1000,
) -> pd.DataFrame:
    """按论文生成器平衡规则计算各selector相对FS0的AUC/AP区间。"""

    if iterations < 1:
        raise ValueError("bootstrap iterations必须为正数")
    aligned = align_selector_scores(scores, baseline=baseline)
    selectors = [
        column.removeprefix("final_score__")
        for column in aligned.columns
        if column.startswith("final_score__")
        and column != f"final_score__{baseline}"
    ]
    rows: list[dict[str, object]] = []
    macro_samples: dict[tuple[str, str], list[np.ndarray]] = {}
    for dataset, dataset_frame in aligned.groupby("dataset", sort=True):
        real = dataset_frame[dataset_frame["subset"].eq("real")]
        pairs = [
            _balanced_real_pair(real, fake, seed)
            for _, fake in dataset_frame[
                dataset_frame["subset"].eq("annotated")
            ].groupby("source_model", sort=True)
        ]
        if not pairs:
            raise ValueError(f"{dataset}没有可用于CAES bootstrap的生成器")
        for selector in selectors:
            candidate_column = f"final_score__{selector}"
            baseline_column = f"final_score__{baseline}"
            rng = np.random.default_rng(_seed(seed, dataset, selector, baseline))
            values = {
                "auc": np.empty(iterations, dtype=np.float64),
                "ap_real": np.empty(iterations, dtype=np.float64),
            }
            for iteration in range(iterations):
                deltas = {"auc": [], "ap_real": []}
                for pair in pairs:
                    pair_real = pair[pair["subset"].eq("real")]
                    pair_fake = pair[pair["subset"].eq("annotated")]
                    # 分层重采样保证每轮同时含real/fake；同一行上的所有selector
                    # 字段仍一起抽取，因此候选与FS0保持严格paired。
                    sampled = pd.concat([
                        pair_real.iloc[
                            rng.integers(0, len(pair_real), len(pair_real))
                        ],
                        pair_fake.iloc[
                            rng.integers(0, len(pair_fake), len(pair_fake))
                        ],
                    ], ignore_index=True)
                    candidate = binary_metrics(sampled, candidate_column)
                    reference = binary_metrics(sampled, baseline_column)
                    deltas["auc"].append(candidate["auc"] - reference["auc"])
                    deltas["ap_real"].append(
                        candidate["real_positive_ap"]
                        - reference["real_positive_ap"]
                    )
                for metric in values:
                    values[metric][iteration] = float(np.mean(deltas[metric]))
            for metric, samples in values.items():
                macro_samples.setdefault((selector, metric), []).append(samples)
                rows.append({
                    "selector": selector,
                    "baseline": baseline,
                    "scope": "generator_pairwise_dataset_macro",
                    "dataset": dataset,
                    "metric": metric,
                    "delta_mean": float(samples.mean()),
                    "ci95_low": float(np.quantile(samples, 0.025)),
                    "ci95_high": float(np.quantile(samples, 0.975)),
                    "bootstrap_iterations": iterations,
                })
    for (selector, metric), per_dataset in macro_samples.items():
        samples = np.mean(np.stack(per_dataset), axis=0)
        rows.append({
            "selector": selector,
            "baseline": baseline,
            "scope": "generator_pairwise_macro3",
            "dataset": "Macro-3",
            "metric": metric,
            "delta_mean": float(samples.mean()),
            "ci95_low": float(np.quantile(samples, 0.025)),
            "ci95_high": float(np.quantile(samples, 0.975)),
            "bootstrap_iterations": iterations,
        })
    return pd.DataFrame(rows)


def evaluate_selector_gate(
    pairwise_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    *,
    baseline: str = "uniform",
    minimum_delta: float = 0.005,
    required_datasets: int = 2,
) -> pd.DataFrame:
    """执行预注册Go门槛；显著性只读取对应Macro bootstrap区间。"""

    baseline_rows = pairwise_metrics[
        pairwise_metrics["selector"].eq(baseline)
        & pairwise_metrics["scope"].eq("generator_pairwise_dataset_macro")
    ].set_index("dataset")
    macro = pairwise_metrics[
        pairwise_metrics["scope"].eq("generator_pairwise_macro3")
    ].set_index("selector")
    outputs = []
    for selector in pairwise_metrics["selector"].drop_duplicates():
        if selector == baseline:
            continue
        candidate_rows = pairwise_metrics[
            pairwise_metrics["selector"].eq(selector)
            & pairwise_metrics["scope"].eq("generator_pairwise_dataset_macro")
        ].set_index("dataset")
        if set(candidate_rows.index) != set(baseline_rows.index):
            raise ValueError(f"selector={selector}的dataset集合与FS0不一致")
        for metric in ("auc", "ap_real"):
            macro_delta = float(macro.loc[selector, metric] - macro.loc[baseline, metric])
            improvements = int(
                (candidate_rows.loc[baseline_rows.index, metric] > baseline_rows[metric]).sum()
            )
            interval = bootstrap[
                bootstrap["selector"].eq(selector)
                & bootstrap["dataset"].eq("Macro-3")
                & bootstrap["metric"].eq(metric)
            ]
            if len(interval) != 1:
                raise ValueError(f"selector={selector}/{metric}缺少唯一Macro bootstrap")
            ci_low = float(interval.iloc[0]["ci95_low"])
            ci_high = float(interval.iloc[0]["ci95_high"])
            stable = ci_low > 0.0
            outputs.append({
                "selector": selector,
                "metric": metric,
                "macro_delta_vs_uniform": macro_delta,
                "datasets_improved": improvements,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "bootstrap_direction_stable": stable,
                "passes_go_gate": bool(
                    macro_delta >= minimum_delta
                    and improvements >= required_datasets
                    and stable
                ),
            })
    return pd.DataFrame(outputs)


__all__ = [
    "IDENTITY_COLUMNS",
    "align_selector_scores",
    "build_matched_selector_pairwise_table",
    "evaluate_selector_gate",
    "paired_selector_bootstrap",
]
