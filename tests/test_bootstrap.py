"""源组联合权重、同分零差值和禁止交集缩小的统计合同。"""

from pathlib import Path
import sys
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation.bootstrap import paired_source_contrast


def fixture():
    rows = []
    pairs = []
    for domain in ("comgenvid", "videofeedback", "genvideo"):
        for i in range(4):
            vid = f"{domain}:{i}"
            subset = "real" if i < 2 else "annotated"
            rows.append(
                dict(
                    video_id=vid,
                    dataset=domain,
                    subset=subset,
                    source_model="real" if i < 2 else "g",
                    final_score=float(4 - i),
                )
            )
            pairs.append(dict(video_id=vid, dataset=domain, subset=subset, generator="g"))
    scores = pd.DataFrame(rows)
    # 每个内容组同时包含一个real和一个fake，验证不是逐行独立重抽样。
    groups = pd.Series(
        {
            r["video_id"]: r["dataset"] + ":" + str(int(r["video_id"].split(":")[1]) % 2)
            for r in rows
        }
    )
    return scores, pd.DataFrame(pairs), groups


def test_identical_scores_have_zero_interval():
    scores, pairs, groups = fixture()
    delta, interval = paired_source_contrast(scores, scores, pairs, groups, seed=17, iterations=30)
    assert "Macro-3" in set(delta.dataset)
    assert delta.delta.eq(0).all()
    assert interval.ci95_low.eq(0).all() and interval.ci95_high.eq(0).all()


def test_missing_and_relabeled_scores_rejected():
    scores, pairs, groups = fixture()
    with pytest.raises(ValueError, match="集合不同"):
        paired_source_contrast(scores, scores.iloc[1:], pairs, groups, iterations=10)
    changed = scores.copy()
    changed.loc[0, "subset"] = "annotated"
    with pytest.raises(ValueError, match="标签"):
        paired_source_contrast(scores, changed, pairs, groups, iterations=10)


def test_generator_average_weights_domains_by_cell_count():
    scores, pairs, groups = fixture()
    extra = scores[(scores.dataset == "genvideo") & (scores.subset == "annotated")].copy()
    extra["video_id"] = extra.video_id + ":g2"
    extra["source_model"] = "g2"
    extra_pairs = pd.concat(
        [
            pairs[(pairs.dataset == "genvideo") & (pairs.subset == "real")],
            extra[["video_id", "dataset", "subset"]],
        ],
        ignore_index=True,
    ).assign(generator="g2")
    scores = pd.concat([scores, extra], ignore_index=True)
    pairs = pd.concat([pairs, extra_pairs], ignore_index=True)
    groups = pd.concat([groups, pd.Series({r.video_id: r.video_id for r in extra.itertuples()})])
    baseline = scores.copy()
    mask = baseline.dataset == "genvideo"
    baseline.loc[mask, "final_score"] = -baseline.loc[mask, "final_score"]
    delta, _ = paired_source_contrast(
        scores, baseline, pairs, groups, iterations=10, include_generator_average=True
    )
    assert delta.query("dataset=='Average' and metric=='auc'").delta.iloc[0] == pytest.approx(0.5)
    assert delta.query("dataset=='Macro-3' and metric=='auc'").delta.iloc[0] == pytest.approx(1 / 3)
