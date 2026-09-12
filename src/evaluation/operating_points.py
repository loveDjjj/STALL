"""独立真实阈值迁移，与评价ROC操作点严格分开。"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from artifacts import paper_json
from reference import file_digest
from evaluation.study_tables import read_evidence, calibrated_scores


def real_threshold(scores, fpr):
    scores = np.sort(np.asarray(scores, dtype=float))
    if len(scores) < 2 or not np.isfinite(scores).all() or not 0 < fpr < 1:
        raise ValueError("阈值输入非法")
    # 判fake为S<tau；并列分数不拆分，校准误报不超过预算。
    return float(scores[int(np.floor(len(scores) * fpr))])


def wilson(successes, total):
    if not total:
        raise ValueError("操作点分母为零")
    z = 1.959963984540054
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return center - half, center + half


def threshold_tables(root, domain, threshold_dir, cdf_dir, evaluation_tables, output):
    root = Path(root)
    output = Path(output)
    evaluation_tables = Path(evaluation_tables)
    tframe = pd.read_csv(root / "data/manifests/active/vatex/threshold.csv", keep_default_na=False)
    cframe = pd.read_csv(root / "data/manifests/active/vatex/cdf.csv", keep_default_na=False)
    if not tframe.subset.eq("real").all() or set(tframe.video_id) & set(cframe.video_id):
        raise ValueError("阈值职责或独立性错误")
    records, ti = read_evidence(threshold_dir, tframe)
    cdf, ci = read_evidence(cdf_dir, cframe)
    if ti["models"] != ci["models"] or ti["config"]["selection"] != ci["config"]["selection"]:
        raise ValueError("阈值与CDF模型/观察不同")
    manifest = json.loads((evaluation_tables / "manifest.json").read_text())
    if (
        manifest["status"] != "completed"
        or manifest["models"] != ci["models"]
        or manifest["selection"] != ci["config"]["selection"]
    ):
        raise ValueError("评价与阈值模型/观察不符")
    path = evaluation_tables / "video_scores.csv.gz"
    if file_digest(path) != manifest["files"][path.name]:
        raise ValueError("评价分数hash改变")
    threshold_scores = calibrated_scores(
        root, domain, tframe, records, cdf, ci["config"]["selection"]["k"]
    )
    evaluation = pd.read_csv(path, float_precision="round_trip")
    source = pd.read_csv(
        root / "data/manifests/active" / domain / "evaluation.csv", keep_default_na=False
    ).set_index("video_id")
    rows = []
    for variant, part in evaluation.groupby("variant"):
        real_reference = threshold_scores[
            threshold_scores.variant.eq(variant)
        ].final_score.to_numpy()
        for budget in (0.001, 0.01):
            tau = real_threshold(real_reference, budget)
            real = part[part.subset.eq("real")].copy()
            fake = part[part.subset.eq("annotated")]
            groups = [
                ("all_real", real),
                *[
                    (name, g)
                    for name, g in real.assign(
                        real_source=source.loc[real.video_id].real_source.to_numpy()
                    ).groupby("real_source")
                ],
            ]
            for name, g in groups:
                errors = int((g.final_score < tau).sum())
                lo, hi = wilson(errors, len(g))
                rows.append(
                    dict(
                        dataset=domain,
                        variant=variant,
                        nominal_real_fpr=budget,
                        threshold=tau,
                        role="real",
                        group=name,
                        n=len(g),
                        count=errors,
                        rate=errors / len(g),
                        ci95_low=lo,
                        ci95_high=hi,
                        threshold_n=len(real_reference),
                        reference_real_fpr=float(np.mean(real_reference < tau)),
                    )
                )
            for name, g in fake.groupby("source_model"):
                hits = int((g.final_score < tau).sum())
                lo, hi = wilson(hits, len(g))
                rows.append(
                    dict(
                        dataset=domain,
                        variant=variant,
                        nominal_real_fpr=budget,
                        threshold=tau,
                        role="fake",
                        group=name,
                        n=len(g),
                        count=hits,
                        rate=hits / len(g),
                        ci95_low=lo,
                        ci95_high=hi,
                        threshold_n=len(real_reference),
                        reference_real_fpr=float(np.mean(real_reference < tau)),
                    )
                )
    output.mkdir(parents=True, exist_ok=False)
    threshold_scores.to_csv(output / "threshold_scores.csv.gz", index=False)
    pd.DataFrame(rows).to_csv(output / "operating_points.csv", index=False)
    paper_json(
        output / "manifest.json",
        dict(
            status="completed",
            dataset=domain,
            threshold_origin="independent VATEX 2000; each target scoring model",
            rule="fake if score < sorted_threshold_scores[floor(n*nominal_fpr)]",
            uncertainty="descriptive clip-level Wilson intervals; not source-correlation or cross-domain FPR guarantees",
            models=ti["models"],
            files={p.name: file_digest(p) for p in output.iterdir() if p.is_file()},
        ),
    )
