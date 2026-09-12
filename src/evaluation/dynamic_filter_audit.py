"""以官方元数据诊断VideoFeedback筛选差异，不修改冻结原始结果。"""

import json
from pathlib import Path
import pandas as pd
from artifacts import paper_json
from reference import file_digest
from evaluation.tables import evaluate_fixed_pairs


def audit(root, output):
    root = Path(root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    metadata = {}
    paths = [
        root / "datasets/recovery/metadata" / f"videofeedback_{s}.json" for s in ("train", "test")
    ]
    for path in paths:
        for row in json.loads(path.read_text()):
            key = str(row["id"])
            if key in metadata and metadata[key]["dynamic degree"] != row["dynamic degree"]:
                raise ValueError("同视频动态等级冲突")
            metadata[key] = row
    frame = pd.read_csv(
        root / "data/manifests/active/videofeedback/evaluation.csv", keep_default_na=False
    )
    fake = frame[frame.subset == "annotated"].copy()
    fake["dynamic_degree"] = [metadata[Path(p).stem]["dynamic degree"] for p in fake.video_path]
    fake[["video_id", "source_model", "dynamic_degree"]].to_csv(
        output / "metadata_join.csv", index=False
    )
    counts = fake.groupby(["source_model", "dynamic_degree"]).size().unstack(fill_value=0)
    counts.to_csv(output / "dynamic_counts.csv")
    # 每个生成器保留等级3/4，真实两来源各取同数，保持原pair表顺序；
    # 奇数fake按原顺序去尾，使两真实源严格等量。该诊断不是原文全量复现。
    pairs = pd.read_csv(
        root / "data/manifests/active/videofeedback/pairs.csv", keep_default_na=False
    )
    indexed = frame.set_index("video_id")
    good = set(fake.loc[fake.dynamic_degree >= 3, "video_id"])
    parts = []
    for generator, group in pairs.groupby("generator", sort=False):
        f = group[(group.subset == "annotated") & group.video_id.isin(good)]
        n = len(f) // 2
        parts.append(f.iloc[: 2 * n])
        real = group[group.subset == "real"]
        for source in ("DiDeMo", "Panda70M"):
            rows = real[real.video_id.map(indexed.source_model).eq(source)].iloc[:n]
            if len(rows) != n:
                raise ValueError("真实源不足")
            parts.append(rows)
    selected = pd.concat(parts, ignore_index=True)
    selected.to_csv(output / "filtered_pairs.csv", index=False)
    outputs = []
    for mode in ("official", "fc3"):
        source = (
            root / "results/runs" / f"paper_tables_videofeedback_{mode}" / "video_scores.csv.gz"
        )
        if mode == "official":
            source = root / "results/runs/paper_tables_videofeedback_official/video_scores.csv.gz"
        scores = pd.read_csv(source, float_precision="round_trip")
        if "variant" not in scores:
            scores["variant"] = "official_single_window"
        for variant, rows in scores.groupby("variant", sort=False):
            if variant not in ("full", "global_only", "official_single_window"):
                continue
            for protocol, chosen in [("current_300", pairs), ("dynamic_3_4_balanced", selected)]:
                table = evaluate_fixed_pairs(rows, chosen)["generator_metrics"]
                outputs.append(table.assign(variant=variant, protocol=protocol))
    results = pd.concat(outputs, ignore_index=True)
    results.to_csv(output / "generator_metrics.csv", index=False)
    metrics = ["auc", "real_positive_ap", "fake_positive_ap"]
    summary = results.groupby(["variant", "protocol"])[metrics].mean()
    summary.to_csv(output / "summary.csv")
    paper_json(
        output / "manifest.json",
        dict(
            status="completed",
            metadata={str(p): file_digest(p) for p in paths},
            scope="现有300条生成器池内等级3/4筛选及匹配real重平衡；不等同原文完整视频池",
            fit_and_scores="unchanged",
            files={p.name: file_digest(p) for p in output.iterdir() if p.is_file()},
        ),
    )
    print(summary.to_string())
