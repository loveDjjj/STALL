"""完整实验验收后导出论文表；不读取历史最优行，不按结果筛选数据域。"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from artifacts import paper_json
from reference import file_digest

STUDIES = {
    "main": [
        ("official", "official_single_window", "原版STALL单窗"),
        ("fc3", "official_global", "官方Global＋FC3"),
        ("fc3", "global_only", "目标Global＋FC3"),
        ("fc3", "full", "完整主线"),
    ],
    "components": [
        ("fc3", v, n)
        for v, n in [
            ("full", "完整主线"),
            ("global_only", "去Local"),
            ("local_only", "去Global"),
            ("without_gs", "去Global空间"),
            ("without_gt", "去Global时序"),
        ]
    ],
    "representation": [
        ("fc3", v, n)
        for v, n in [
            ("local_d1_target", "Local归一化D1"),
            ("full", "Local归一化D2"),
            ("local_raw_d2_target", "Local未归一化D2"),
            ("local_d2_norm_target", "Local D2幅度"),
            ("global_d2_target", "Global归一化D2"),
        ]
    ],
    "representation_local": [
        ("fc3", r + "_target_" + suffix, label + "（" + stage + "）")
        for r, label in [
            ("local_d1", "Local D1"),
            ("local_d2", "Local D2"),
            ("local_raw_d2", "未归一化D2"),
            ("local_d2_norm", "D2幅度"),
            ("global_d2", "Global D2"),
        ]
        for suffix, stage in [("raw", "原始分数"), ("only", "视频CDF后")]
    ],
    "reference_sources": [
        ("fc3", v, n)
        for v, n in [
            ("global_source_local_source", "Global源／Local源"),
            ("global_source_local_target", "Global源／Local目标"),
            ("global_target_local_source", "Global目标／Local源"),
            ("full", "Global目标／Local目标"),
        ]
    ],
    "local_metric": [
        ("fc3", v, n)
        for v, n in [
            ("global_target_local_source", "源均值／源白化"),
            ("local_d2_target_mean_source_whitening", "目标均值／源白化"),
            ("local_d2_source_mean_target_whitening", "源均值／目标白化"),
            ("full", "目标均值／目标完整白化"),
            ("local_d2_target_diagonal", "目标对角白化"),
        ]
    ],
    "observation": [
        (s, "full", n)
        for s, n in [
            ("uniform1", "Uniform K1"),
            ("fc1", "Feature-change K1"),
            ("uniform3", "Uniform K3"),
            ("fc3", "Feature-change K3"),
        ]
    ],
}
CONTRASTS = [
    "global_only",
    "local_only",
    "without_gs",
    "without_gt",
    "official_global",
    "local_d1_target",
    "local_raw_d2_target",
    "local_d2_norm_target",
    "global_d2_target",
    "global_source_local_source",
    "global_target_local_source",
    "global_source_local_target",
    "local_d2_target_mean_source_whitening",
    "local_d2_source_mean_target_whitening",
    "local_d2_target_diagonal",
    "official",
    "uniform1",
    "fc1",
    "uniform3",
]
TITLES = {
    "main": "1. 主实验",
    "components": "2. 组件消融",
    "representation": "3. 表示消融",
    "representation_local": "3.1 Local单分支",
    "reference_sources": "4.1 参考来源四格",
    "local_metric": "4.2 Local均值与白化",
    "observation": "5. 观察预算",
}


def export_paper(root, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    output = Path(output)
    runs = root / "results/runs"
    sources = {}

    def read(run, name):
        run = Path(run)
        manifest = json.loads((run / "manifest.json").read_text())
        if manifest["status"] != "completed":
            raise ValueError("未完成的论文输入")
        if name not in manifest["files"] or file_digest(run / name) != manifest["files"][name]:
            raise ValueError("论文输入hash错误")
        sources[str(run.relative_to(root))] = file_digest(run / "manifest.json")
        return pd.read_csv(run / name, float_precision="round_trip")

    modes = {}
    for mode in ("official", "fc3", "uniform1", "fc1", "uniform3"):
        run = runs / f"paper_tables_all_{mode}"
        modes[mode] = {
            name: read(run, name + ".csv")
            for name in ("generator_metrics", "dataset_metrics", "macro_metrics")
        }
    intervals = []
    for name in CONTRASTS:
        run = runs / f"paper_interval_full_vs_{name}"
        delta = read(run, "deltas.csv")
        ci = read(run, "intervals.csv")
        metadata = json.loads((run / "manifest.json").read_text())["inputs"]
        if metadata["iterations"] != 1000 or metadata.get("csv_float_policy") != "round_trip":
            raise ValueError("区间协议不是本轮完整版本")
        intervals.append(
            delta.merge(ci, on=["dataset", "metric"], validate="one_to_one").assign(
                contrast="full minus " + name
            )
        )
    timings = []
    operating = []
    for domain in ("comgenvid", "videofeedback", "genvideo"):
        timings.append(read(runs / f"paper_runtime_{domain}", "timings.csv"))
        operating.append(read(runs / f"paper_operating_points_{domain}", "operating_points.csv"))
    pairs = pd.read_csv(root / "data/manifests/active/pairs.csv")
    pairs = pairs[pairs.dataset.isin(("comgenvid", "videofeedback", "genvideo"))]
    expected = set(map(tuple, pairs[["dataset", "generator"]].drop_duplicates().to_numpy()))
    if len(expected) != 20:
        raise ValueError("主表不是固定20单元")
    exports = {}
    for study, definitions in STUDIES.items():
        summary = []
        generators = []
        for mode, variant, label in definitions:
            cell = modes[mode]["generator_metrics"]
            cell = cell[cell.variant.eq(variant)]
            if (
                len(cell) != 20
                or set(map(tuple, cell[["dataset", "generator"]].to_numpy())) != expected
            ):
                raise ValueError(f"{study}/{variant}没有覆盖20单元")
            generators.append(cell.assign(method=label, observation=mode))
            d = modes[mode]["dataset_metrics"]
            m = modes[mode]["macro_metrics"].rename(columns={"scope": "dataset"})
            combined = pd.concat(
                [d[d.variant.eq(variant)], m[m.variant.eq(variant)]], ignore_index=True
            )
            if len(combined) != 4:
                raise ValueError("缺少域级或Macro结果")
            summary.append(combined.assign(method=label, observation=mode))
        exports[study] = pd.concat(summary, ignore_index=True)
        exports[study + "_generators"] = pd.concat(generators, ignore_index=True)
    exports["confidence_intervals"] = pd.concat(intervals, ignore_index=True)
    exports["operating_points"] = pd.concat(operating, ignore_index=True)
    if (
        exports["operating_points"].reference_real_fpr
        > exports["operating_points"].nominal_real_fpr + 1e-12
    ).any():
        raise ValueError("阈值参考误报超过预定预算")
    timing = pd.concat(timings, ignore_index=True)
    exports["runtime_videos"] = timing
    exports["runtime_summary"] = (
        timing.groupby(["dataset", "selector"])
        .agg(
            n_measurements=("total_seconds", "size"),
            n_videos=("video_id", "nunique"),
            mean_seconds=("total_seconds", "mean"),
            median_seconds=("total_seconds", "median"),
            p95_seconds=("total_seconds", lambda x: x.quantile(0.95)),
            mean_unique_frames=("observed_unique_frames", "mean"),
            peak_gpu_mib=("peak_gpu_allocated_mib", "max"),
        )
        .reset_index()
    )
    new = read(runs / "paper_tables_all_fc3", "video_scores.csv.gz")
    old = pd.read_csv(
        root / "results/reference/baseline/video_scores.csv", float_precision="round_trip"
    )
    old = old[old.dataset.isin(("comgenvid", "videofeedback", "genvideo"))].set_index("video_id")
    reproduction = []
    for variant, column in [
        ("full", "final_score"),
        ("global_only", "global_score"),
        ("local_only", "local_score"),
    ]:
        part = new[new.variant.eq(variant)].set_index("video_id")
        if set(part.index) != set(old.index) or len(part) != 21421:
            raise ValueError("重跑身份不完整")
        delta = part.final_score.to_numpy() - old.loc[part.index, column].to_numpy()
        reproduction.append(
            dict(
                variant=variant,
                videos=len(part),
                max_abs_difference=float(abs(delta).max()),
                different_scores=int(np.count_nonzero(delta)),
            )
        )
    exports["reproduction"] = pd.DataFrame(reproduction)
    output.mkdir(parents=True, exist_ok=False)
    for name, table in exports.items():
        table.to_csv(output / (name + ".csv"), index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    labels = {
        "uniform1": "Uniform K1",
        "fc1": "Feature-change K1",
        "uniform3": "Uniform K3",
        "fc3": "Feature-change K3",
    }
    for axis, domain in zip(axes, ("comgenvid", "videofeedback", "genvideo")):
        for mode in labels:
            point = (
                exports["observation"].query("dataset == @domain and observation == @mode").iloc[0]
            )
            cost = (
                exports["runtime_summary"].query("dataset == @domain and selector == @mode").iloc[0]
            )
            axis.scatter(cost.mean_seconds, point.auc, label=labels[mode])
        axis.set_title(domain)
        axis.set_xlabel("Seconds per video")
        axis.set_ylabel("AUC")
        axis.grid(alpha=0.2)
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "observation_cost.png", dpi=180)
    plt.close(fig)
    lines = [
        "# 论文前五组实验结果",
        "",
        "本轮使用当前主线重新拟合与评分。主方法：目标Global＋Local归一化D2，Feature-change K3，0.5/0.5。",
        "开发评分池21421视频；主表固定20个数据集—生成器单元、13033唯一视频、20870配对行。Macro先域内生成器等权，再三域等权。AP指AP-real，AP-fake另列。",
        "目标拟合每域200片段；ComGenVid为132源组，其余两域各200源组。同预算外部统计使用独立VATEX200；CDF为另一批2000，阈值再独立2000。",
        "所有区间条件于固定参考，采用1000次源组配对Poisson bootstrap，未作多重比较校正。独立阈值迁移不保证目标域名义误报率。",
        "",
    ]
    for study in STUDIES:
        lines += [
            "## " + TITLES[study],
            "",
            "| 方法 | Macro AUC | AP-real |",
            "| --- | ---: | ---: |",
        ]
        for row in exports[study][exports[study].dataset.eq("Macro-3")].itertuples():
            lines.append(f"| {row.method} | {row.auc:.6f} | {row.real_positive_ap:.6f} |")
        lines += ["", f"逐域见`{study}.csv`；全部20单元见`{study}_generators.csv`。", ""]
    lines += [
        "## 解释边界",
        "",
        "主比较相对官方STALL的差异同时包含观察预算和目标真实数据适配；Local独立作用看同目标预算Global-only控制。",
        "源／目标Global×Local四格存在交互，不能把两项收益直接相加；均值／白化交换是在固定目标Global下的Local控制。",
        "Global D2与Local每视频拟合向量数不同：Global D2使用所有有效时间差分，Local固定256；它们共享视频与观察窗口预算。",
        "官方对照使用官方固定随机2秒窗与NumPy评分，工程前向统一batch8。当前Global窗口CDF仍为VATEX Uniform首窗；观察消融只给Local建立匹配视频CDF。",
        "成本使用实际单模型评分器，无特征缓存，每选择器预热后测两遍；OS页缓存不受控，不能把第一遍叫严格冷磁盘。GPU峰值与进程RSS口径见各runtime manifest。",
        "成本子集包含ComGenVid 6、VideoFeedback 24、GenVideo 18条视频，各策略各测两遍，共384次计时；图中横轴为该子集成本，纵轴为全量配对AUC，各面板纵轴范围不同。",
        "不按数据域切换方法；VideoFeedback等不利结果完整保留。数据已用于开发，不称未接触确认集。",
        "",
        "完整差值区间见`confidence_intervals.csv`；独立阈值见`operating_points.csv`；成本见`runtime_summary.csv`及`runtime_videos.csv`；新旧分数核对见`reproduction.csv`。",
        "",
        "![观察预算与成本](observation_cost.png)",
    ]
    (output / "RESULTS_zh.md").write_text("\n".join(lines) + "\n")
    paper_json(
        output / "manifest.json",
        dict(
            status="completed",
            sources=sources,
            scope="first five paper experiment groups",
            files={p.name: file_digest(p) for p in output.iterdir() if p.is_file()},
        ),
    )
    return output
