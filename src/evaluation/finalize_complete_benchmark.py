"""验收后发布完整表格说明和成本资产，旧20单元导出保持只读。"""

import argparse
import json
from pathlib import Path
import pandas as pd
from artifacts import paper_json
from reference import file_digest
from evaluation.paper_export import STUDIES


def finalize(root, directory):
    root = Path(root)
    directory = Path(directory)
    verification = json.loads((directory / "verification.json").read_text())
    if verification["status"] != "verified" or verification["cells"] != 23:
        raise ValueError("完整实验尚未验收")
    original = json.loads((directory / "manifest.json").read_text())
    timings = []
    sources = {}
    for d in ("comgenvid", "videofeedback", "genvideo"):
        p = root / "results/runs" / f"paper_runtime_{d}/timings.csv"
        timings.append(pd.read_csv(p).assign(window_frames=16))
        sources[str(p)] = file_digest(p)
    p = root / "results/runs/complete23_short_runtime/timings.csv"
    timings.append(pd.read_csv(p).assign(window_frames=8))
    sources[str(p)] = file_digest(p)
    timing = pd.concat(timings, ignore_index=True)
    timing.to_csv(directory / "runtime_videos.csv", index=False)
    timing.groupby(["dataset", "window_frames", "selector"]).agg(
        n_measurements=("total_seconds", "size"),
        n_clip_ids=("video_id", "nunique"),
        mean_seconds=("total_seconds", "mean"),
        median_seconds=("total_seconds", "median"),
        mean_unique_frames=("observed_unique_frames", "mean"),
        peak_gpu_mib=("peak_gpu_allocated_mib", "max"),
    ).reset_index().to_csv(directory / "runtime_summary.csv", index=False)
    lines = [
        "# 完整23单元论文实验结果",
        "",
        f"协议：{original['scope']}。所有主比较及消融覆盖VideoFeedback 11、GenVideo 10、ComGenVid 2单元。",
        "paper_aligned按原文元数据保留动态等级3–4，并采用官方seed42真实配对；coverage_only仅补短视频，保留原长视频池。协议分别报告，不按结果择优。",
        "",
        f"当前{verification['evaluation_clip_ids']}个带时长评价身份，对应{verification['physical_video_paths']}个视频文件、{verification['pair_rows']}条配对行。",
        "本方法与受控消融的既有16帧分数不变；原生STALL对照另外用官方单视频默认batch32上限、无补尾批重算。新增8帧单窗、对应CDF及阈值，所有变体共享同一协议内配对身份。",
        "",
        "AUC/AP-real高为真实；Average按23单元等权，Macro-3按三域等权。CI为1000次源组配对Poisson bootstrap，同一real的1/2秒共享源组，条件于固定参考且未作全部探索的多重校正。",
        "",
    ]
    for study in STUDIES:
        table = pd.read_csv(directory / f"{study}.csv")
        lines += [
            "## " + study,
            "",
            "| 方法 | VideoFeedback AUC/AP | GenVideo AUC/AP | ComGenVid AUC/AP | Average AUC/AP | Macro-3 AUC/AP |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for (variant, mode), part in table.groupby(["variant", "observation"], sort=False):
            values = []
            for domain in ("videofeedback", "genvideo", "comgenvid", "Average", "Macro-3"):
                r = part[part.dataset == domain].iloc[0]
                values.append(f"{r.auc:.3f}/{r.real_positive_ap:.3f}")
            lines.append("| " + part.method.iloc[0] + " | " + " | ".join(values) + " |")
        lines += ["", f"全部23单元的精确数据：[逐生成器CSV]({study}_generators.csv)。", ""]
    lines += [
        "## 解释与复现边界",
        "",
        "原文明确对三个短生成器采用8帧；本研究将相同例外扩展到全部消融，短片段的四种观察策略均退化为相同单窗。",
        "官方原生STALL仍使用官方NPZ；我们的8帧分支保留目标Gaussian，重建对应8帧参考。二者不混称。",
        "全部生成器覆盖不等于与原文完全相同的视频数量：VF现有真实池500，短Hotshot配对500；原文有更大的真实/生成池。",
        "动态等级筛选只能解释部分复现差异。GenVideo短HotShot的官方复现与原文在两位精度一致；原文AP文字/代码与跨基准Average仍有未解歧义。",
        "最终对齐协议的Local增量AUC/AP区间均为正，但AP下界接近零；中间的旧配对筛选版本AP区间跨零。应标明协议与固定参考的区间边界，不按显著性选配对。",
        "成本为原48个长窗测试身份加12个短窗身份，共480次测量；这些是明确的计时子集，非全评测池平均延迟，也不要求计时子集均进入动态筛选后的主评价。",
        "短窗计时使用实际主方法而非全部消融共享评分器；96次输出均与正式分数一致。",
        "独立阈值按8/16帧分开建立；操作点按长度报告，不把同一真实视频两个裁剪当成两条独立来源。",
        "",
        "验收见[verification.json](verification.json)，区间见[confidence_intervals.csv](confidence_intervals.csv)，操作点见[operating_points.csv](operating_points.csv)。",
    ]
    (directory / "RESULTS_zh.md").write_text("\n".join(lines) + "\n")
    original.update(
        status="completed",
        verification=verification,
        tests_passed=52,
        runtime_sources=sources,
        finalized_code_sha256=file_digest(Path(__file__)),
        files={
            p.name: file_digest(p)
            for p in directory.iterdir()
            if p.is_file() and p.name != "manifest.json"
        },
    )
    paper_json(directory / "manifest.json", original)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory")
    a = p.parse_args()
    finalize(Path.cwd(), a.directory)
