"""将已验收的局部方向实验汇成论文域表、参考敏感性和审计收据。"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from artifacts import atomic_csv, paper_json
from reference import file_digest


DISPLAY = (
    ("global_only", "Global"),
    ("d1_anchor", "Global＋Local D1"),
    ("d2_anchor", "Global＋Local D2"),
    ("pooled_d2", "Global＋pooled D2"),
    ("ttr", "Global＋TTR"),
    ("split", "Global＋SPLIT统计"),
)
ORDER = ("videofeedback", "genvideo", "comgenvid", "Average")


def table(frame, definitions):
    data = frame.set_index(["variant", "dataset"])
    rows = []
    maxima = {}
    for d in ORDER:
        for m in ("auc", "real_positive_ap"):
            maxima[d, m] = max(round(float(data.loc[v, d][m]), 3) for v, label in definitions)
    for v, label in definitions:
        cells = []
        for d in ORDER:
            texts = []
            for m in ("auc", "real_positive_ap"):
                val = float(data.loc[v, d][m])
                s = f"{val:.3f}"
                if round(val, 3) == maxima[d, m]:
                    s = (
                        f'<span style="color:#C00000"><strong>{s}</strong></span>'
                        if d == "Average"
                        else "**" + s + "**"
                    )
                texts.append(s)
            cells.append("/".join(texts))
        rows.append("| " + label + " | " + " | ".join(cells) + " |")
    return "\n".join(
        [
            "| 配置 | VideoFeedback | GenVideo | ComGenVid | Average |",
            "| --- | ---: | ---: | ---: | ---: |",
            *rows,
        ]
    )


def build(root):
    out = root / "results/runs/local_direction_evidence"
    source = json.loads((out / "evaluation.json").read_text())
    for p, h in source["files"].items():
        if file_digest(out / p) != h:
            raise ValueError("评价产物改变")
    analysis = json.loads((out / "analysis.json").read_text())
    if analysis["sha256"] != file_digest(out / "contrasts.csv") or analysis[
        "source"
    ] != file_digest(out / "video_scores.csv.gz"):
        raise ValueError("配对分析与分数不符")
    domain = pd.read_csv(out / "dataset_metrics.csv")
    macro = pd.read_csv(out / "macro_metrics.csv").rename(columns={"scope": "dataset"})
    frame = pd.concat([domain, macro])
    ci = pd.read_csv(out / "contrasts.csv")
    reg = pd.read_csv(out / "regression.csv")
    meta = pd.read_csv(root / "results/paper_complete/evaluation.csv", keep_default_na=False)
    if (
        len(reg) != 15569
        or set(reg.video_id) != set(meta.video_id)
        or reg.final_error.max() > 1e-12
        or reg.raw_error.max() > 1e-8
    ):
        raise ValueError("baseline回归未通过")
    full = frame[frame.variant.eq("d2_anchor")].set_index("dataset")
    g = frame[frame.variant.eq("global_only")].set_index("dataset")
    reps = []
    index = frame.set_index(["variant", "dataset"])
    for seed in (17, 29, 43, 59, 71):
        for d in ORDER:
            a = index.loc[f"d2_boot_{seed}", d]
            b = index.loc[f"d1_boot_{seed}", d]
            diag = index.loc[f"d2_diag_boot_{seed}", d]
            for metric in ("auc", "real_positive_ap"):
                reps.append(
                    dict(
                        dataset=d,
                        seed=seed,
                        metric=metric,
                        full=float(a[metric]),
                        global_only=float(g.loc[d, metric]),
                        full_minus_global=float(a[metric] - g.loc[d, metric]),
                        d2_minus_d1=float(a[metric] - b[metric]),
                        full_minus_diagonal=float(a[metric] - diag[metric]),
                        full_minus_anchor=float(a[metric] - full.loc[d, metric]),
                    )
                )
    repeats = pd.DataFrame(reps)
    atomic_csv(out / "reference_repeats.csv", repeats)
    summary = []
    for (d, metric), part in repeats.groupby(["dataset", "metric"], sort=False):
        for name in (
            "full",
            "full_minus_global",
            "d2_minus_d1",
            "full_minus_diagonal",
            "full_minus_anchor",
        ):
            v = part[name].to_numpy()
            summary.append(
                dict(
                    dataset=d,
                    metric=metric,
                    quantity=name,
                    mean=v.mean(),
                    sd=v.std(ddof=1),
                    minimum=v.min(),
                    maximum=v.max(),
                    positive_runs=int((v > 0).sum()),
                    repeats=len(v),
                )
            )
    atomic_csv(out / "reference_summary.csv", pd.DataFrame(summary))
    local_defs = [
        (v + "_raw", label.replace("Global＋", "")) for v, label in DISPLAY if v != "global_only"
    ]
    text = [
        "# 局部方向补实验：完成结果",
        "",
        "范围：当前23单元、15569评价身份；固定目标Global，Gaussian只使用原200真实片段池。",
        "Average为三域等权；下表保留三位，逐生成器与所有指标见CSV。",
        "",
        "## 同Global的强替代控制",
        "",
        table(frame, DISPLAY),
        "",
        "## 局部分支原始排序",
        "",
        table(frame, local_defs),
        "",
        "## 参考重拟合的敏感性",
        "",
        "五次原拟合池源组bootstrap：固定Global，重新拟合Local及其VATEX CDF。",
        "不是五套新独立真实库；各次包含重复源，片段数可变。均值、范围和标准差不解释为总体置信区间。",
        "",
        pd.DataFrame(summary)
        .query('dataset == "Average"')
        .to_markdown(index=False, floatfmt=".6f"),
        "",
        "## 固定参考下的配对差值",
        "",
        ci[ci.dataset.eq("Average")].to_markdown(index=False, floatfmt=".6f"),
        "",
        f"原D2窗口最大误差：{reg.raw_error.max():.3g}；原Final最大误差：{reg.final_error.max():.3g}。",
        "主方法不因某个域/seed的点估计而自动更换；负结果与已定义的停止条件一并保留。",
        "",
    ]
    (out / "RESULTS_zh.md").write_text("\n".join(text))
    # 相同参考池的五个重复只画散点及均值，不画伪95%置信带。
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), layout="constrained")
    for ax, (metric, title) in zip(axes, [("auc", "AUC"), ("real_positive_ap", "AP-real")]):
        for i, d in enumerate(ORDER):
            v = repeats[
                repeats.dataset.eq(d) & repeats.metric.eq(metric)
            ].full_minus_global.to_numpy()
            ax.scatter(i + np.linspace(-0.12, 0.12, len(v)), v, color="#245B83", s=24)
            ax.plot([i - 0.2, i + 0.2], [v.mean(), v.mean()], color="#C00000", linewidth=2)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xticks(range(4), ["VideoFeedback", "GenVideo", "ComGenVid", "Average"], rotation=15)
        ax.set_ylabel("Full - matched Global")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(out / "reference_sensitivity.png", dpi=180)
    plt.close(fig)
    paper_json(
        out / "verification.json",
        dict(
            status="verified",
            scope="fixed 23-cell evaluation / original-fit source bootstrap",
            evaluation_clips=15569,
            generator_cells=23,
            reference_repeats=5,
            independent_new_fit_pools=False,
            baseline_final_max_error=float(reg.final_error.max()),
            baseline_raw_max_error=float(reg.raw_error.max()),
            source_evaluation_sha256=file_digest(out / "evaluation.json"),
            source_analysis_sha256=file_digest(out / "analysis.json"),
            reporter_sha256=file_digest(Path(__file__)),
            files={
                p: file_digest(out / p)
                for p in [
                    "RESULTS_zh.md",
                    "reference_repeats.csv",
                    "reference_summary.csv",
                    "reference_sensitivity.png",
                ]
            },
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument(
        "--wait-pid",
        type=int,
        help="等待已确认的评分调度器退出，再验收状态并分析；不会启动/重启评分",
    )
    a = p.parse_args()
    root = a.root.resolve()
    if a.wait_pid is not None:
        import os, select

        # pidfd绑定当前进程实例，避免PID复用或日志暂时不更新造成误判断。
        if hasattr(os, "pidfd_open"):
            fd = os.pidfd_open(a.wait_pid)
        else:
            import ctypes

            libc = ctypes.CDLL(None, use_errno=True)
            if hasattr(libc, "pidfd_open"):
                fd = libc.pidfd_open(a.wait_pid, 0)
            elif os.uname().machine == "x86_64":
                fd = libc.syscall(434, a.wait_pid, 0)
            else:
                raise RuntimeError("此平台没有可用pidfd接口，请评分完成后直接运行report")
            if fd < 0:
                raise OSError(ctypes.get_errno(), "无法绑定评分进程句柄")
        try:
            while not select.select([fd], [], [], 30)[0]:
                pass
        finally:
            os.close(fd)
        status = json.loads(
            (root / "results/runs/local_direction_evidence/status.json").read_text()
        )
        if status.get("status") != "scores_complete":
            raise RuntimeError("评分未成功结束，不能汇报完整实验")
        from evaluation.direction_analysis import representations

        representations(root)
    build(root)
