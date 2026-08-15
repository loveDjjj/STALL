"""审计 Alpha-STALLED 结果与 STALL 参考文献实验体系的对齐程度。

输入是当前仓库已有的轻量结果资产；输出是一个可复现的 coverage matrix，
用于回答：参考文献 2603.15026v2 主文和附录做了哪些实验，我们当前哪些已经有证据，
哪些只完成了局部版本，哪些必须重新提特征/重建 cache，哪些不适合作为 Alpha-STALLED
的下一步优先项。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CHECKS = [
    {
        "reference_section": "Table 1 / Sec. 5.2",
        "reference_item": "三 benchmark 主结果与检测器对比",
        "alpha_stalled_counterpart": "三数据集 global-only / patch-only / Alpha-STALLED 主表",
        "required_paths": [
            "results/paper_tables/alpha_stalled_main_summary.md",
            "results/paper_tables/ablation_summary.md",
        ],
        "status_if_exists": "complete_for_our_method",
        "gap": "尚未重跑 AEROBLADE/RIGID/ZED/D3/T2VE/AIGVDet 等外部 baseline；当前论文可重点和 STALL global-only / patch branch 比较。",
        "next_action": "若投稿需要完整 SOTA baseline 表，再单独建立 baseline 环境重跑；否则不建议优先做。",
        "priority": "P2",
    },
    {
        "reference_section": "Fig. 6c / D.6.6",
        "reference_item": "calibration source ablation",
        "alpha_stalled_counterpart": "真实视频百分位校准假设与 cross-dataset frozen hyperparameter",
        "required_paths": [
            "results/journal_experiments/cross_dataset_frozen_hyperparams/cross_dataset_frozen_hyperparams.md",
        ],
        "status_if_exists": "partial",
        "gap": "还没有用 VATEX/MSR-VTT/Panda70M/DiDeMo 等不同真实校准源重建 Alpha-STALLED patch params。",
        "next_action": "需要重新构造真实校准集并生成 patch params；计算成本高，建议只在审稿要求时做一个代表数据集。",
        "priority": "P2",
    },
    {
        "reference_section": "Fig. 7a / D.6.5",
        "reference_item": "calibration set size ablation",
        "alpha_stalled_counterpart": "当前只审计 cache/storage；未按校准集大小重建参数",
        "required_paths": [
            "results/journal_experiments/runtime_storage_audit/runtime_storage_audit.md",
        ],
        "status_if_exists": "missing_but_cost_audited",
        "gap": "缺少 1k/5k/10k/... real calibration size 下的 patch/global 校准重建和评测。",
        "next_action": "需要从真实集抽样、多次重建 patch params；优先级低于 duration/window 代表实跑。",
        "priority": "P3",
    },
    {
        "reference_section": "Table 2 / D.6.4",
        "reference_item": "backbone encoder ablation",
        "alpha_stalled_counterpart": "当前 Alpha-STALLED 固定 DINOv3；无 MobileNet/ResNet/ViCLIP/VideoMAE 对照",
        "required_paths": [
            "results/paper_tables/alpha_stalled_main_summary.md",
        ],
        "status_if_exists": "missing_requires_feature_extraction",
        "gap": "需要为其他 backbone 重新提 global/patch embedding 与校准参数；不是 CSV 级分析。",
        "next_action": "不建议当前补；若要做，先选 ComGenVid 小规模 smoke，再决定是否铺开。",
        "priority": "P3",
    },
    {
        "reference_section": "Fig. 14 / D.2.1",
        "reference_item": "spatial-only / temporal-only / combined component ablation",
        "alpha_stalled_counterpart": "global-only / patch-only / Alpha-STALLED；ComGenVid patch spatial 与多种 temporal 定义",
        "required_paths": [
            "results/paper_tables/ablation_summary.md",
            "results/paper_scores/comgenvid_patch_spatial.csv",
            "results/paper_scores/comgenvid_patch_second_order.csv",
        ],
        "status_if_exists": "complete",
        "gap": "已覆盖我们的核心组件；不需要重复基础消融。",
        "next_action": "写论文时强调 global STALL 与 patch 二阶时序的互补，而不是沿用原文 spatial/temporal 命名。",
        "priority": "done",
    },
    {
        "reference_section": "Fig. 15 / D.2.2",
        "reference_item": "frame-level aggregation min/mean/max ablation",
        "alpha_stalled_counterpart": "patch region size、mean vs bottom-k aggregation、bottom-k ratio",
        "required_paths": [
            "results/journal_experiments/region_sensitivity/region_sensitivity_summary.md",
            "results/journal_experiments/aggregation_sensitivity/aggregation_sensitivity_summary.md",
            "results/journal_experiments/bottomk_sensitivity/bottomk_sensitivity_summary.md",
        ],
        "status_if_exists": "complete_for_patch_branch",
        "gap": "没有逐帧 min/max 原文式 ablation；但已覆盖 Alpha-STALLED patch 分支更相关的局部聚合边界。",
        "next_action": "不建议补原文 min/max；当前 region/aggregation 结果更贴合我们的创新点。",
        "priority": "done",
    },
    {
        "reference_section": "Fig. 13 / D.1",
        "reference_item": "temporal derivative order D=1/2/3/4",
        "alpha_stalled_counterpart": "ComGenVid lag-1、multi-lag、motion-hard/soft、同网格 D=2/3/4 patch temporal derivative 对照",
        "required_paths": [
            "results/paper_scores/comgenvid_patch_lag1.csv",
            "results/paper_scores/comgenvid_patch_multilag.csv",
            "results/paper_scores/comgenvid_patch_second_order_ablation.csv",
            "results/journal_experiments/temporal_derivative_order/comgenvid_temporal_derivative_order.md",
        ],
        "status_if_exists": "representative_derivative_order_complete",
        "gap": "已在 ComGenVid 上完成 D=3/D=4 代表性对照；尚未铺开三数据集完整 derivative-order sweep。",
        "next_action": "当前足以说明二阶局部时序证据不是任意选择；除非审稿要求，不建议扩展到三数据集。",
        "priority": "done",
    },
    {
        "reference_section": "Fig. 8 / D.6.1-D.6.3",
        "reference_item": "step size / FPS / video length ablation",
        "alpha_stalled_counterpart": "duration/window feasibility audit + ComGenVid 1s representative run",
        "required_paths": [
            "results/journal_experiments/duration_window_feasibility/duration_window_feasibility.md",
            "results/journal_experiments/duration_window_representative/comgenvid_duration_window_representative.md",
        ],
        "status_if_exists": "representative_run_complete",
        "gap": "已补 ComGenVid 1s vs 2s 代表性 patch-only 对照；尚未铺开三数据集 1s/3s/4s 全网格。",
        "next_action": "当前足以作为短窗口边界证据；除非审稿要求，不建议直接铺开 3 数据集 × 4 时长。",
        "priority": "done",
    },
    {
        "reference_section": "Fig. 7b / D.5",
        "reference_item": "image perturbation robustness",
        "alpha_stalled_counterpart": "当前无 JPEG/blur/crop/noise 扰动结果",
        "required_paths": [],
        "status_if_exists": "missing_requires_video_or_frame_reprocessing",
        "gap": "需要对视频帧施加扰动并重新提 DINOv3/patch 特征；不属于现有 CSV 可推导。",
        "next_action": "不建议当前补；若审稿要求，先做 GenVideo 小子集和 global-only/Alpha-STALLED 对照。",
        "priority": "P3",
    },
    {
        "reference_section": "D.4",
        "reference_item": "temporal perturbation robustness",
        "alpha_stalled_counterpart": "当前无 reverse/shuffle/flash 扰动实验",
        "required_paths": [],
        "status_if_exists": "missing_requires_video_or_cache_rebuild",
        "gap": "需要重排/插帧并重新提特征；patch cache 不能直接表达扰动后视频。",
        "next_action": "不建议当前补；可在未来用 400 个真实视频做小规模机制验证。",
        "priority": "P3",
    },
    {
        "reference_section": "B",
        "reference_item": "normality / Gaussian assumption tests",
        "alpha_stalled_counterpart": "当前没有针对 patch 二阶差分的 AD/DP 正态性检验",
        "required_paths": [],
        "status_if_exists": "missing_optional_theory",
        "gap": "可从 patch embedding cache 抽样做正态性检验，但 cache 体积大，且理论上我们主要继承原文 DINOv3 likelihood 假设。",
        "next_action": "可作为低优先级理论补充；优先先写明继承 STALL 全局校准假设，patch 分支为经验增强。",
        "priority": "P3",
    },
    {
        "reference_section": "A.4",
        "reference_item": "D3 baseline protocol audit",
        "alpha_stalled_counterpart": "D3 protocol audit：采样、类别平衡、FPS 处理、encoder 和指标方向",
        "required_paths": [
            "results/journal_experiments/d3_protocol_audit/d3_protocol_audit.md",
            "results/journal_experiments/d3_protocol_audit/d3_protocol_audit.csv",
        ],
        "status_if_exists": "protocol_audit_complete_without_external_rerun",
        "gap": "已完成 protocol audit；尚未重跑外部 D3 baseline。",
        "next_action": "若审稿要求 D3 数值，再按 audit 中的同索引、同帧索引、pairwise balanced 协议单独重跑。",
        "priority": "done",
    },
    {
        "reference_section": "E",
        "reference_item": "inference time / memory analysis",
        "alpha_stalled_counterpart": "runtime/storage audit + CSV-stage + video-stage runtime benchmark",
        "required_paths": [
            "results/journal_experiments/runtime_storage_audit/runtime_storage_audit.md",
            "results/journal_experiments/runtime_benchmark/csv_stage_runtime_benchmark.md",
            "results/journal_experiments/video_stage_runtime_benchmark/video_stage_runtime_benchmark.md",
        ],
        "status_if_exists": "representative_runtime_complete",
        "gap": "已有 storage footprint、CSV-stage 计时和 ComGenVid 小样本 clean-cache video-stage 计时；尚未做三数据集全量 clean-cache 端到端总耗时。",
        "next_action": "当前足以报告代表性阶段成本；除非审稿要求，不建议清空三数据集 cache 后重跑全量端到端。",
        "priority": "done",
    },
    {
        "reference_section": "D.7 / Fig. 19-20",
        "reference_item": "qualitative examples",
        "alpha_stalled_counterpart": "patch anomaly case visualization + 原视频关键帧解释 + failure case audit",
        "required_paths": [
            "results/journal_experiments/case_visualizations/patch_anomaly_cases.svg",
            "results/journal_experiments/keyframe_case_explanations/keyframe_patch_anomaly_cases.svg",
            "results/journal_experiments/keyframe_case_explanations/keyframe_case_explanations.md",
            "results/journal_experiments/failure_case_audit/failure_case_audit.md",
        ],
        "status_if_exists": "complete_with_raw_keyframes",
        "gap": "已完成关键帧级可视化和解释表；具体语义原因仍需作者人工复核原视频。",
        "next_action": "主文只选 2–3 个代表案例；完整 6 例放补充材料。",
        "priority": "done",
    },
    {
        "reference_section": "Statistical reporting",
        "reference_item": "显著性/稳定性分析",
        "alpha_stalled_counterpart": "逐生成器 paired bootstrap + 宏平均 paired bootstrap",
        "required_paths": [
            "results/paper_sensitivity/paired_bootstrap_delta_summary.csv",
            "results/journal_experiments/macro_average_bootstrap/macro_average_paired_bootstrap_delta.md",
        ],
        "status_if_exists": "complete_extra",
        "gap": "该项比原文主文更充分，可用于论文 Average 行稳定性。",
        "next_action": "写作时区分宏平均总体提升与 VideoFeedback 内部负迁移。",
        "priority": "done",
    },
]


def evaluate(root: Path) -> pd.DataFrame:
    rows = []
    for item in CHECKS:
        required = item["required_paths"]
        existing = [path for path in required if (root / path).exists()]
        missing = [path for path in required if not (root / path).exists()]
        if not required:
            evidence_status = "no_local_evidence_required_paths"
            status = item["status_if_exists"]
        elif missing:
            evidence_status = "missing_required_paths"
            status = "missing_evidence"
        else:
            evidence_status = "all_required_paths_exist"
            status = item["status_if_exists"]
        rows.append(
            {
                "reference_section": item["reference_section"],
                "reference_item": item["reference_item"],
                "alpha_stalled_counterpart": item["alpha_stalled_counterpart"],
                "status": status,
                "priority": item["priority"],
                "evidence_status": evidence_status,
                "existing_evidence": "; ".join(existing),
                "missing_evidence": "; ".join(missing),
                "gap": item["gap"],
                "next_action": item["next_action"],
            }
        )
    return pd.DataFrame(rows)


def write_markdown(df: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# 参考文献实验体系对齐审计",
        "",
        "本审计以 `2603.15026v2` 的主文与附录实验为参照，检查当前 Alpha-STALLED 结果资产的覆盖程度。",
        "它只检查已有文件，不重新提取特征、不运行模型。",
        "",
        "## 总览",
        "",
        "| priority | status | count |",
        "|---|---|---:|",
    ]
    summary = df.groupby(["priority", "status"]).size().reset_index(name="count")
    for row in summary.sort_values(["priority", "status"]).itertuples(index=False):
        lines.append(f"| {row.priority} | {row.status} | {row.count} |")

    lines.extend(
        [
            "",
            "## 逐项对照",
            "",
            "| reference section | reference experiment | our counterpart | status | priority | evidence | next action |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in df.itertuples(index=False):
        evidence = row.existing_evidence if row.existing_evidence else row.evidence_status
        lines.append(
            f"| {row.reference_section} | {row.reference_item} | {row.alpha_stalled_counterpart} | "
            f"{row.status} | {row.priority} | {evidence} | {row.next_action} |"
        )

    actionable = df[df["priority"].isin(["P1", "P2", "P3"])].copy()
    lines.extend(
        [
            "",
            "## 当前建议",
            "",
            "- `done` 项已经足够支撑当前手稿，不建议重复。",
            "- 当前已无 P1 缺口；若继续投入，应只处理审稿明确要求的 P2/P3 项。",
            "- `P2` 项只在版面、审稿或对比需求明确时做：外部 baseline、calibration source 代表性消融。",
            "- `P3` 项需要重新提特征或重建大量 cache，当前不建议作为下一步默认任务。",
            "",
            "## 尚未完成但可执行的缺口",
            "",
            "| priority | reference experiment | gap | concrete next action |",
            "|---|---|---|---|",
        ]
    )
    for row in actionable.sort_values(["priority", "reference_section"]).itertuples(index=False):
        if row.priority == "done":
            continue
        lines.append(f"| {row.priority} | {row.reference_item} | {row.gap} | {row.next_action} |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "results/journal_experiments/reference_alignment_audit",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = evaluate(root)
    df.to_csv(out_dir / "reference_experiment_alignment.csv", index=False)
    write_markdown(df, out_dir / "reference_experiment_alignment.md")
    print(f"rows={len(df)}")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
