#!/usr/bin/env python3
"""从已验收结果导出三张论文表；--check 只校验身份、汇总与逐字一致性。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAPER = "results/paper_complete"
DIRECTION = "results/runs/local_direction_evidence"
CONFIRM = "results/runs/reference_confirmation/results"
DOMAINS = ("videofeedback", "genvideo", "comgenvid")
DOMAIN_NAMES = dict(zip(DOMAINS, ("VideoFeedback", "GenVideo", "ComGenVid")))
METRICS = ("auc", "real_positive_ap")
PUBLISHED_SHA256 = "dc98f2c30efe3bc35a04dc483e60c11f6c9a63683c994737a3d365abe681a7be"
# 固定已验收输入身份；更新实验范围时应显式审阅这些锚点。
HASHES = {
    f"{PAPER}/main_generators.csv": "636f2c9fbd129d37bfc6fb8139a6691cbf8504109701226f419afaf47d829887",
    f"{PAPER}/main.csv": "77bd40dd2d8bc30a58ffd8f86fa40a983696df8ab18adc6c92130f85924b5329",
    f"{PAPER}/components.csv": "97c30b7eea0ed9a1499c22564bcdefffc731426bd987f13f3cbfba858f6b9842",
    f"{PAPER}/representation.csv": "bfcf14ce571cec55816a94d5484d304c4f253d9574f6196d30ea30df9584bd6e",
    f"{DIRECTION}/dataset_metrics.csv": "312557bd1a4ddadf57a4ac278a68e0a04883f190ef2046b764e15006bae7859b",
    f"{DIRECTION}/macro_metrics.csv": "edd1015efe3a53ea03a7982eef79414d410ef14049470420bc09215d6f1d9a40",
    f"{DIRECTION}/scalar_control/dataset_metrics.csv": "bdd1a9a5293e66eafd14707a271e8c6f38a7e81b763e6cc80ea1e6970ee95481",
    f"{DIRECTION}/scalar_control/macro_metrics.csv": "0fb87a233cb32da9a051fadf723b8796c743d34e836479965dacf0c2e021a450",
    f"{CONFIRM}/dataset_metrics.csv": "54110256e0609f55300b070d4f5b09226d65c9f57db48bd928c50a867ab6b72b",
    f"{CONFIRM}/macro_metrics.csv": "233219e75036601c4bb4328a20ba9e12c5e2ea88fd1651961a725e24a23efaaa",
    f"{CONFIRM}/reference_means.csv": "86b1c96777b7ddce1f1c8851faeb98c4bb7bb9518090e7bc45fadf758ee298b0",
    f"{CONFIRM}/contrasts.csv": "a32440fba899e6719725a1947b0bdbca1361f6b47556626fe3cfb8aad35fbca9",
}


def read_csv(path: str) -> list[dict[str, str]]:
    payload = (ROOT / path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != HASHES[path]:
        raise ValueError(f"输入身份改变，须重新审阅：{path}")
    return list(csv.DictReader(payload.decode("utf-8").splitlines()))


def one(rows: list[dict[str, str]], **keys: str) -> dict[str, str]:
    selected = [r for r in rows if all(r[k] == v for k, v in keys.items())]
    if len(selected) != 1:
        raise ValueError(f"应有唯一结果：{keys}，实际 {len(selected)}")
    return selected[0]


def pair(row: dict[str, str]) -> tuple[float, float]:
    return tuple(float(row[k]) for k in METRICS)


def equal(left: float, right: float) -> None:
    if not math.isclose(left, right, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"数值不一致：{left} != {right}")


def formatted(value: float, best: bool = False, red: bool = False) -> str:
    out = f"{value:.3f}"
    if best:
        out = rf"\textbf{{{out}}}"
    if red:
        out = rf"\textcolor{{mainred}}{{{out}}}"
    return out


def grouped_metrics(data: list[tuple[float, float]]) -> list[str]:
    """按三位显示值比较每个指标，保留显示并列。"""
    maxima = [max(f"{p[i]:.3f}" for p in data) for i in range(2)]
    return [formatted(v, f"{v:.3f}" == maxima[i])
            for values in data for i, v in enumerate(values)]


def table_start(caption: str, label: str, wide: bool = False) -> list[str]:
    kind = "table*" if wide else "table"
    return ["% 由 export_tables.py 生成；CSV 身份及汇总经校验。",
            rf"\begin{{{kind}}}[t]", r"  \centering", rf"  \caption{{{caption}}}",
            rf"  \label{{tab:{label}}}", r"  {\scriptsize" if wide else r"  {\footnotesize"]


def published_rows() -> list[tuple[str, str, list[float]]]:
    # 发表值唯一源为作者核验的 STALL v2 Table 1 转录，不从本地 STALL 结果取数。
    text = (ROOT / "docs/MANUSCRIPT_REVISION_PLAN_zh.md").read_text(encoding="utf-8")
    header = "| Benchmark | Model | AEROBLADE"
    block = text[text.index(header):].split("\n\n", 1)[0]
    rows = []
    for line in block.splitlines()[2:]:
        fields = [s.strip().replace("**", "") for s in line.strip("|").split("|")]
        if len(fields) != 16:
            raise ValueError("发表表结构改变，须重新审阅")
        numbers = [float(re.search(r"0\.\d+", s).group()) for s in fields[2:14]]
        rows.append((fields[0], fields[1], numbers))
    if len(rows) != 27:
        raise ValueError("发表表应含 23 个生成器、3 个域平均和总体平均")
    digest = hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()
    if digest != PUBLISHED_SHA256:
        raise ValueError("发表值转录改变，须重新核对 STALL v2 Table 1")
    return rows


def main_table(data: dict[str, list[dict[str, str]]]) -> str:
    published = published_rows()
    generators = [r for r in data[f"{PAPER}/main_generators.csv"] if r["variant"] == "full"]
    if len(generators) != 23:
        raise ValueError("主表必须包含 23 个生成器单元")
    domain_rows = data[f"{PAPER}/main.csv"]
    aliases = {("videofeedback", "LaVie"): "LaVie-base",
               ("videofeedback", "Sora"): "SoRA-Clip",
               ("videofeedback", "Text2Video"): "Text2Video-Zero",
               ("videofeedback", "ZeroScope"): "ZeroScope-576w",
               ("genvideo", "Show 1"): "Show_1",
               ("genvideo", "HotShot-XL"): "HotShot"}
    lines = table_start(
        r"三个基准的逐生成器结果。左组为 STALL v2 Table 1 的发表值\cite{benhayun2026stall}；"
        r"右组为本文计算值。两组的评价身份、目标真实信息和观察预算不完全相同，分组展示不构成同协议排名。",
        "main_results", wide=True)
    lines += [r"  \setlength{\tabcolsep}{1.4pt}", r"  \renewcommand{\arraystretch}{1.08}",
              r"  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ll*{12}{c}*{2}{>{\columncolor{mainred!4}}c}@{}}", r"    \toprule",
              r"    & & \multicolumn{12}{c}{发表结果$^{\dagger}$} & \multicolumn{2}{c}{本次计算$^{\ddagger}$} \\",
              r"    \cmidrule(lr){3-14}\cmidrule(l){15-16}",
              r"    数据集 & 生成器 & \multicolumn{2}{c}{AEROBLADE} & \multicolumn{2}{c}{RIGID} & \multicolumn{2}{c}{ZED} & \multicolumn{2}{c}{D3 (L2)} & \multicolumn{2}{c}{D3 (cos)} & \multicolumn{2}{c}{STALL} & \multicolumn{2}{c}{\method} \\",
              "    & & " + " & ".join(["AUC & AP"] * 7) + r" \\", r"    \midrule"]
    visited = set()
    for domain in DOMAINS:
        group = [r for r in published if r[0] == DOMAIN_NAMES[domain]]
        if sum(r[1] != "Average" for r in group) != len([r for r in generators if r["dataset"] == domain]):
            raise ValueError(f"生成器覆盖不同：{domain}")
        for index, (_, model, values) in enumerate(group):
            if model == "Average":
                ours = pair(one(domain_rows, dataset=domain, variant="full"))
                for i in range(2):
                    equal(ours[i], mean(pair(r)[i] for r in generators if r["dataset"] == domain))
                lines.append(r"    \cmidrule(l){2-16}")
                model = r"\textbf{Average}"
            else:
                key = aliases.get((domain, model), model)
                row = one(generators, dataset=domain, generator=key)
                visited.add((domain, key))
                ours = pair(row)
            domain_label = "CGV" if domain == "comgenvid" else rf"\rotatebox{{90}}{{{DOMAIN_NAMES[domain]}}}"
            prefix = rf"\multirow{{{len(group)}}}{{*}}{{{domain_label}}}" if index == 0 else ""
            compared = grouped_metrics(list(zip(values[::2], values[1::2])))
            lines.append("    " + " & ".join([prefix, model, *compared, *(formatted(x) for x in ours)]) + r" \\")
        lines.append(r"    \midrule")
    if len(visited) != 23:
        raise ValueError("主表生成器映射不唯一")
    _, _, values = published[-1]
    ours = pair(one(domain_rows, dataset="Macro-3", variant="full"))
    for i, expected in enumerate((0.8744719007139845, 0.8770747547781038)):
        equal(ours[i], expected)
        equal(ours[i], mean(pair(one(domain_rows, dataset=d, variant="full"))[i] for d in DOMAINS))
    lines.append(r"    \multicolumn{2}{l}{\textbf{总体 Average}} & " + " & ".join([
        *grouped_metrics(list(zip(values[::2], values[1::2]))),
        *(formatted(x, best=True, red=True) for x in ours)]) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular*}", r"  \par\vspace{2pt}",
              r"  \begin{minipage}{\textwidth}\scriptsize",
              r"  $^{\dagger}$发表 AP 与各级 Average 按原报告照录，补零不增加原始精度。"
              r"$^{\ddagger}$本文 AP 为 AP-real，总体 Average 为域内生成器等权后再三域等权。"
              r"左组粗体标记组内逐行显示最高值（含并列）；浅底色定位本文两列，深红强调本文总体结果。CGV：ComGenVid。",
              r"  \end{minipage}}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def component_table(data: dict[str, list[dict[str, str]]]) -> str:
    specs = [
        (r"$G$", PAPER, "components", "global_only"),
        (r"$L$（D2）", PAPER, "components", "local_only"),
        (r"$G+L$（Full）", PAPER, "components", "full"),
        (r"pooled D2", DIRECTION, "", "pooled_d2"),
        (r"D2（同位置数）", CONFIRM, "", "original_matched"),
        (r"D1", PAPER, "representation", "local_d1_target"),
        (r"TTR", DIRECTION, "", "ttr"),
        (r"SPLIT 统计", DIRECTION, "", "split"),
        (r"二维 Gaussian", DIRECTION + "/scalar_control", "", "scalar_gaussian"),
    ]
    values = []
    for _, directory, filename, variant in specs:
        rows = data[f"{directory}/{filename or 'dataset_metrics'}.csv"]
        by_domain = [pair(one(rows, dataset=d, variant=variant)) for d in DOMAINS]
        if filename:
            aggregate = pair(one(rows, dataset="Macro-3", variant=variant))
        else:
            aggregate = pair(one(data[f"{directory}/macro_metrics.csv"], scope="Average", variant=variant))
        for i in range(2):
            equal(aggregate[i], mean(p[i] for p in by_domain))
        values.append([*by_domain, aggregate])
    lines = table_start(
        r"局部分支的受控贡献。每格为 AUC/AP-real，Average 为三域等权。"
        r"粗体为整表各列各指标的显示最高值（含并列）；浅底色标出冻结的完整方法。",
        "component_ablation")
    lines += [r"  \setlength{\tabcolsep}{2.2pt}", r"  \renewcommand{\arraystretch}{1.15}",
              r"  \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcccc@{}}", r"    \toprule",
              r"    配置 & VF & GV & CGV & Average \\", r"    \midrule"]
    blocks = {0: "A. 分支互补", 3: "B. 先测量后聚合（固定 $G$）", 5: "C. 时间与标量替代（固定 $G$）"}
    for index, (label, *_rest) in enumerate(specs):
        if index in blocks:
            if index:
                lines.append(r"    \addlinespace[3pt]")
            lines.append(rf"    \multicolumn{{5}}{{@{{}}l}}{{\textbf{{{blocks[index]}}}}} \\")
        if specs[index][-1] == "full":
            lines.append(r"    \rowcolor{mainred!4}")
        cells = []
        for j in range(4):
            ranked = grouped_metrics([v[j] for v in values])
            cells.append("/".join(ranked[2 * index:2 * index + 2]))
        lines.append("    " + " & ".join([label, *cells]) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular*}", r"  \par\vspace{2pt}",
              r"  \begin{minipage}{\columnwidth}\scriptsize",
              r"  VF/GV/CGV 为 VideoFeedback/GenVideo/ComGenVid。$G$ 为目标 Global，$L$ 为视频 CDF 后的 Local D2。"
              r"B、C 固定 $G$ 与 FC 窗口，等权融合，各候选重建 CDF。"
              r"同位置数指匹配 pooled 的拟合向量数（至多42/片段）。"
              r"二维 Gaussian 拟合 (TTR, LSMI)；TTR、SPLIT 为同 DINO 控制。",
              r"  \end{minipage}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def reference_table(data: dict[str, list[dict[str, str]]]) -> str:
    rows = data[f"{CONFIRM}/reference_means.csv"]
    repeated = data[f"{CONFIRM}/macro_metrics.csv"]
    per_domain = data[f"{CONFIRM}/dataset_metrics.csv"]
    specs = (("global", "$G$"), ("d1", r"$G+$D1"), ("full", r"$G+$D2"))
    values = []
    standard_deviations = []
    seed_members = None
    for variant, label in specs:
        members = [r for r in repeated if re.fullmatch(rf"s\d+_{variant}", r["variant"])]
        seeds = {r["variant"].split("_", 1)[0] for r in members}
        if len(members) != 5 or len(seeds) != 5:
            raise ValueError(f"新池 {variant} 应恰有五次：{[r['variant'] for r in members]}")
        if seed_members is not None and seeds != seed_members:
            raise ValueError("新池各配置的五次抽样身份不一致")
        seed_members = seeds
        domain_values = []
        for domain in (*DOMAINS, "Average"):
            row = one(rows, dataset=domain, variant=variant)
            selected = members if domain == "Average" else [
                one(per_domain, dataset=domain, variant=r["variant"]) for r in members]
            for source, prefix in zip(METRICS, ("auc", "ap")):
                observations = [float(r[source]) for r in selected]
                equal(float(row[prefix + "_mean"]), mean(observations))
                equal(float(row[prefix + "_sd"]), stdev(observations))
            domain_values.append((float(row["auc_mean"]), float(row["ap_mean"])))
            if domain == "Average":
                standard_deviations.append(
                    rf"{label} {float(row['auc_sd']):.3f}/{float(row['ap_sd']):.3f}")
        for i in range(2):
            equal(domain_values[-1][i], mean(p[i] for p in domain_values[:-1]))
        values.append(domain_values)
    lines = table_start(
        r"更换真实拟合视频后的五次重复。上半为逐域五次平均 AUC/AP-real，Average 为三域等权；"
        r"下半为总体配对差值均值及95\%区间。逐次重拟合后计算指标，不对预测集成。",
        "reference_stability")
    lines += [r"  \setlength{\tabcolsep}{2.2pt}", r"  \renewcommand{\arraystretch}{1.16}",
              r"  \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcccc@{}}", r"    \toprule",
              r"    同池配置 & VF & GV & CGV & Average \\", r"    \midrule"]
    for index, (_, label) in enumerate(specs):
        cells = []
        for j in range(4):
            ranked = grouped_metrics([v[j] for v in values])
            cells.append("/".join(ranked[2 * index:2 * index + 2]))
        lines.append("    " + " & ".join([label, *cells]) + r" \\")
    lines += [r"  \end{tabular*}", r"  \par\vspace{3pt}",
              r"  \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcc@{}}", r"    \midrule",
              r"    Average 增量 & $\Delta$ AUC [95\% CI] & $\Delta$ AP [95\% CI] \\"]
    for contrast, label, baseline in (("new_Full_vs_Global", r"Full$-G$", "global"),
                                      ("new_D2_vs_D1", r"Full$-$D1融合", "d1")):
        cells = []
        for metric, prefix in (("auc", "auc"), ("ap_real", "ap")):
            row = one(data[f"{CONFIRM}/contrasts.csv"], dataset="Average", contrast=contrast, metric=metric)
            delta = float(row["delta"])
            equal(delta, float(one(rows, dataset="Average", variant="full")[prefix + "_mean"])
                  - float(one(rows, dataset="Average", variant=baseline)[prefix + "_mean"]))
            cells.append(rf"\shortstack{{$+{delta:.3f}$\\$[{float(row['ci95_low']):.3f},\,{float(row['ci95_high']):.3f}]$}}")
        lines.append("    " + " & ".join([label, *cells]) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular*}", r"  \par\vspace{2pt}",
              r"  \begin{minipage}{\columnwidth}\scriptsize",
              "  Average 样本标准差（AUC/AP）：" + "，".join(standard_deviations) + "。",
              r"  VF/GV/CGV 同表~\ref{tab:component_ablation}；Full 为 $G+$D2，粗体为上半表各列最高值。"
              r"每域每池200片段，共2357个新真实视频；新池与旧拟合、评价和参考的已知源组隔离。"
              r"五池可重叠，区间条件于这五个已选池。",
              r"  \end{minipage}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只检查，不改写表格")
    args = parser.parse_args()
    data = {path: read_csv(path) for path in HASHES}
    outputs = {"main_results.tex": main_table(data),
               "component_ablation.tex": component_table(data),
               "reference_stability.tex": reference_table(data)}
    for name, contents in outputs.items():
        path = HERE / "tables" / name
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != contents:
                raise ValueError(f"表格与已验收输入不一致：{path}")
        else:
            path.write_text(contents, encoding="utf-8")
    published_digest = hashlib.sha256(json.dumps(published_rows(), ensure_ascii=False).encode()).hexdigest()
    print(f"{'校验' if args.check else '导出'}通过：12 份 CSV 哈希、23 生成器映射、三域等权汇总、五次均值/SD、配对差值、3 张表逐字一致。")
    print(f"发表值转录身份 SHA256：{published_digest}")
    print("冻结 Full AUC/AP-real：0.8744719007139845 / 0.8770747547781038")


if __name__ == "__main__":
    main()
