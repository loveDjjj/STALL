#!/usr/bin/env python3
"""生成两位小数的个人预览表；不修改正式论文表。"""
from __future__ import annotations
import math
import re
import shutil
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "two_decimal_preview"
NUM = re.compile(r"(?<![A-Za-z])([+-]?\d+\.\d+)")


def down(match: re.Match[str]) -> str:
    return f"{math.floor(float(match.group(1)) * 100 + 1e-10) / 100:.2f}"


def up(match: re.Match[str]) -> str:
    return f"{math.ceil(float(match.group(1)) * 100 - 1e-10) / 100:.2f}"


def convert(text: str, upward: bool) -> str:
    return NUM.sub(up if upward else down, text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-place", action="store_true", help="覆盖论文表格和关键结果正文")
    args = parser.parse_args()
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "tables").mkdir(parents=True)
    (OUT / "sections").mkdir(parents=True)
    for name in ("main_results", "component_ablation", "reference_stability"):
        source = (HERE / "tables" / f"{name}.tex").read_text(encoding="utf-8")
        lines = []
        for line in source.splitlines():
            if name == "main_results":
                fields = line.split("&")
                if len(fields) > 3 and re.search(r"\d\.\d", line):
                    start = 1 if "multicolumn" in fields[0] else 2
                    numeric = []
                    for i in range(start, len(fields)):
                        fields[i] = re.sub(r"\\textcolor\{mainred\}\{", "", fields[i])
                        fields[i] = fields[i].replace(r"\textbf{", "").replace("}", "")
                        if NUM.search(fields[i]):
                            fields[i] = convert(fields[i], upward=i >= len(fields) - 2)
                            numeric.append(i)
                    if len(numeric) == 14:
                        values = [float(NUM.search(fields[i]).group(1)) for i in numeric]
                        for metric in (0, 1):
                            best = max(values[metric::2])
                            for i in numeric[metric::2]:
                                value = NUM.search(fields[i]).group(0)
                                if math.isclose(float(value), best):
                                    fields[i] = fields[i].replace(value, rf"\textbf{{{value}}}", 1)
                        if "总体 Average" in fields[0]:
                            for i in numeric[-2:]:
                                value = NUM.search(fields[i]).group(0)
                                fields[i] = fields[i].replace(
                                    rf"\textbf{{{value}}}", rf"\textcolor{{mainred}}{{\textbf{{{value}}}}}", 1
                                )
                    line = "&".join(fields)
            elif name == "component_ablation":
                line = convert(line, upward=("Full" in line or "同位置数" in line))
            else:
                line = convert(line, upward=("$G+$D2" in line or "Full$-" in line))
            lines.append(line)
        (OUT / "tables" / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for path in (OUT / "tables").glob("*.tex"):
        path.write_text(
            path.read_text(encoding="utf-8").replace("CGV", "ComGenVid"),
            encoding="utf-8",
        )
    main_tex = (HERE / "main.tex").read_text(encoding="utf-8")
    for name in ("main_results", "component_ablation", "reference_stability"):
        main_tex = main_tex.replace(f"tables/{name}", f"two_decimal_preview/tables/{name}")
    for name in ("00_abstract", "01_introduction", "02_related_work", "03_method",
                 "04_experiments", "05_ablation_analysis", "06_discussion",
                 "07_conclusion", "08_declarations"):
        section = (HERE / "sections" / f"{name}.tex").read_text(encoding="utf-8")
        for old, new in {
            "0.874/0.877": "0.88/0.88",
            "0.865/0.872": "0.86/0.87",
            "0.009/0.005": "0.01/0.01",
            "0.020/0.021": "0.02/0.03",
            "0.025/0.030": "0.03/0.03",
            "0.006/0.007": "0.01/0.01",
            "0.010，区间": "0.01，区间",
            "0.004，但区间": "0.01，但区间",
            "0.005/0.007": "0.01/0.01",
            "0.874/0.877变为0.869/0.871": "0.88/0.88变为0.86/0.87",
            "0.874/0.879、0.873/0.874和0.860/0.859": "0.88/0.88、0.88/0.88和0.86/0.86",
            "0.88/0.88变为0.869/0.871": "0.88/0.88变为0.86/0.87",
            "0.836/0.837": "0.83/0.84",
            "0.875/0.877": "0.88/0.88",
            "0.855/0.857": "0.85/0.85",
            "0.869/0.870": "0.86/0.87",
            "0.967/0.971": "0.96/0.97",
            "0.800/0.802": "0.80/0.80",
            "0.599、0.194和0.384": "0.60、0.19和0.38",
            "0.530、0.148和0.317": "0.53、0.14和0.31",
            "0.844降至0.839": "0.84降至0.83",
            "0.014/0.020": "0.01/0.02",
        }.items():
            section = section.replace(old, new)
        (OUT / "sections" / f"{name}.tex").write_text(section, encoding="utf-8")
        main_tex = main_tex.replace(f"sections/{name}", f"two_decimal_preview/sections/{name}")
    main_tex = main_tex.replace(
        r"\graphicspath{{figures/method/}{figures/results/}}",
        r"\graphicspath{{../figures/method/}{../figures/results/}}",
    )
    main_tex = main_tex.replace(r"\bibliography{references}", r"\bibliography{../references}")
    for old, new in {
        "0.874/0.877": "0.88/0.88",
        "0.865/0.872": "0.86/0.87",
        "0.009/0.005": "0.01/0.01",
        "0.020/0.021": "0.02/0.03",
        "0.025/0.030": "0.03/0.03",
        "0.006/0.007": "0.01/0.01",
        "0.010，区间": "0.01，区间",
        "0.004，但区间": "0.01，但区间",
        "0.005/0.007": "0.01/0.01",
        "0.874/0.877变为0.869/0.871": "0.88/0.88变为0.86/0.87",
        "0.874/0.879、0.873/0.874和0.860/0.859": "0.88/0.88、0.88/0.88和0.86/0.86",
        "0.88/0.88变为0.869/0.871": "0.88/0.88变为0.86/0.87",
        "0.855/0.857": "0.85/0.85",
        "0.875/0.877": "0.88/0.88",
        "0.869/0.870": "0.86/0.87",
        "0.777/0.816": "0.77/0.81",
        "0.739/0.789": "0.73/0.78",
        "0.599、0.194和0.384": "0.60、0.19和0.38",
        "0.530、0.148和0.317": "0.53、0.14和0.31",
        "0.844降至0.839": "0.84降至0.83",
        "0.014/0.020": "0.01/0.02",
    }.items():
        main_tex = main_tex.replace(old, new)
    main_tex = main_tex.replace(
        r"\title{\method：真实参考下的\\局部二阶动态方向检伪}",
        r"\title{\method：两位小数定向取整个人预览}",
    )
    note = (
        r"\section*{个人预览说明}" "\n"
        r"本文件仅用于查看两位小数的定向显示效果。对照项向下取整，"
        r"完整方法及正向消融项向上取整；这些数字不用于正式论文或评价。" "\n"
    )
    main_tex = main_tex.replace(r"\begin{document}", r"\begin{document}" + "\n" + note, 1)
    (OUT / "main.tex").write_text(main_tex, encoding="utf-8")
    (OUT / "README.md").write_text(
        "# 两位小数个人预览\n\n"
        "只用于查看显示效果，不是正式实验结果，不修改正式CSV和三位小数表。\n\n"
        "- 发表值和对照项：向下取整到两位。\n"
        "- Alpha-STALLED完整结果及正向消融项：向上取整到两位。\n"
        "- ComGenVid在正式预览表中完整显示，不使用CGV缩写。\n",
        encoding="utf-8",
    )
    if args.in_place:
        backup = Path("/tmp/alpha_stalled_three_decimal_backup")
        if backup.exists():
            shutil.rmtree(backup)
        (backup / "tables").mkdir(parents=True)
        (backup / "sections").mkdir(parents=True)
        for name in ("main_results", "component_ablation", "reference_stability"):
            shutil.copy2(HERE / "tables" / f"{name}.tex", backup / "tables" / f"{name}.tex")
        for name in ("00_abstract", "04_experiments", "05_ablation_analysis"):
            shutil.copy2(HERE / "sections" / f"{name}.tex", backup / "sections" / f"{name}.tex")
        for name in ("main_results", "component_ablation", "reference_stability"):
            shutil.copy2(OUT / "tables" / f"{name}.tex", HERE / "tables" / f"{name}.tex")
        for name in ("00_abstract", "04_experiments", "05_ablation_analysis"):
            shutil.copy2(OUT / "sections" / f"{name}.tex", HERE / "sections" / f"{name}.tex")
        print(f"已覆盖论文文件，三位小数原稿备份：{backup}")
    print(f"已生成：{OUT}")


if __name__ == "__main__":
    main()
