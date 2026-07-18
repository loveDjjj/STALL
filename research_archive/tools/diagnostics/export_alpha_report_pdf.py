#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "results" / "alpha_stalled_method_report_zh.md"
PDF_PATH = ROOT / "results" / "alpha_stalled_method_report_zh.pdf"


def inline_md(text: str) -> str:
    text = escape(text)
    text = re.sub(r"`([^`]+)`", r'<font name="STSong-Light">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def build_story(lines: list[str], styles) -> list[object]:
    story: list[object] = []
    in_code = False
    code_lines: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            if not in_code:
                in_code = True
                code_lines = []
            else:
                story.append(Preformatted("\n".join(code_lines), styles["CNCode"], maxLineLength=95))
                in_code = False
            i += 1
            continue
        if in_code:
            code_lines.append(line)
            i += 1
            continue
        if not line.strip():
            story.append(Spacer(1, 4))
            i += 1
            continue

        if line.strip().startswith("|") and i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            rows = []
            for idx, table_line in enumerate(table_lines):
                cells = [cell.strip() for cell in table_line.strip().strip("|").split("|")]
                if idx == 1 and all(set(cell.replace(":", "").strip()) <= {"-"} for cell in cells):
                    continue
                rows.append([Paragraph(inline_md(cell), styles["CNBody"]) for cell in cells])
            if rows:
                col_count = max(len(row) for row in rows)
                available = A4[0] - 1.4 * inch
                table = Table(rows, colWidths=[available / col_count] * col_count, repeatRows=1)
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 4),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                            ("TOPPADDING", (0, 0), (-1, -1), 3),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                        ]
                    )
                )
                story.append(table)
                story.append(Spacer(1, 8))
            continue

        if line.startswith("# "):
            story.append(Paragraph(inline_md(line[2:].strip()), styles["CNTitle"]))
        elif line.startswith("## "):
            story.append(Paragraph(inline_md(line[3:].strip()), styles["CNH2"]))
        elif line.startswith("### "):
            story.append(Paragraph(inline_md(line[4:].strip()), styles["CNH3"]))
        elif line.startswith("#### "):
            story.append(Paragraph(inline_md(line[5:].strip()), styles["CNH4"]))
        elif re.match(r"\d+\.\s+", line.strip()):
            story.append(Paragraph(inline_md(line.strip()), styles["CNBullet"]))
        elif line.strip().startswith("- "):
            story.append(Paragraph("• " + inline_md(line.strip()[2:]), styles["CNBullet"]))
        else:
            para = [line.strip()]
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                if (
                    not nxt.strip()
                    or nxt.startswith("#")
                    or nxt.startswith("```")
                    or nxt.strip().startswith("|")
                    or nxt.strip().startswith("- ")
                    or re.match(r"\d+\.\s+", nxt.strip())
                ):
                    break
                para.append(nxt.strip())
                j += 1
            story.append(Paragraph(inline_md(" ".join(para)), styles["CNBody"]))
            i = j
            continue
        i += 1
    return story


def main() -> None:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("CNTitle", fontName="STSong-Light", fontSize=20, leading=26, spaceAfter=14))
    styles.add(ParagraphStyle("CNH2", fontName="STSong-Light", fontSize=15, leading=20, spaceBefore=14, spaceAfter=8))
    styles.add(ParagraphStyle("CNH3", fontName="STSong-Light", fontSize=12.5, leading=17, spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle("CNH4", fontName="STSong-Light", fontSize=11, leading=15, spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle("CNBody", fontName="STSong-Light", fontSize=9.5, leading=14, spaceAfter=5, alignment=TA_LEFT))
    styles.add(
        ParagraphStyle("CNBullet", fontName="STSong-Light", fontSize=9.5, leading=14, leftIndent=14, firstLineIndent=-8, spaceAfter=4)
    )
    styles.add(
        ParagraphStyle(
            "CNCode",
            fontName="STSong-Light",
            fontSize=8,
            leading=10,
            leftIndent=8,
            rightIndent=8,
            backColor=colors.HexColor("#f6f8fa"),
            borderPadding=5,
            spaceBefore=4,
            spaceAfter=6,
        )
    )

    story = build_story(MD_PATH.read_text(encoding="utf-8").splitlines(), styles)
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="Alpha-STALLED 当前方法说明与性能报告",
    )
    doc.build(story)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
