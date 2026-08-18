#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const manuscriptPath = path.join(root, "results", "alpha_stalled_project_manuscript_zh.md");
const outputDir = path.join(
  root,
  "paper",
  "ieee_alpha_stalled",
  "tables",
  "generated_manuscript"
);
const markdown = fs.readFileSync(manuscriptPath, "utf8");

fs.mkdirSync(outputDir, { recursive: true });

function escapeXml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function stripMarkup(value) {
  return String(value)
    .replaceAll("**", "")
    .replaceAll("<u>", "")
    .replaceAll("</u>", "")
    .trim();
}

function splitMarkdownRow(line) {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

function tableAfter(marker) {
  const markerIndex = markdown.indexOf(marker);
  if (markerIndex < 0) throw new Error(`Marker not found: ${marker}`);
  const lines = markdown.slice(markerIndex).split(/\r?\n/);
  const start = lines.findIndex((line) => line.trim().startsWith("|"));
  if (start < 0) throw new Error(`Table not found after: ${marker}`);
  const tableLines = [];
  for (let i = start; i < lines.length; i += 1) {
    if (!lines[i].trim().startsWith("|")) break;
    tableLines.push(lines[i]);
  }
  if (tableLines.length < 3) throw new Error(`Incomplete table after: ${marker}`);
  return {
    header: splitMarkdownRow(tableLines[0]),
    rows: tableLines.slice(2).map(splitMarkdownRow),
  };
}

function metricPart(raw, inherited = {}) {
  const text = String(raw).trim();
  return {
    value: stripMarkup(text),
    bold: inherited.bold || text.includes("**"),
    underline: inherited.underline || text.includes("<u>") || text.includes("</u>"),
  };
}

function metricPair(raw) {
  const cell = String(raw).trim();
  if (cell === "--" || cell === "—") {
    return [metricPart("--"), metricPart("--")];
  }
  const pairBold = /^\*\*[^*]+\s\/\s[^*]+\*\*$/.test(cell);
  const pairUnderline = /^<u>.*\s\/\s.*<\/u>$/.test(cell);
  const cleaned = pairBold
    ? cell.slice(2, -2)
    : pairUnderline
      ? cell.slice(3, -4)
      : cell;
  const parts = cleaned.split(/\s+\/\s+/);
  if (parts.length === 1) return [metricPart(parts[0]), metricPart(parts[0])];
  return [
    metricPart(parts[0], { bold: pairBold, underline: pairUnderline }),
    metricPart(parts[1], { bold: pairBold, underline: pairUnderline }),
  ];
}

function svgText(x, y, value, options = {}) {
  const {
    anchor = "middle",
    size = 16,
    weight = 400,
    underline = false,
    family = "Times New Roman, Noto Serif CJK SC, serif",
    fill = "#111111",
  } = options;
  const text = `<text x="${x}" y="${y}" text-anchor="${anchor}" dominant-baseline="middle" font-family="${family}" font-size="${size}" font-weight="${weight}" fill="${fill}">${escapeXml(value)}</text>`;
  if (!underline) return text;
  const estimatedWidth = Math.max(size * 0.45, String(value).length * size * 0.47);
  const startX = anchor === "start" ? x : x - estimatedWidth / 2;
  const endX = anchor === "start" ? x + estimatedWidth : x + estimatedWidth / 2;
  return `${text}\n${svgLine(startX, y + size * 0.48, endX, y + size * 0.48, 1.05, fill)}`;
}

function svgLine(x1, y1, x2, y2, width = 1, color = "#222222") {
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}"/>`;
}

function svgRect(x, y, width, height, fill = "#f2f2f2") {
  return `<rect x="${x}" y="${y}" width="${width}" height="${height}" fill="${fill}"/>`;
}

function svgDocument(width, height, body, label) {
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeXml(label)}">
<rect width="100%" height="100%" fill="#ffffff"/>
${body.join("\n")}
</svg>
`;
}

function renderMainTable() {
  const datasets = [
    { name: "VideoFeedback", ...tableAfter("#### VideoFeedback") },
    { name: "GenVideo", ...tableAfter("#### GenVideo") },
    { name: "ComGenVid", ...tableAfter("#### ComGenVid") },
  ];
  const summary = tableAfter("#### 跨基准汇总");
  const methods = datasets[0].header.slice(1);
  const margin = 14;
  const benchmarkWidth = 205;
  const modelWidth = 238;
  const metricWidth = 68;
  const headerHeights = [42, 46, 40];
  const rowHeight = 34;
  const totalRows = datasets.reduce((sum, dataset) => sum + dataset.rows.length, 0) + summary.rows.length;
  const width = margin * 2 + benchmarkWidth + modelWidth + methods.length * metricWidth * 2;
  const headerHeight = headerHeights.reduce((sum, value) => sum + value, 0);
  const height = margin * 2 + headerHeight + totalRows * rowHeight;
  const body = [];
  const xBenchmarkEnd = margin + benchmarkWidth;
  const xModelEnd = xBenchmarkEnd + modelWidth;
  const xImageEnd = xModelEnd + 3 * metricWidth * 2;
  const tableRight = width - margin;
  const tableTop = margin;
  const dataTop = tableTop + headerHeight;

  body.push(svgLine(margin, tableTop, tableRight, tableTop, 2.6));
  body.push(svgText(margin + 14, tableTop + headerHeight / 2, "Benchmark", { anchor: "start", size: 21, weight: 600 }));
  body.push(svgText(xBenchmarkEnd + modelWidth / 2, tableTop + headerHeight / 2, "Model", { size: 21, weight: 600 }));
  body.push(svgText(xModelEnd + 3 * metricWidth, tableTop + headerHeights[0] / 2, "Image Detectors", { size: 22, weight: 600 }));
  body.push(svgText(xImageEnd + 4 * metricWidth, tableTop + headerHeights[0] / 2, "Video Detectors", { size: 22, weight: 600 }));

  const methodY = tableTop + headerHeights[0] + headerHeights[1] / 2;
  methods.forEach((method, index) => {
    const x = xModelEnd + index * metricWidth * 2 + metricWidth;
    body.push(svgText(x, methodY, stripMarkup(method), { size: method === "Alpha-STALLED" ? 15 : 17, weight: 600 }));
  });

  const metricTop = tableTop + headerHeights[0] + headerHeights[1];
  methods.forEach((_, methodIndex) => {
    const aucX = xModelEnd + methodIndex * metricWidth * 2;
    body.push(svgRect(aucX + 2, metricTop + 1, metricWidth - 4, headerHeights[2] - 2));
    body.push(svgText(aucX + metricWidth / 2, metricTop + headerHeights[2] / 2, "AUC", { size: 17, weight: 600 }));
    body.push(svgText(aucX + metricWidth + metricWidth / 2, metricTop + headerHeights[2] / 2, "AP", { size: 17, weight: 600 }));
  });

  body.push(svgLine(xBenchmarkEnd, tableTop + 8, xBenchmarkEnd, height - margin, 1));
  body.push(svgLine(xModelEnd - 4, tableTop + 8, xModelEnd - 4, height - margin, 1.1));
  body.push(svgLine(xModelEnd + 2, tableTop + 8, xModelEnd + 2, height - margin, 1.1));
  body.push(svgLine(xImageEnd, tableTop + 8, xImageEnd, height - margin, 1.2));
  methods.forEach((_, methodIndex) => {
    const x = xModelEnd + (methodIndex + 1) * metricWidth * 2;
    if (methodIndex < methods.length - 1 && x !== xImageEnd) body.push(svgLine(x, tableTop + headerHeights[0], x, height - margin, 0.65, "#555555"));
  });
  body.push(svgLine(margin, dataTop - 4, tableRight, dataTop - 4, 1.2));
  body.push(svgLine(margin, dataTop + 2, tableRight, dataTop + 2, 1.2));

  let y = dataTop;
  for (const dataset of datasets) {
    const averageIndex = dataset.rows.findIndex((row) => stripMarkup(row[0]) === "Average");
    const generatorCount = averageIndex >= 0 ? averageIndex : dataset.rows.length;
    body.push(svgLine(margin, y, tableRight, y, 1.1));
    body.push(svgText(margin + 14, y + generatorCount * rowHeight / 2, dataset.name, { anchor: "start", size: 19, weight: 500 }));

    dataset.rows.forEach((row, rowIndex) => {
      const rowY = y + rowIndex * rowHeight;
      const isAverage = stripMarkup(row[0]) === "Average";
      if (isAverage) body.push(svgLine(margin, rowY, tableRight, rowY, 1.1));
      body.push(svgText(xBenchmarkEnd + 12, rowY + rowHeight / 2, stripMarkup(row[0]), {
        anchor: "start",
        size: 17,
        weight: isAverage ? 600 : 500,
      }));
      row.slice(1).forEach((cell, methodIndex) => {
        const metrics = metricPair(cell);
        metrics.forEach((metric, metricIndex) => {
          const cellX = xModelEnd + (methodIndex * 2 + metricIndex) * metricWidth;
          if (metricIndex === 0) body.push(svgRect(cellX + 2, rowY + 2, metricWidth - 4, rowHeight - 4));
          body.push(svgText(cellX + metricWidth / 2, rowY + rowHeight / 2, metric.value, {
            size: 16,
            weight: metric.bold ? 750 : (isAverage ? 550 : 400),
            underline: metric.underline,
          }));
        });
      });
    });
    y += dataset.rows.length * rowHeight;
    body.push(svgLine(margin, y, tableRight, y, 1.3));
  }

  body.push(svgLine(margin, y + 3, tableRight, y + 3, 1.1));
  summary.rows.forEach((row) => {
    body.push(svgText(margin + 14, y + rowHeight / 2, "All Benchmarks", { anchor: "start", size: 17, weight: 600 }));
    body.push(svgText(xBenchmarkEnd + 12, y + rowHeight / 2, "Average", { anchor: "start", size: 17, weight: 600 }));
    row.slice(1).forEach((cell, methodIndex) => {
      metricPair(cell).forEach((metric, metricIndex) => {
        const cellX = xModelEnd + (methodIndex * 2 + metricIndex) * metricWidth;
        if (metricIndex === 0) body.push(svgRect(cellX + 2, y + 2, metricWidth - 4, rowHeight - 4));
        body.push(svgText(cellX + metricWidth / 2, y + rowHeight / 2, metric.value, {
          size: 16,
          weight: metric.bold ? 750 : 550,
          underline: metric.underline,
        }));
      });
    });
    y += rowHeight;
  });
  body.push(svgLine(margin, y, tableRight, y, 2.6));

  return svgDocument(width, height, body, "Zero-shot detection results on VideoFeedback, GenVideo, and ComGenVid");
}

function renderAblationTable(marker, options) {
  const table = tableAfter(marker);
  const {
    outputName,
    benchmarkColumn,
    leadingWidths,
    label,
  } = options;
  const margin = 14;
  const metricWidth = 92;
  const headerRowHeight = 44;
  const rowHeight = 40;
  const leadingCount = leadingWidths.length;
  const width = margin * 2 + leadingWidths.reduce((sum, value) => sum + value, 0) + metricWidth * 6;
  const height = margin * 2 + headerRowHeight * 2 + rowHeight * table.rows.length;
  const body = [];
  const tableRight = width - margin;
  const leadingX = [margin];
  for (const value of leadingWidths) leadingX.push(leadingX.at(-1) + value);
  const metricsX = leadingX.at(-1);
  const dataTop = margin + headerRowHeight * 2;

  body.push(svgLine(margin, margin, tableRight, margin, 2.5));
  table.header.slice(0, leadingCount).forEach((header, index) => {
    body.push(svgText(leadingX[index] + 12, margin + headerRowHeight, stripMarkup(header), { anchor: "start", size: 18, weight: 600 }));
  });
  const groupHeaders = table.header.slice(leadingCount);
  groupHeaders.forEach((header, index) => {
    body.push(svgText(metricsX + index * metricWidth * 2 + metricWidth, margin + headerRowHeight / 2, stripMarkup(header), { size: 18, weight: 600 }));
    const aucX = metricsX + index * metricWidth * 2;
    body.push(svgRect(aucX + 2, margin + headerRowHeight + 2, metricWidth - 4, headerRowHeight - 4));
    body.push(svgText(aucX + metricWidth / 2, margin + headerRowHeight * 1.5, "AUC", { size: 16, weight: 600 }));
    body.push(svgText(aucX + metricWidth * 1.5, margin + headerRowHeight * 1.5, "AP", { size: 16, weight: 600 }));
  });
  leadingX.slice(1).forEach((x, index) => {
    const doubleLine = index === leadingX.length - 2;
    body.push(svgLine(x - (doubleLine ? 3 : 0), margin + 7, x - (doubleLine ? 3 : 0), height - margin, 0.9));
    if (doubleLine) body.push(svgLine(x + 3, margin + 7, x + 3, height - margin, 0.9));
  });
  groupHeaders.forEach((_, index) => {
    const x = metricsX + (index + 1) * metricWidth * 2;
    if (index < groupHeaders.length - 1) body.push(svgLine(x, margin + headerRowHeight, x, height - margin, 0.7, "#555555"));
  });
  body.push(svgLine(margin, dataTop - 3, tableRight, dataTop - 3, 1.1));
  body.push(svgLine(margin, dataTop + 2, tableRight, dataTop + 2, 1.1));

  let lastBenchmark = null;
  table.rows.forEach((row, rowIndex) => {
    const y = dataTop + rowIndex * rowHeight;
    if (benchmarkColumn) {
      const benchmark = stripMarkup(row[0]);
      if (benchmark !== lastBenchmark) {
        body.push(svgLine(margin, y, tableRight, y, 1.1));
        const span = table.rows.filter((candidate) => stripMarkup(candidate[0]) === benchmark).length;
        body.push(svgText(margin + 12, y + span * rowHeight / 2, benchmark, { anchor: "start", size: 17, weight: 500 }));
        lastBenchmark = benchmark;
      }
    }
    for (let index = benchmarkColumn ? 1 : 0; index < leadingCount; index += 1) {
      body.push(svgText(leadingX[index] + 12, y + rowHeight / 2, stripMarkup(row[index]), {
        anchor: "start",
        size: index === leadingCount - 1 ? 15 : 17,
        weight: stripMarkup(row[index]) === "Alpha-STALLED" ? 650 : 400,
      }));
    }
    row.slice(leadingCount).forEach((cell, groupIndex) => {
      metricPair(cell).forEach((metric, metricIndex) => {
        const x = metricsX + (groupIndex * 2 + metricIndex) * metricWidth;
        if (metricIndex === 0) body.push(svgRect(x + 2, y + 2, metricWidth - 4, rowHeight - 4));
        body.push(svgText(x + metricWidth / 2, y + rowHeight / 2, metric.value, {
          size: 16,
          weight: metric.bold ? 750 : 400,
          underline: metric.underline,
        }));
      });
    });
  });
  body.push(svgLine(margin, height - margin, tableRight, height - margin, 2.5));

  fs.writeFileSync(path.join(outputDir, outputName), svgDocument(width, height, body, label));
}

fs.writeFileSync(path.join(outputDir, "zero_shot_detection.svg"), renderMainTable());
renderAblationTable("**表 2. Global/Patch 组件消融。**", {
  outputName: "component_ablation.svg",
  benchmarkColumn: true,
  leadingWidths: [180, 220, 320],
  label: "Global and patch component ablation",
});
renderAblationTable("**表 3. 局部时序定义消融。**", {
  outputName: "temporal_ablation.svg",
  benchmarkColumn: false,
  leadingWidths: [270, 350],
  label: "Local temporal evidence ablation",
});

console.log(`Rendered manuscript tables to ${outputDir}`);
