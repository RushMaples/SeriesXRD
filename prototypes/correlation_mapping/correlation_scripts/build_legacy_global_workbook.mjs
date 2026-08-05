import fs from "node:fs/promises";
import path from "node:path";
import {
  FileBlob,
  SpreadsheetFile,
  Workbook,
} from "/Users/stanley/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const ROOT = "/Users/stanley/x-ray";
const RUN = path.resolve(
  process.argv[2]
    || ROOT + "/correlations/results/uote_refinement_legacy_global_per_peak_strict_lower_triangle_20260716",
);
const OUT = path.resolve(
  process.argv[3]
    || RUN + "/UOTe_Legacy_All_Correlations_StrictLowerTriangle_20260716.xlsx",
);
const QA = RUN + "/validation/workbook_qa";

const C = {
  navy: "#17324D", blue: "#2F6F9F", blue2: "#4B86B4", paleBlue: "#DCEAF4",
  teal: "#DCEFEF", paleGreen: "#E2F0D9", green: "#2E7D32", paleAmber: "#FFF2CC",
  amber: "#9C6500", paleRed: "#FCE4D6", red: "#A61B1B", grid: "#D9E2EA",
  text: "#2F3B4A", white: "#FFFFFF", gray: "#F4F7F9",
};

function parseCsv(text) {
  const rows = []; let row = []; let field = ""; let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"') {
        if (text[i + 1] === '"') { field += '"'; i += 1; } else quoted = false;
      } else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") { row.push(field); field = ""; }
    else if (ch === "\n") { row.push(field); if (row.some((v) => v !== "")) rows.push(row); row = []; field = ""; }
    else if (ch !== "\r") field += ch;
  }
  if (field !== "" || row.length) { row.push(field); if (row.some((v) => v !== "")) rows.push(row); }
  return rows;
}

function coerce(value) {
  const s = String(value ?? "").trim();
  if (s === "" || s.toLowerCase() === "nan" || s.toLowerCase() === "none") return null;
  if (s === "True") return true;
  if (s === "False") return false;
  if (/^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(s)) return Number(s);
  return s;
}

async function readCsv(file) {
  const text = await fs.readFile(file, "utf8");
  return parseCsv(text).map((row) => row.map(coerce));
}

function toObjects(rows) {
  const headers = rows[0].map(String);
  return rows.slice(1).map((row) => Object.fromEntries(headers.map((header, i) => [header, row[i] ?? null])));
}

function rectangular(rows) {
  const width = Math.max(...rows.map((row) => row.length));
  return rows.map((row) => row.concat(Array(width - row.length).fill(null)));
}

function strictLowerTriangleDisplay(rows) {
  return rows.map((row, rowIndex) => row.map((value, columnIndex) => {
    if (rowIndex === 0 || columnIndex === 0) return value;
    return columnIndex >= rowIndex ? null : value;
  }));
}

function colName(index) {
  let n = index; let out = "";
  while (n > 0) { const rem = (n - 1) % 26; out = String.fromCharCode(65 + rem) + out; n = Math.floor((n - 1) / 26); }
  return out;
}

function shorten(value) {
  if (typeof value !== "string") return value;
  return value.replace(RUN + "/", "run/")
    .replace(ROOT + "/correlations/UOTe XRD Data Refinement/", "Refinement/")
    .replace(ROOT + "/Data/", "Data/");
}

function shortenColumns(rows, headers) {
  const names = rows[0].map(String); const indices = headers.map((name) => names.indexOf(name)).filter((i) => i >= 0);
  return [rows[0], ...rows.slice(1).map((row) => row.map((value, i) => indices.includes(i) ? shorten(value) : value))];
}

function setTitle(sheet, title, subtitle, lastCol) {
  sheet.showGridLines = false;
  const titleRange = sheet.getRangeByIndexes(0, 0, 1, lastCol); titleRange.merge(); titleRange.values = [[title]];
  titleRange.format.fill = C.navy; titleRange.format.font = { bold: true, color: C.white, size: 16 };
  titleRange.format.rowHeight = 28; titleRange.format.verticalAlignment = "center";
  const sub = sheet.getRangeByIndexes(1, 0, 1, lastCol); sub.merge(); sub.values = [[subtitle]];
  sub.format.fill = C.paleBlue; sub.format.font = { color: C.text, italic: true }; sub.format.wrapText = true;
  sub.format.rowHeight = 33; sub.format.verticalAlignment = "center";
}

function setSection(sheet, row, title, lastCol, fill = C.navy) {
  const range = sheet.getRangeByIndexes(row - 1, 0, 1, lastCol); range.merge(); range.values = [[title]];
  range.format.fill = fill; range.format.font = { bold: true, color: C.white }; range.format.rowHeight = 22;
}

function writeTable(sheet, startRow, startCol, rows, options = {}) {
  const data = rectangular(rows); const rowCount = data.length; const colCount = data[0].length;
  const range = sheet.getRangeByIndexes(startRow - 1, startCol - 1, rowCount, colCount); range.values = data;
  range.format.font = { color: C.text }; range.format.borders = { preset: "all", style: "thin", color: C.grid };
  range.format.verticalAlignment = "center";
  const header = sheet.getRangeByIndexes(startRow - 1, startCol - 1, 1, colCount);
  header.format.fill = options.headerFill || C.blue; header.format.font = { bold: true, color: C.white };
  header.format.wrapText = true; header.format.horizontalAlignment = "center"; header.format.rowHeight = options.headerHeight || 32;
  if (rowCount > 1) {
    const body = sheet.getRangeByIndexes(startRow, startCol - 1, rowCount - 1, colCount);
    body.format.wrapText = Boolean(options.wrapBody); body.format.rowHeight = options.bodyHeight || 19;
  }
  return { startRow, endRow: startRow + rowCount - 1, startCol, endCol: startCol + colCount - 1, headers: data[0] };
}

function applyFormats(sheet, table) {
  const bodyRows = table.endRow - table.startRow; if (bodyRows <= 0) return;
  table.headers.forEach((header, offset) => {
    const name = String(header ?? "");
    const range = sheet.getRangeByIndexes(table.startRow, table.startCol - 1 + offset, bodyRows, 1);
    if (/dd_dp/i.test(name)) range.format.numberFormat = "0.00E+00";
    else if (/similarity|correlation|median|auc|_r$|_r2$|jaccard|pearson|difference|rmse|fraction/i.test(name)) range.format.numberFormat = "0.000";
    else if (/q_|d_A|matched_d|two_theta|azim|halfwidth|fwhm|center/i.test(name)) range.format.numberFormat = "0.00000";
    else if (/pressure|GPa/i.test(name)) range.format.numberFormat = "0.00";
    else if (/area|intensity|counts|exposure|snr/i.test(name)) range.format.numberFormat = "0.000";
    else if (/passed$|included$|exists$|exact$|_count$|frame$|track$|frames$|tracks$|points$|pixels$|bins$|support$/i.test(name)) range.format.numberFormat = "0";
  });
}

function setWidths(sheet, rowCount, widths, defaultWidth = 14) {
  const maxCol = Math.max(...Object.keys(widths).map(Number), 0);
  for (let i = 0; i <= maxCol; i += 1) sheet.getRangeByIndexes(0, i, Math.max(rowCount, 1), 1).format.columnWidth = widths[i] ?? defaultWidth;
}

function fixedColorScale(range, min, mid, max) {
  range.conditionalFormats.add("colorScale", {
    colors: ["#440154", "#21918C", "#FDE725"],
    thresholds: [{ type: "num", value: min }, { type: "num", value: mid }, { type: "num", value: max }],
  });
}

function metricRow(key, system, family, metric, value, unit, note) { return [key, system, family, metric, value, unit, note]; }

async function walk(dir, base = dir) {
  const out = [];
  for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...await walk(full, base));
    else if (entry.isFile()) { const stat = await fs.stat(full); out.push([path.relative(base, full), path.extname(full).slice(1), stat.size]); }
  }
  return out;
}

await fs.mkdir(QA, { recursive: true });
const manifest = JSON.parse(await fs.readFile(RUN + "/run_manifest.json", "utf8"));
const validationReport = JSON.parse(await fs.readFile(RUN + "/validation/validation_report.json", "utf8"));
const guiSummary = JSON.parse(await fs.readFile(RUN + "/validation/gui_crosscheck/crosscheck_summary.json", "utf8"));

const singleRegistry = await readCsv(RUN + "/inputs/single_frame_registry.csv");
const singleSelected = await readCsv(RUN + "/inputs/single_whole_selected.csv");
const powderRegistry = await readCsv(RUN + "/inputs/powder_frame_registry.csv");
const scFrames = await readCsv(RUN + "/single_crystal/per_peak_all_frames/frame_registry.csv");
const scTrack = await readCsv(RUN + "/single_crystal/per_peak_all_frames/track_summary.csv");
const powderTrack = await readCsv(RUN + "/powder/per_peak/track_summary.csv");
const scObs = await readCsv(RUN + "/single_crystal/per_peak_all_frames/track_observations.csv");
const powderObs = await readCsv(RUN + "/powder/per_peak/track_observations.csv");
const validationChecks = await readCsv(RUN + "/validation/validation_checks.csv");
const parity = await readCsv(RUN + "/validation/powder_legacy_parity.csv");
const roiQc = await readCsv(RUN + "/validation/single_roi_extraction_qc.csv");
const guiChecks = await readCsv(RUN + "/validation/gui_crosscheck/gui_crosscheck_checks.csv");
const guiAgreement = await readCsv(RUN + "/validation/gui_crosscheck/gui_pattern_algorithm_agreement.csv");
const guiCoverage = await readCsv(RUN + "/validation/gui_crosscheck/gui_coverage_and_limits.csv");
const powderSourceCheck = await readCsv(RUN + "/validation/gui_crosscheck/powder_track_source_crosscheck.csv");

const windows = [
  ["Single crystal 0°", "sample / separate orientation", await readCsv(RUN + "/single_crystal/whole_and_windows/0deg/across_frames/window_summary.csv")],
  ["Single crystal 10°", "sample / separate orientation", await readCsv(RUN + "/single_crystal/whole_and_windows/10deg/across_frames/window_summary.csv")],
  ["Powder spots", "primary UOTe signal", await readCsv(RUN + "/powder/whole_and_windows/spots/across_frames/window_summary.csv")],
  ["Powder fit", "W/background control", await readCsv(RUN + "/powder/whole_and_windows/fit/across_frames/window_summary.csv")],
];

const matrices = {
  scLoc: await readCsv(RUN + "/single_crystal/per_peak_all_frames/aggregate_location_matrix.csv"),
  scArea: await readCsv(RUN + "/single_crystal/per_peak_all_frames/aggregate_normalized_area_matrix.csv"),
  sc0Whole: await readCsv(RUN + "/single_crystal/whole_and_windows/0deg/whole_pattern/aggregate_matrix.csv"),
  sc10Whole: await readCsv(RUN + "/single_crystal/whole_and_windows/10deg/whole_pattern/aggregate_matrix.csv"),
  powderWhole: await readCsv(RUN + "/powder/whole_and_windows/spots/whole_pattern/aggregate_matrix.csv"),
  powderLoc: await readCsv(RUN + "/powder/per_peak/aggregate_location_matrix.csv"),
  powderArea: await readCsv(RUN + "/powder/per_peak/aggregate_normalized_area_matrix.csv"),
  powderFit: await readCsv(RUN + "/powder/whole_and_windows/fit/whole_pattern/aggregate_matrix.csv"),
};

function pickColumns(rows, columns, labels = columns) {
  const indexes = columns.map((name) => rows[0].map(String).indexOf(name));
  if (indexes.some((index) => index < 0)) {
    const missing = columns.filter((name, index) => indexes[index] < 0);
    throw new Error("Missing columns: " + missing.join(", "));
  }
  return [
    labels,
    ...rows.slice(1).map((row) => indexes.map((index) => row[index] ?? null)),
  ];
}

function adaptiveColorScale(range, reverse = false) {
  range.conditionalFormats.add("colorScale", {
    criteria: reverse
      ? [
          { type: "lowestValue", color: "#63BE7B" },
          { type: "percentile", value: 50, color: "#FFEB84" },
          { type: "highestValue", color: "#F8696B" },
        ]
      : [
          { type: "lowestValue", color: "#F8696B" },
          { type: "percentile", value: 50, color: "#FFEB84" },
          { type: "highestValue", color: "#63BE7B" },
        ],
  });
}

function addPassFormatting(range) {
  range.conditionalFormats.add("cellIs", {
    operator: "equal",
    formula: 1,
    format: { fill: C.paleGreen, font: { bold: true, color: C.green } },
  });
  range.conditionalFormats.add("cellIs", {
    operator: "equal",
    formula: 0,
    format: { fill: C.paleRed, font: { bold: true, color: C.red } },
  });
}

function categoryFor(relativePath) {
  if (relativePath.startsWith("single_crystal/per_peak_all_frames/")) return "Single per-peak global";
  if (relativePath.startsWith("single_crystal/whole_and_windows/")) return "Single whole/windows";
  if (relativePath.startsWith("powder/per_peak/")) return "Powder per-peak";
  if (relativePath.startsWith("powder/whole_and_windows/")) return "Powder whole/windows";
  if (relativePath.startsWith("validation/gui_crosscheck/")) return "GUI/desktop crosscheck";
  if (relativePath.startsWith("validation/")) return "Validation";
  if (relativePath.startsWith("inputs/")) return "Inputs";
  return "Run-level";
}

function matchCount(ndjson) {
  let total = 0;
  for (const line of String(ndjson ?? "").split("\n").filter(Boolean)) {
    try {
      const record = JSON.parse(line);
      if (Array.isArray(record.matches)) total += record.matches.length;
      else if (record.address) total += 1;
    } catch {
      // Preserve inspection output even if a future artifact-tool record is not JSON.
    }
  }
  return total;
}

const singleRegistryDisplay = shortenColumns(singleRegistry, ["file_path"]);
const singleSelectedDisplay = shortenColumns(singleSelected, ["file_path"]);
const scObsDisplay = shortenColumns(scObs, ["raw_tiff"]);
const parityDisplay = shortenColumns(parity, ["current", "reference"]);
const roiQcDisplay = shortenColumns(roiQc, ["raw_tiff"]);

const sw = manifest.single_crystal.whole_and_windows;
const spt = manifest.single_crystal.per_peak_all_frames;
const pw = manifest.powder.whole_and_windows;
const ppt = manifest.powder.per_track;
const guiByFamily = new Map(
  guiSummary.gui_pattern_algorithm_agreement.map((row) => [row.family, row]),
);

const metricRows = [
  metricRow("sc0_whole_r", "Single crystal", "Whole pattern 0°", "r(similarity, |ΔP|)", sw.orientation_0deg.whole_pattern.corr_vs_pressure_gap_r, "r", "11-frame 0° ladder; 55 pairs"),
  metricRow("sc10_whole_r", "Single crystal", "Whole pattern 10°", "r(similarity, |ΔP|)", sw.orientation_10deg.whole_pattern.corr_vs_pressure_gap_r, "r", "11-frame 10° ladder; 55 pairs"),
  metricRow("sc0_across_near", "Single crystal", "Across-frame ACF 0°", "near-pair median", sw.orientation_0deg.across_frames.near_median, "similarity", "Far set unavailable because maximum pressure gap is below 15 GPa"),
  metricRow("sc10_across_near", "Single crystal", "Across-frame ACF 10°", "near-pair median", sw.orientation_10deg.across_frames.near_median, "similarity", "Far set unavailable because maximum pressure gap is below 15 GPa"),
  metricRow("sc0_within_nonoverlap", "Single crystal", "Within-frame ACF 0°", "non-overlap median", sw.orientation_0deg.within_frame.nonoverlap_pair_median, "similarity", "Four non-overlapping windows"),
  metricRow("sc10_within_nonoverlap", "Single crystal", "Within-frame ACF 10°", "non-overlap median", sw.orientation_10deg.within_frame.nonoverlap_pair_median, "similarity", "Four non-overlapping windows"),
  metricRow("sc_per_peak_observations", "Single crystal", "Per-peak global", "raw observations", spt.raw_observations, "rows", "All available Masked observations"),
  metricRow("sc_per_peak_frames", "Single crystal", "Per-peak global", "Masked frames", spt.masked_frames, "frames", "Includes f0011 decompression and f0027 alternate/repeat"),
  metricRow("sc_per_peak_tracks", "Single crystal", "Per-peak global", "global tracks", spt.tracks, "tracks", "Track identity is not split by 0°/10°"),
  metricRow("sc_per_peak_features", "Single crystal", "Per-peak global", "collapsed track-frame features", spt.frame_track_features, "features", "Median collapse within duplicate track-frame groups"),
  metricRow("sc_per_peak_duplicate_groups", "Single crystal", "Per-peak global", "duplicate track-frame groups", spt.duplicate_frame_track_features, "groups", "12 extra raw rows across 11 groups"),
  metricRow("sc_per_peak_ge2", "Single crystal", "Per-peak global", "tracks with at least 2 frames", spt.comparable_tracks_ge2_frames, "tracks", "Have at least one cross-frame comparison"),
  metricRow("sc_per_peak_ge3", "Single crystal", "Per-peak global", "tracks with at least 3 frames", spt.usable_trajectories_ge3_frames, "tracks", "Eligible for a d(P) slope fit"),
  metricRow("sc_per_peak_singletons", "Single crystal", "Per-peak global", "singleton tracks", spt.singleton_tracks, "tracks", "Heatmap retained with annotation; no self-correlation diagonal is shown"),
  metricRow("sc_per_peak_pairs", "Single crystal", "Per-peak global", "unique track-frame-pair comparisons", spt.location_unique_pairs, "comparisons", "653 unique (track, frame_a, frame_b) combinations; not 653 distinct frame-axis pairs"),
  metricRow("sc_cross_orientation_tracks", "Single crystal", "Per-peak global", "tracks crossing 0°/10°", spt.cross_orientation_tracks, "tracks", "Same peak track has observations from both orientations"),
  metricRow("sc_cross_orientation_pairs", "Single crystal", "Per-peak global", "cross-orientation pairs", spt.cross_orientation_location_pairs, "pairs", "Orientation is metadata, not a grouping key"),
  metricRow("sc_heatmaps_each", "Single crystal", "Per-peak global", "heatmaps per family", spt.paired_heatmaps, "files", "75 location, 75 normalized-area, and 75 paired"),
  metricRow("powder_spots_whole_r", "Powder", "Spots whole pattern", "r(similarity, |ΔP|)", pw.spots.whole_pattern.corr_vs_pressure_gap_r, "r", "Primary UOTe-sensitive 1D channel"),
  metricRow("powder_fit_whole_r", "Powder", "Fit whole pattern", "r(similarity, |ΔP|)", pw.fit.whole_pattern.corr_vs_pressure_gap_r, "r", "W/background control"),
  metricRow("powder_spots_across_auc", "Powder", "Spots across-frame ACF", "near-vs-far AUC", pw.spots.across_frames.near_vs_far_auc, "probability", "Near ≤1.5 GPa; far ≥15 GPa"),
  metricRow("powder_fit_across_auc", "Powder", "Fit across-frame ACF", "near-vs-far AUC", pw.fit.across_frames.near_vs_far_auc, "probability", "Control; not UOTe-specific evidence"),
  metricRow("powder_spots_within", "Powder", "Spots within-frame ACF", "non-overlap median", pw.spots.within_frame.nonoverlap_pair_median, "similarity", "Five non-overlapping windows"),
  metricRow("powder_fit_within", "Powder", "Fit within-frame ACF", "non-overlap median", pw.fit.within_frame.nonoverlap_pair_median, "similarity", "Control"),
  metricRow("powder_frames", "Powder", "All Reduced .xy", "included frames", pw.spots.files, "frames", "56 scans and 19 pressures"),
  metricRow("powder_tracks", "Powder", "Masked per-peak", "curated tracks", ppt.tracks, "tracks", "167 observations"),
  metricRow("legacy_parity", "Validation", "Powder frozen legacy", "exact arrays/axes", validationReport.powder_legacy_parity.length, "arrays", "Maximum absolute difference 0"),
  metricRow("roi_jaccard", "Validation", "Single ROI mask", "minimum Jaccard", guiSummary.single_roi_mask_min_jaccard, "fraction", "Threshold ≥0.99"),
  metricRow("gui_whole_r", "GUI crosscheck", "Pattern map", "whole-pattern matrix agreement", guiByFamily.get("whole_pattern").cell_pearson_r, "r", "Only powder spots scan048"),
  metricRow("gui_across_r", "GUI crosscheck", "Pattern map", "across-frame matrix agreement", guiByFamily.get("across_frames_window_acf").cell_pearson_r, "r", "Only powder spots scan048"),
  metricRow("gui_within_r", "GUI crosscheck", "Pattern map", "within-frame matrix agreement", guiByFamily.get("within_frame_window_acf").cell_pearson_r, "r", "Only powder spots scan048"),
  metricRow("powder_source_slope_r", "Validation", "Powder track source", "slope Pearson agreement", guiSummary.powder_track_source_slope_correlation, "r", "10/10 source track metadata exact"),
  metricRow("validation_passed", "Validation", "Overall", "all required checks passed", manifest.validation_passed ? 1 : 0, "boolean", "Numerical, structural, source, GUI, and desktop checks"),
];

const artifactRows = (await walk(RUN))
  .filter((row) => row[0] !== path.basename(OUT))
  .filter((row) => !row[0].startsWith("validation/workbook_qa/"))
  .sort((a, b) => a[0].localeCompare(b[0]))
  .map((row) => [row[0], row[1] || "(none)", row[2], categoryFor(row[0])]);

const workbook = Workbook.create();
const sheetNames = [
  "Summary",
  "Core Metrics",
  "Method & Scope",
  "Input Audit",
  "Window ACF",
  "SC Global Track Index",
  "Powder Track Summary",
  "SC Global Obs",
  "Powder Obs",
  "Validation",
  "Matrix SC Per Peak",
  "Matrix SC Whole",
  "Matrix Powder Spots",
  "Matrix Powder Fit",
  "Artifact Index",
];
const S = Object.fromEntries(sheetNames.map((name) => [name, workbook.worksheets.add(name)]));

const metricsSheet = S["Core Metrics"];
setTitle(
  metricsSheet,
  "UOTe core correlation metrics",
  "Values used by the Summary formulas. Per-peak entries are 0–1 similarities, not Pearson correlations or fitted peak areas.",
  7,
);
const metricsTable = writeTable(
  metricsSheet,
  4,
  1,
  [["Key", "System", "Analysis family", "Metric", "Value", "Unit", "Interpretation / population"], ...metricRows],
  { wrapBody: true, bodyHeight: 29 },
);
applyFormats(metricsSheet, metricsTable);
metricRows.forEach((row, index) => {
  const unit = String(row[5]);
  const excelRow = 5 + index;
  metricsSheet.getRange("E" + excelRow).format.numberFormat =
    /rows|frames|tracks|features|groups|pairs|files|arrays|comparisons|boolean/.test(unit)
      ? "0"
      : "0.000";
});
metricsSheet.freezePanes.freezeRows(4);
metricsSheet.freezePanes.freezeColumns(2);
setWidths(metricsSheet, metricsTable.endRow, { 0: 30, 1: 18, 2: 26, 3: 31, 4: 15, 5: 15, 6: 62 });
const metricMap = new Map(metricRows.map((row, index) => [row[0], 5 + index]));
const metricFormula = (key) => "='Core Metrics'!E" + metricMap.get(key);

const summary = S.Summary;
setTitle(
  summary,
  "UOTe XRD correlation — corrected final result",
  "最重要的区分：0°/10° 是同一单晶的两个测量朝向。整谱按朝向分开；逐峰在全部 12 个 Masked frames 中全局追踪，不按朝向拆。",
  7,
);
setSection(summary, 4, "先看这个：0° / 10° 到底在哪里分开？", 7);
const decisionTable = writeTable(summary, 5, 1, [
  ["分析层次", "0°/10°怎么处理", "原因", "本次执行", "输入", "输出", "结论"],
  ["整谱、跨帧 window ACF、单帧内 window ACF", "分开", "晶体转 10° 后，能看到的峰和相对强度会改变；两组不属于同一测量几何下的连续压力序列", "0° 一套 11 frames；10° 一套 11 frames", "Initial Reduction 连续谱", "各自的矩阵和 heatmaps", "正确地分开"],
  ["逐峰 location / normalized area", "不分开", "peak identity 由全局 track 定义；角度只说明该观测来自哪个取向", "75 tracks 跨全部 12 个 Masked frames", "Masked kept observations", "每个 track 一套跨 frame 矩阵和 heatmaps", "全局追踪"],
  ["GUI / desktop 对照", "按现有覆盖解释", "GUI H5 只覆盖 powder spots scan048；没有 single 或 fit H5", "只直接验证可覆盖的 whole/across/within", "GUI Pattern map / Peak map / Review", "交叉检查表和图", "不夸大 per-peak 验证"],
], { wrapBody: true, bodyHeight: 58 });
summary.getRange("B6:B8").conditionalFormats.add("containsText", {
  text: "不分开",
  format: { fill: C.paleGreen, font: { bold: true, color: C.green } },
});
summary.getRange("B6:B8").conditionalFormats.add("containsText", {
  text: "分开",
  format: { fill: C.paleAmber, font: { bold: true, color: C.amber } },
});
setSection(summary, decisionTable.endRow + 2, "关键数量与结果（公式联动 Core Metrics）", 7);
const kpiStart = decisionTable.endRow + 3;
const kpis = [
  ["Single per-peak Masked frames", metricFormula("sc_per_peak_frames"), "0", "12 frames；包括 whole-pattern 排除的 f0011 和 f0027"],
  ["Single global peak tracks", metricFormula("sc_per_peak_tracks"), "0", "75 个全局 track；不再拆成 0° / 10° 两套身份"],
  ["Single tracks with ≥2 frames", metricFormula("sc_per_peak_ge2"), "0", "49 个 track 有真正的跨 frame comparison"],
  ["Single per-track frame-pair comparisons", metricFormula("sc_per_peak_pairs"), "0", "653 个唯一 (track, frame_a, frame_b) 组合；不是 653 种不同的 frame 轴配对"],
  ["Single cross-orientation tracks", metricFormula("sc_cross_orientation_tracks"), "0", "24 个 track 同时含 0° 和 10° 观测"],
  ["Single 0° whole-pattern r vs |ΔP|", metricFormula("sc0_whole_r"), "0.000", "整谱只在 0° 的 11-frame ladder 内比较"],
  ["Single 10° whole-pattern r vs |ΔP|", metricFormula("sc10_whole_r"), "0.000", "整谱只在 10° 的 11-frame ladder 内比较"],
  ["GUI whole-pattern matrix agreement", metricFormula("gui_whole_r"), "0.000", "仅 powder spots scan048；不是 single per-peak 验证"],
];
const kpiRows = [["Question / metric", "Value", "How to read it"], ...kpis.map((row) => [row[0], null, row[3]])];
const kpiTable = writeTable(summary, kpiStart, 1, kpiRows, { wrapBody: true, bodyHeight: 39 });
kpis.forEach((row, index) => {
  const excelRow = kpiStart + 1 + index;
  summary.getRange("B" + excelRow).formulas = [[row[1]]];
  summary.getRange("B" + excelRow).format.numberFormat = row[2];
});
summary.getRange("A" + (kpiTable.endRow + 2) + ":G" + (kpiTable.endRow + 2)).merge();
summary.getRange("A" + (kpiTable.endRow + 2)).values = [[
  "显示规则：所有 correlation matrix 只显示严格左下三角；对角线和右上三角隐藏。完整数值仍保留在 CSV/NPZ。per-peak score 是 similarity，不是 Pearson correlation。",
]];
summary.getRange("A" + (kpiTable.endRow + 2)).format = {
  fill: C.paleAmber,
  font: { bold: true, color: C.amber },
  wrapText: true,
  rowHeight: 43,
  borders: { preset: "all", style: "thin", color: C.grid },
};
summary.freezePanes.freezeRows(5);
setWidths(summary, kpiTable.endRow + 3, { 0: 34, 1: 21, 2: 59, 3: 29, 4: 30, 5: 31, 6: 29 });

const method = S["Method & Scope"];
setTitle(
  method,
  "Method and scope",
  "Frozen legacy correlation algorithms, corrected single-crystal per-peak population, and explicit interpretation limits.",
  8,
);
setSection(method, 4, "Analysis methods", 8);
const methodTable = writeTable(method, 5, 1, [
  ["Analysis", "Input", "Preprocessing / formula", "Population", "0°/10° rule", "Missing values", "Interpretation", "Status"],
  ["Whole pattern", "Continuous .xy patterns", "Subtract row P5; divide shifted row by P99; SG(9,2)−SG(101,2); row z-score; Pearson", "All common-grid bins", "Single crystal separated into two 11-frame ladders", "Unavailable pairs blank", "Global spectral-shape similarity", "Frozen"],
  ["Across frames", "5° windows stepped by 1°", "Positive-lag FFT ACF; z-score; best Pearson over same window ±1 neighbor", "Same orientation for single; same scan for powder", "Single crystal separated", "Unavailable far set blank", "Local motif persistence across pressure", "Frozen"],
  ["Within frame", "Windows within one frame", "Window-to-window ACF Pearson; non-overlap control", "One measured frame at a time", "No cross-orientation comparison is involved", "Unavailable pairs blank", "Repeated local pattern within one frame", "Frozen"],
  ["Per-peak location", "Masked curated track observations", "clip(1−|Δ2θ|/0.06°,0,1)", "Same global track across measured frames", "Not split; orientation is frame metadata", "NaN; no redetection or zero fill", "0–1 location similarity, not Pearson", "Corrected global population"],
  ["Per-peak normalized area", "Background-subtracted ROI excess", "Single: excess / exposure / effective pixels; score=min/max. Powder: area / pixels / D1s filename exposure", "Same global track across measured frames", "Single not split", "NaN; no invented absence", "0–1 normalized-area similarity, not fitted area", "Single exposure verified; powder exploratory"],
  ["Near / far summary", "Pairwise pressure gaps", "Near ≤1.5 GPa; far ≥15 GPa; Mann–Whitney probability AUC", "Available pairs only", "Single far unavailable", "Reported as blank, not zero", "Ordering probability, not phase probability", "Frozen"],
], { wrapBody: true, bodyHeight: 56 });
setSection(method, methodTable.endRow + 2, "Scope decisions and provenance", 8);
const scopeTable = writeTable(method, methodTable.endRow + 3, 1, [
  ["Item", "Whole-pattern decision", "Per-peak decision", "Reason", "Verified value", "Status", "Source", "Impact"],
  ["0° / 10°", "Separate ladders", "One global track population", "Different geometry affects full spectra; global track identity links the same peak", "22 whole frames; 12 Masked frames", "PASS", "run_manifest.json", "Prevents geometry mixing without losing cross-orientation peak links"],
  ["f0011 decompression", "Excluded", "Included", "Not part of formal loading ladder, but it is a real Masked peak observation", "2.4 GPa, 10° decomp", "PASS", "frame_registry.csv", "Per-peak uses all measured evidence"],
  ["f0027 alternate/repeat", "Excluded", "Included", "Avoid duplicate weighting in whole pattern; preserve actual peak observation", "9.8 GPa, 10° alt", "PASS", "frame_registry.csv", "Per-peak uses all measured evidence"],
  ["Legacy runner", "SHA locked", "Formula-compatible outputs", "Prevents algorithm drift", manifest.legacy_runner_sha256, "PASS", shorten(manifest.legacy_runner), "Exact method identity"],
  ["Powder parity", "Exact", "Source tracks crosschecked", "Proves frozen legacy numerical reproduction", "34 arrays/axes; 10/10 tracks", "PASS", "validation/", "Reproducibility"],
  ["GUI coverage", "Direct for one powder scan", "No direct curated per-peak overlap", "GUI scan048 has no curated powder track frames and no single H5", "19 frames; 1/56 scans", "LIMITED", "gui_crosscheck/", "Do not claim GUI validates single per-peak"],
], { wrapBody: true, bodyHeight: 50 });
method.freezePanes.freezeRows(5);
setWidths(method, scopeTable.endRow, { 0: 24, 1: 33, 2: 34, 3: 44, 4: 31, 5: 16, 6: 33, 7: 46 });

const audit = S["Input Audit"];
setTitle(
  audit,
  "Input audit and exact population",
  "Whole-pattern selection and per-peak population are intentionally different. The table makes every inclusion rule explicit.",
  11,
);
setSection(audit, 4, "Coverage overview", 11);
const auditOverview = writeTable(audit, 5, 1, [
  ["Dataset", "Source rows/files", "Whole included", "Whole excluded", "Per-peak observations", "Per-peak frames", "Tracks", "Scans/orientations", "Pressures", "Exposure", "Status"],
  ["Single Initial Reduction", singleRegistry.length - 1, singleSelected.length - 1, singleRegistry.length - singleSelected.length, null, null, null, "0°: 11; 10°: 11", 11, "not used for amplitude scaling", "Whole selection exact"],
  ["Single Masked", scObs.length - 1, null, null, spt.raw_observations, spt.masked_frames, spt.tracks, "0° / 10° retained as metadata", 12, "29.999 s TIFF metadata", "All available per-peak rows included"],
  ["Powder Reduced .xy", powderRegistry.length - 1, 1060, powderRegistry.length - 1 - 1060, null, null, null, "56 scans", 19, "shape only", "Legacy parity exact"],
  ["Powder Masked tracks", powderObs.length - 1, null, null, ppt.raw_observations, guiSummary.powder_curated_per_peak_frames, ppt.tracks, "26 scans with curated spots", 18, "D1s → 1.0 s assumption", "167/167 coordinate joins"],
], { wrapBody: true, bodyHeight: 38 });
setSection(audit, auditOverview.endRow + 2, "Single per-peak 12-frame registry — global axis used by every track heatmap", 11);
const scFrameTable = writeTable(audit, auditOverview.endRow + 3, 1, scFrames, { wrapBody: true, bodyHeight: 28 });
applyFormats(audit, scFrameTable);
setSection(audit, scFrameTable.endRow + 2, "Exact single whole-pattern selection — 0° and 10° ladders", 11);
const selectedTable = writeTable(audit, scFrameTable.endRow + 3, 1, singleSelectedDisplay, { wrapBody: true, bodyHeight: 25 });
applyFormats(audit, selectedTable);
setSection(audit, selectedTable.endRow + 2, "Full single registry and exclusion reasons", 11);
const registryTable = writeTable(audit, selectedTable.endRow + 3, 1, singleRegistryDisplay, { wrapBody: true, bodyHeight: 26 });
applyFormats(audit, registryTable);
const powderObjects = toObjects(powderRegistry);
const scanMap = new Map();
for (const row of powderObjects) {
  const key = String(row.scan);
  if (!scanMap.has(key)) scanMap.set(key, { total: 0, included: 0, pressures: new Set(), steps: new Set() });
  const record = scanMap.get(key);
  record.total += 1;
  if (Number(row.included) === 1) record.included += 1;
  if (Number(row.included) === 1) record.pressures.add(Number(row.pressure_GPa));
  record.steps.add(String(row.parsed_pressure_step));
}
const powderScanRows = [["Scan", "Manifest rows", "Included frames", "Excluded frames", "Pressure count", "Min pressure (GPa)", "Max pressure (GPa)", "Parsed step count"]];
for (const [scan, record] of [...scanMap.entries()].sort()) {
  const pressures = [...record.pressures].sort((a, b) => a - b);
  powderScanRows.push([scan, record.total, record.included, record.total - record.included, pressures.length, Math.min(...pressures), Math.max(...pressures), record.steps.size]);
}
setSection(audit, registryTable.endRow + 2, "Powder coverage by scan", 11);
const powderScanTable = writeTable(audit, registryTable.endRow + 3, 1, powderScanRows, { bodyHeight: 20 });
applyFormats(audit, powderScanTable);
audit.freezePanes.freezeRows(5);
audit.freezePanes.freezeColumns(2);
setWidths(audit, powderScanTable.endRow, { 0: 24, 1: 20, 2: 19, 3: 19, 4: 22, 5: 20, 6: 18, 7: 26, 8: 18, 9: 33, 10: 40 });

const windowSheet = S["Window ACF"];
setTitle(
  windowSheet,
  "Fixed-window ACF across frames",
  "Legacy 5° windows, 1° step, and ±1 neighboring-window tolerance. Single-crystal whole/window analyses remain separated by orientation.",
  13,
);
const windowObjects = windows.map((group) => group[2].slice(1).map((row) => Object.fromEntries(group[2][0].map((header, index) => [String(header), row[index] ?? null]))));
const bestSpots = windowObjects[2].filter((row) => row.near_vs_far_auc != null).sort((a, b) => Number(b.near_vs_far_auc) - Number(a.near_vs_far_auc))[0];
const bestFit = windowObjects[3].filter((row) => row.near_vs_far_auc != null).sort((a, b) => Number(b.near_vs_far_auc) - Number(a.near_vs_far_auc))[0];
setSection(windowSheet, 4, "Window overview", 13);
const windowOverview = writeTable(windowSheet, 5, 1, [
  ["Dataset", "Window", "AUC", "Role", "0°/10° handling", "Near", "Far", "Width", "Step", "Tolerance", "Population", "Limit", "Status"],
  ["Powder spots", String(bestSpots.start_deg) + "–" + String(bestSpots.end_deg) + "°", bestSpots.near_vs_far_auc, "Primary", "not applicable", "≤1.5 GPa", "≥15 GPa", "5°", "1°", "±1°", "same scan", "descriptive window selection", "Available"],
  ["Powder fit", String(bestFit.start_deg) + "–" + String(bestFit.end_deg) + "°", bestFit.near_vs_far_auc, "Control", "not applicable", "≤1.5 GPa", "≥15 GPa", "5°", "1°", "±1°", "same scan", "not UOTe-specific", "Available"],
  ["Single crystal", "17 windows per orientation", null, "Sample", "0° and 10° separate", "≤1.5 GPa", "≥15 GPa", "5°", "1°", "±1°", "within orientation", "max ΔP <15 GPa", "Far unavailable"],
], { wrapBody: true, bodyHeight: 38 });
applyFormats(windowSheet, windowOverview);
setSection(windowSheet, windowOverview.endRow + 2, "All window-level metrics", 13);
const windowHeader = ["Dataset", "Role", ...windows[0][2][0]];
const windowRows = [windowHeader];
for (const group of windows) {
  for (const row of group[2].slice(1)) windowRows.push([group[0], group[1], ...row]);
}
const windowTable = writeTable(windowSheet, windowOverview.endRow + 3, 1, windowRows, { bodyHeight: 20 });
applyFormats(windowSheet, windowTable);
const aucColumn = windowHeader.indexOf("near_vs_far_auc");
const rColumn = windowHeader.indexOf("score_vs_pressure_gap_r");
if (aucColumn >= 0) adaptiveColorScale(windowSheet.getRangeByIndexes(windowTable.startRow, aucColumn, windowTable.endRow - windowTable.startRow, 1));
if (rColumn >= 0) adaptiveColorScale(windowSheet.getRangeByIndexes(windowTable.startRow, rColumn, windowTable.endRow - windowTable.startRow, 1), true);
windowSheet.freezePanes.freezeRows(windowTable.startRow);
windowSheet.freezePanes.freezeColumns(3);
setWidths(windowSheet, windowTable.endRow, { 0: 22, 1: 28, 2: 17, 3: 14, 4: 18, 5: 18, 6: 18, 7: 18, 8: 18, 9: 19, 10: 20, 11: 22, 12: 18 });

const scTrackDisplay = pickColumns(
  scTrack,
  [
    "track",
    "matched_d_A_reference_median",
    "observed_d_median_A",
    "frame_count",
    "pressure_min_GPa",
    "pressure_max_GPa",
    "orientations_observed",
    "branches_observed",
    "raw_observation_count",
    "duplicate_frame_track_count",
    "status",
    "dd_dp_A_per_GPa",
    "d_slope_r2",
    "finite_location_pairs",
    "finite_normalized_area_pairs",
    "cross_orientation_pair_count",
    "paired_heatmap",
  ],
  [
    "Track",
    "Reference d (Å)",
    "Observed median d (Å)",
    "Frames",
    "P min (GPa)",
    "P max (GPa)",
    "Orientations",
    "Branches",
    "Raw observations",
    "Duplicate groups",
    "Status",
    "dd/dP (Å/GPa)",
    "d(P) R²",
    "Location pairs",
    "Area pairs",
    "Cross-orientation pairs",
    "Paired heatmap",
  ],
);
const scTrackSheet = S["SC Global Track Index"];
setTitle(
  scTrackSheet,
  "Single-crystal global per-peak track index",
  "75 global tracks across all 12 Masked frames. Orientation and branch are observation metadata, not peak-identity split keys.",
  scTrackDisplay[0].length,
);
const scTrackTable = writeTable(scTrackSheet, 4, 1, scTrackDisplay, { wrapBody: true, bodyHeight: 27 });
applyFormats(scTrackSheet, scTrackTable);
scTrackSheet.getRange("N5:P" + scTrackTable.endRow).format.numberFormat = "0";
scTrackSheet.getRange("K5:K" + scTrackTable.endRow).conditionalFormats.add("containsText", {
  text: "singleton",
  format: { fill: C.paleAmber, font: { color: C.amber } },
});
adaptiveColorScale(scTrackSheet.getRange("M5:M" + scTrackTable.endRow));
scTrackSheet.freezePanes.freezeRows(4);
scTrackSheet.freezePanes.freezeColumns(4);
setWidths(scTrackSheet, scTrackTable.endRow, { 0: 10, 1: 18, 2: 21, 3: 12, 4: 16, 5: 16, 6: 21, 7: 20, 8: 19, 9: 18, 10: 25, 11: 18, 12: 15, 13: 17, 14: 17, 15: 23, 16: 48 });

const powderTrackDisplay = pickColumns(
  powderTrack,
  [
    "track",
    "match_hkl",
    "matched_d_A_reference_median",
    "pressure_points",
    "pressure_min_GPa",
    "pressure_max_GPa",
    "scan_count",
    "frame_count",
    "raw_observation_count",
    "duplicate_frame_track_count",
    "trajectory_status",
    "dd_dp_A_per_GPa",
    "d_slope_r2",
    "location_near_median",
    "location_far_median",
    "location_near_vs_far_auc",
    "intensity_near_median",
    "intensity_far_median",
    "intensity_near_vs_far_auc",
  ],
  [
    "Track",
    "hkl",
    "Reference d (Å)",
    "Pressure points",
    "P min (GPa)",
    "P max (GPa)",
    "Scans",
    "Frames",
    "Raw observations",
    "Duplicate groups",
    "Status",
    "dd/dP (Å/GPa)",
    "d(P) R²",
    "Location near",
    "Location far",
    "Location AUC",
    "Area near",
    "Area far",
    "Area AUC",
  ],
);
const powderTrackSheet = S["Powder Track Summary"];
setTitle(
  powderTrackSheet,
  "Powder curated per-peak track summary",
  "Location is primary. Normalized area uses the D1s filename exposure assumption because raw TIFF metadata was unavailable.",
  powderTrackDisplay[0].length,
);
const powderTrackTable = writeTable(powderTrackSheet, 4, 1, powderTrackDisplay, { wrapBody: true, bodyHeight: 28 });
applyFormats(powderTrackSheet, powderTrackTable);
adaptiveColorScale(powderTrackSheet.getRange("M5:M" + powderTrackTable.endRow));
powderTrackSheet.freezePanes.freezeRows(4);
powderTrackSheet.freezePanes.freezeColumns(4);
setWidths(powderTrackSheet, powderTrackTable.endRow, { 0: 10, 1: 11, 2: 18, 3: 17, 4: 16, 5: 16, 6: 12, 7: 12, 8: 19, 9: 18, 10: 16, 11: 18, 12: 15, 13: 17, 14: 17, 15: 16, 16: 17, 17: 17, 18: 16 });

const scObsSheet = S["SC Global Obs"];
setTitle(
  scObsSheet,
  "Single-crystal global Masked observations",
  "275 raw observations from all 12 available Masked frames. Whole-pattern exclusions do not remove per-peak observations.",
  scObsDisplay[0].length,
);
const scObsTable = writeTable(scObsSheet, 4, 1, scObsDisplay, { bodyHeight: 21 });
applyFormats(scObsSheet, scObsTable);
scObsSheet.freezePanes.freezeRows(4);
scObsSheet.freezePanes.freezeColumns(7);
setWidths(scObsSheet, scObsTable.endRow, { 0: 16, 1: 23, 2: 23, 3: 12, 4: 15, 5: 10, 6: 14, 7: 19, 8: 29, 9: 10, 10: 12, 11: 15, 12: 15, 13: 17, 14: 15, 15: 18, 16: 18, 17: 16, 18: 16, 19: 16, 20: 16, 21: 33, 22: 17, 23: 30, 24: 38, 25: 58 });

const powderObsSheet = S["Powder Obs"];
setTitle(
  powderObsSheet,
  "Powder curated Masked observations",
  "167/167 unique coordinate joins. Normalized area is exploratory because exposure comes from the D1s filename token.",
  powderObs[0].length,
);
const powderObsTable = writeTable(powderObsSheet, 4, 1, powderObs, { bodyHeight: 21 });
applyFormats(powderObsSheet, powderObsTable);
powderObsSheet.freezePanes.freezeRows(4);
powderObsSheet.freezePanes.freezeColumns(7);
setWidths(powderObsSheet, powderObsTable.endRow, { 0: 15, 1: 18, 2: 14, 3: 10, 4: 15, 5: 10, 6: 12, 7: 15, 8: 15, 9: 17, 10: 15, 11: 18, 12: 18, 13: 16, 14: 14, 15: 19, 16: 19, 17: 18, 18: 16, 19: 15, 20: 36, 21: 18, 22: 32, 23: 39, 24: 31, 25: 58 });

const validationSheet = S.Validation;
setTitle(
  validationSheet,
  "Validation, GUI comparison, and reproducibility",
  "All required checks passed. GUI coverage is explicitly limited to powder spots scan048; no direct single-crystal per-peak GUI claim is made.",
  18,
);
setSection(validationSheet, 4, "Main run checks", 18);
const checksTable = writeTable(validationSheet, 5, 1, validationChecks, { wrapBody: true, bodyHeight: 27 });
applyFormats(validationSheet, checksTable);
const checksPassedIndex = validationChecks[0].map(String).indexOf("passed");
if (checksPassedIndex >= 0) addPassFormatting(validationSheet.getRangeByIndexes(checksTable.startRow, checksPassedIndex, checksTable.endRow - checksTable.startRow, 1));
setSection(validationSheet, checksTable.endRow + 2, "GUI and desktop checks", 18);
const guiChecksTable = writeTable(validationSheet, checksTable.endRow + 3, 1, guiChecks, { wrapBody: true, bodyHeight: 28 });
applyFormats(validationSheet, guiChecksTable);
const guiPassedIndex = guiChecks[0].map(String).indexOf("passed");
if (guiPassedIndex >= 0) addPassFormatting(validationSheet.getRangeByIndexes(guiChecksTable.startRow, guiPassedIndex, guiChecksTable.endRow - guiChecksTable.startRow, 1));
setSection(validationSheet, guiChecksTable.endRow + 2, "GUI Pattern-map numerical agreement", 18);
const guiAgreementTable = writeTable(validationSheet, guiChecksTable.endRow + 3, 1, guiAgreement, { wrapBody: true, bodyHeight: 30 });
applyFormats(validationSheet, guiAgreementTable);
setSection(validationSheet, guiAgreementTable.endRow + 2, "GUI coverage and limits", 18);
const guiCoverageTable = writeTable(validationSheet, guiAgreementTable.endRow + 3, 1, guiCoverage, { wrapBody: true, bodyHeight: 48 });
setSection(validationSheet, guiCoverageTable.endRow + 2, "Powder source-track crosscheck", 18);
const powderSourceTable = writeTable(validationSheet, guiCoverageTable.endRow + 3, 1, powderSourceCheck, { wrapBody: true, bodyHeight: 32 });
applyFormats(validationSheet, powderSourceTable);
setSection(validationSheet, powderSourceTable.endRow + 2, "Powder exact legacy parity — 34 arrays and axes", 18);
const parityTable = writeTable(validationSheet, powderSourceTable.endRow + 3, 1, parityDisplay, { wrapBody: true, bodyHeight: 38 });
applyFormats(validationSheet, parityTable);
const parityPassedIndex = parity[0].map(String).indexOf("passed");
if (parityPassedIndex >= 0) addPassFormatting(validationSheet.getRangeByIndexes(parityTable.startRow, parityPassedIndex, parityTable.endRow - parityTable.startRow, 1));
setSection(validationSheet, parityTable.endRow + 2, "Single ROI reconstruction QC", 18);
const roiTable = writeTable(validationSheet, parityTable.endRow + 3, 1, roiQcDisplay, { wrapBody: true, bodyHeight: 34 });
applyFormats(validationSheet, roiTable);
validationSheet.freezePanes.freezeRows(5);
validationSheet.freezePanes.freezeColumns(2);
setWidths(validationSheet, roiTable.endRow, { 0: 49, 1: 46, 2: 27, 3: 19, 4: 19, 5: 19, 6: 20, 7: 56, 8: 18, 9: 20, 10: 20, 11: 21, 12: 23, 13: 24, 14: 22, 15: 22, 16: 24, 17: 40 });

function writeMatrixSection(sheet, startRow, title, note, data, colorMode) {
  const displayData = strictLowerTriangleDisplay(data);
  const columns = displayData[0].length;
  setSection(sheet, startRow, title, columns);
  const noteRange = sheet.getRangeByIndexes(startRow, 0, 1, columns);
  noteRange.merge();
  noteRange.values = [[note]];
  noteRange.format.fill = C.paleBlue;
  noteRange.format.font = { italic: true, color: C.text };
  noteRange.format.wrapText = true;
  noteRange.format.rowHeight = 34;
  const table = writeTable(sheet, startRow + 2, 1, displayData, { bodyHeight: 24 });
  if (displayData.length > 1 && displayData[0].length > 1) {
    const body = sheet.getRangeByIndexes(table.startRow, 1, displayData.length - 1, displayData[0].length - 1);
    body.format.numberFormat = "0.000";
    if (colorMode === "fixed01") fixedColorScale(body, 0, 0.5, 1);
    else adaptiveColorScale(body);
  }
  sheet.getRangeByIndexes(table.startRow, 0, displayData.length - 1, 1).format.fill = C.teal;
  sheet.getRangeByIndexes(table.startRow, 0, displayData.length - 1, 1).format.font = { bold: true, color: C.text };
  return table.endRow + 2;
}

function buildMatrixSheet(sheet, title, subtitle, sections) {
  const width = Math.max(...sections.map((section) => section.data[0].length));
  setTitle(sheet, title, subtitle, width);
  let row = 4;
  for (const section of sections) {
    row = writeMatrixSection(sheet, row, section.title, section.note, section.data, section.colorMode);
  }
  sheet.freezePanes.freezeRows(2);
  sheet.freezePanes.freezeColumns(1);
  setWidths(
    sheet,
    row,
    Object.fromEntries(Array.from({ length: width }, (_, index) => [index, index === 0 ? 26 : 11])),
  );
}

buildMatrixSheet(
  S["Matrix SC Per Peak"],
  "Single-crystal global per-peak similarity matrices",
  "Display is strict lower triangle only: diagonal and upper triangle omitted. Exact full symmetric 12×12 matrices remain in CSV/NPZ.",
  [
    {
      title: "Global track-median location similarity",
      note: "Strict lower display, no diagonal. clip(1−|Δ2θ|/0.06°,0,1), median across matching global tracks; 0°/10° remain metadata.",
      data: matrices.scLoc,
      colorMode: "fixed01",
    },
    {
      title: "Global track-median normalized-area similarity",
      note: "Strict lower display, no diagonal. min/max of background-subtracted ROI excess / 29.999 s / effective pixels.",
      data: matrices.scArea,
      colorMode: "fixed01",
    },
  ],
);

buildMatrixSheet(
  S["Matrix SC Whole"],
  "Single-crystal whole-pattern correlation matrices",
  "Strict lower triangle display with no diagonal. The 0° and 10° whole-pattern ladders remain separate because measurement geometry differs.",
  [
    {
      title: "0° whole-pattern Pearson — Initial Reduction",
      note: "Only comparisons within the formal 0° pressure ladder.",
      data: matrices.sc0Whole,
      colorMode: "adaptive",
    },
    {
      title: "10° whole-pattern Pearson — Initial Reduction",
      note: "Only comparisons within the formal 10° pressure ladder.",
      data: matrices.sc10Whole,
      colorMode: "adaptive",
    },
  ],
);

buildMatrixSheet(
  S["Matrix Powder Spots"],
  "Powder sample-channel and curated per-peak matrices",
  "Strict lower triangle display with no diagonal. Spots is primary; per-peak location is primary and normalized area remains exploratory.",
  [
    {
      title: "Whole-pattern Pearson — Reduced .xy spots",
      note: "Median across same-scan matrices; 56 scans and 19 pressure values.",
      data: matrices.powderWhole,
      colorMode: "adaptive",
    },
    {
      title: "Curated track-median location similarity",
      note: "10 curated tracks; fixed 0–1 color scale; missing observations remain blank.",
      data: matrices.powderLoc,
      colorMode: "fixed01",
    },
    {
      title: "Curated track-median normalized-area similarity",
      note: "Area / pixels / D1s filename exposure; fixed 0–1 color scale.",
      data: matrices.powderArea,
      colorMode: "fixed01",
    },
  ],
);

buildMatrixSheet(
  S["Matrix Powder Fit"],
  "Powder fit-channel control matrix",
  "Strict lower triangle display with no diagonal. The W/background-dominated fit channel is a specificity control, not direct UOTe evidence.",
  [
    {
      title: "Whole-pattern Pearson — Reduced .xy fit control",
      note: "Exact frozen-legacy reproduction; median across 56 same-scan matrices.",
      data: matrices.powderFit,
      colorMode: "adaptive",
    },
  ],
);

const artifactSheet = S["Artifact Index"];
setTitle(
  artifactSheet,
  "Artifact index",
  "Every source result present before workbook rendering, grouped by analysis family. Workbook QA files are recorded separately in validation/workbook_qa.",
  4,
);
const artifactTable = writeTable(
  artifactSheet,
  4,
  1,
  [["Relative path", "Extension", "Bytes", "Category"], ...artifactRows],
  { wrapBody: true, bodyHeight: 27 },
);
artifactSheet.getRange("C5:C" + artifactTable.endRow).format.numberFormat = "#,##0";
artifactSheet.freezePanes.freezeRows(4);
artifactSheet.freezePanes.freezeColumns(1);
setWidths(artifactSheet, artifactTable.endRow, { 0: 78, 1: 14, 2: 16, 3: 28 });

const previewRanges = {
  Summary: "A1:G28",
  "Core Metrics": "A1:G36",
  "Method & Scope": "A1:H22",
  "Input Audit": "A1:K28",
  "Window ACF": "A1:M30",
  "SC Global Track Index": "A1:Q28",
  "Powder Track Summary": "A1:S16",
  "SC Global Obs": "A1:Z22",
  "Powder Obs": "A1:Z22",
  Validation: "A1:R34",
  "Matrix SC Per Peak": "A1:M34",
  "Matrix SC Whole": "A1:L34",
  "Matrix Powder Spots": "A1:T70",
  "Matrix Powder Fit": "A1:T28",
  "Artifact Index": "A1:D30",
};

await fs.mkdir(QA, { recursive: true });
const previewFiles = [];
for (let index = 0; index < sheetNames.length; index += 1) {
  const sheetName = sheetNames[index];
  const image = await workbook.render({
    sheetName,
    range: previewRanges[sheetName],
    scale: sheetName.startsWith("Matrix") ? 0.9 : 1.0,
    format: "png",
  });
  const previewPath = path.join(
    QA,
    String(index + 1).padStart(2, "0") + "_" + sheetName.toLowerCase().replace(/[^a-z0-9]+/g, "_") + ".png",
  );
  await fs.writeFile(previewPath, new Uint8Array(await image.arrayBuffer()));
  previewFiles.push(previewPath);
}

const summaryInspection = await workbook.inspect({
  kind: "table",
  range: "Summary!A1:G28",
  include: "values,formulas",
  tableMaxRows: 28,
  tableMaxCols: 7,
  maxChars: 18000,
});
const preExportErrors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!",
  options: { useRegex: true, maxResults: 300 },
  summary: "pre-export formula error scan",
  maxChars: 12000,
});
const preExportErrorCount = matchCount(preExportErrors.ndjson);
if (preExportErrorCount !== 0) throw new Error("Pre-export formula errors: " + preExportErrorCount);

const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(OUT);

const reimported = await SpreadsheetFile.importXlsx(await FileBlob.load(OUT));
const importedSheets = await reimported.inspect({ kind: "sheet", include: "id,name", maxChars: 12000 });
const importedSummary = await reimported.inspect({
  kind: "table",
  range: "Summary!A1:G28",
  include: "values,formulas",
  tableMaxRows: 28,
  tableMaxCols: 7,
  maxChars: 18000,
});
const postImportErrors = await reimported.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!",
  options: { useRegex: true, maxResults: 300 },
  summary: "reimported formula error scan",
  maxChars: 12000,
});
const postImportErrorCount = matchCount(postImportErrors.ndjson);
if (postImportErrorCount !== 0) throw new Error("Reimported formula errors: " + postImportErrorCount);

for (const [name, range] of [
  ["Summary", "A1:G28"],
  ["SC Global Track Index", "A1:Q28"],
  ["Matrix SC Per Peak", "A1:M34"],
  ["Validation", "A1:R34"],
]) {
  const image = await reimported.render({ sheetName: name, range, scale: 1.0, format: "png" });
  const qaName = "reimport_" + name.toLowerCase().replace(/[^a-z0-9]+/g, "_") + ".png";
  await fs.writeFile(path.join(QA, qaName), new Uint8Array(await image.arrayBuffer()));
}

await fs.writeFile(QA + "/summary_inspection.ndjson", summaryInspection.ndjson ?? String(summaryInspection));
await fs.writeFile(QA + "/pre_export_formula_errors.ndjson", preExportErrors.ndjson ?? String(preExportErrors));
await fs.writeFile(QA + "/reimported_sheets.ndjson", importedSheets.ndjson ?? String(importedSheets));
await fs.writeFile(QA + "/reimported_summary.ndjson", importedSummary.ndjson ?? String(importedSummary));
await fs.writeFile(QA + "/reimported_formula_errors.ndjson", postImportErrors.ndjson ?? String(postImportErrors));
await fs.writeFile(
  QA + "/workbook_verification.json",
  JSON.stringify(
    {
      workbook: OUT,
      sheets: sheetNames,
      sheet_count: sheetNames.length,
      pre_export_formula_error_count: preExportErrorCount,
      post_import_formula_error_count: postImportErrorCount,
      rendered_sheet_count: previewFiles.length,
      reimport_rendered_key_sheets: 4,
      source_counts: {
        single_global_observations: scObs.length - 1,
        single_global_tracks: scTrack.length - 1,
        powder_observations: powderObs.length - 1,
        powder_tracks: powderTrack.length - 1,
        artifacts_indexed: artifactRows.length,
      },
    },
    null,
    2,
  ),
);

process.stdout.write(
  JSON.stringify(
    {
      ok: true,
      workbook: OUT,
      sheets: sheetNames.length,
      rendered_sheets: previewFiles.length,
      pre_export_formula_errors: preExportErrorCount,
      post_import_formula_errors: postImportErrorCount,
      qa: QA,
    },
    null,
    2,
  ) + "\n",
);
