import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const ROOT = "/Users/stanley/x-ray";
const RUN = path.resolve(
  process.argv[2] || ROOT + "/correlations/results/uote_uncertainty_aware_correlations_v3_20260719",
);
const OUT = path.resolve(
  process.argv[3] || RUN + "/UOTe_Uncertainty_Aware_Correlations_v3_20260719.xlsx",
);
const QA = RUN + "/validation/workbook_qa";

const C = {
  navy: "#17324D", blue: "#2F6F9F", blue2: "#4B86B4", paleBlue: "#DCEAF4",
  teal: "#DCEFEF", paleGreen: "#E2F0D9", green: "#2E7D32", paleAmber: "#FFF2CC",
  amber: "#9C6500", paleRed: "#FCE4D6", red: "#A61B1B", grid: "#D9E2EA",
  text: "#2F3B4A", white: "#FFFFFF", gray: "#F4F7F9", purple: "#E4DFEC",
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
    else if (ch === "\n") {
      row.push(field); if (row.some((value) => value !== "")) rows.push(row);
      row = []; field = "";
    } else if (ch !== "\r") field += ch;
  }
  if (field !== "" || row.length) {
    row.push(field); if (row.some((value) => value !== "")) rows.push(row);
  }
  return rows;
}

function coerce(value) {
  const s = String(value ?? "").trim();
  if (s === "" || ["nan", "none", "null"].includes(s.toLowerCase())) return null;
  if (s === "True" || s === "true") return true;
  if (s === "False" || s === "false") return false;
  if (/^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(s)) return Number(s);
  return s;
}

async function readCsv(file) {
  return parseCsv(await fs.readFile(file, "utf8")).map((row) => row.map(coerce));
}

function objects(rows) {
  const headers = rows[0].map(String);
  return rows.slice(1).map((row) => Object.fromEntries(headers.map((header, i) => [header, row[i] ?? null])));
}

function rectangular(rows) {
  const width = Math.max(...rows.map((row) => row.length));
  return rows.map((row) => row.concat(Array(width - row.length).fill(null)));
}

function pick(rows, columns, labels = columns) {
  const headers = rows[0].map(String);
  const indexes = columns.map((column) => headers.indexOf(column));
  if (indexes.some((index) => index < 0)) {
    throw new Error("Missing columns: " + columns.filter((_, i) => indexes[i] < 0).join(", "));
  }
  return [labels, ...rows.slice(1).map((row) => indexes.map((index) => row[index] ?? null))];
}

function strictLower(rows) {
  return rows.map((row, rowIndex) => row.map((value, colIndex) => {
    if (rowIndex === 0 || colIndex === 0) return value;
    return colIndex >= rowIndex ? null : value;
  }));
}

function setTitle(sheet, title, subtitle, columns) {
  sheet.showGridLines = false;
  const titleRange = sheet.getRangeByIndexes(0, 0, 1, columns);
  titleRange.merge(); titleRange.values = [[title]];
  titleRange.format.fill = C.navy;
  titleRange.format.font = { bold: true, color: C.white, size: 16 };
  titleRange.format.rowHeight = 29; titleRange.format.verticalAlignment = "center";
  const sub = sheet.getRangeByIndexes(1, 0, 1, columns);
  sub.merge(); sub.values = [[subtitle]];
  sub.format.fill = C.paleBlue; sub.format.font = { italic: true, color: C.text };
  sub.format.wrapText = true; sub.format.rowHeight = 36; sub.format.verticalAlignment = "center";
}

function setSection(sheet, row, title, columns, fill = C.navy) {
  const range = sheet.getRangeByIndexes(row - 1, 0, 1, columns);
  range.merge(); range.values = [[title]];
  range.format.fill = fill; range.format.font = { bold: true, color: C.white };
  range.format.rowHeight = 23;
}

function writeTable(sheet, startRow, startCol, rows, options = {}) {
  const data = rectangular(rows);
  const range = sheet.getRangeByIndexes(startRow - 1, startCol - 1, data.length, data[0].length);
  range.values = data;
  range.format.font = { color: C.text };
  range.format.borders = { preset: "all", style: "thin", color: C.grid };
  range.format.verticalAlignment = "center";
  const header = sheet.getRangeByIndexes(startRow - 1, startCol - 1, 1, data[0].length);
  header.format.fill = options.headerFill || C.blue;
  header.format.font = { bold: true, color: C.white };
  header.format.wrapText = true; header.format.horizontalAlignment = "center";
  header.format.rowHeight = options.headerHeight || 34;
  if (data.length > 1) {
    const body = sheet.getRangeByIndexes(startRow, startCol - 1, data.length - 1, data[0].length);
    body.format.wrapText = Boolean(options.wrapBody);
    body.format.rowHeight = options.bodyHeight || 21;
  }
  return {
    startRow, endRow: startRow + data.length - 1,
    startCol, endCol: startCol + data[0].length - 1, headers: data[0],
  };
}

function applyFormats(sheet, table) {
  const rows = table.endRow - table.startRow;
  if (rows <= 0) return;
  table.headers.forEach((header, offset) => {
    const name = String(header ?? "");
    const range = sheet.getRangeByIndexes(table.startRow, table.startCol - 1 + offset, rows, 1);
    if (/sha256|path|status|reason|interpretation|source|formula|method/i.test(name)) return;
    if (/passed|eligible|candidate|confounded|included|identifiable|flag/i.test(name)) range.format.numberFormat = "0";
    else if (/frame|track|pairs|scans|windows|files|groups|observations|rows|count|support/i.test(name)) range.format.numberFormat = "0";
    else if (/pressure|GPa|two_theta|fwhm|width|shift|tau|coefficient|median|auc|correlation|partial|fraction|similarity|gain|change|excess|r$/i.test(name)) range.format.numberFormat = "0.000";
    else if (/bytes/i.test(name)) range.format.numberFormat = "#,##0";
  });
}

function setWidths(sheet, rowCount, widths, fallback = 14) {
  const max = Math.max(...Object.keys(widths).map(Number), 0);
  for (let i = 0; i <= max; i += 1) {
    sheet.getRangeByIndexes(0, i, Math.max(rowCount, 1), 1).format.columnWidth = widths[i] ?? fallback;
  }
}

function colorScale(range, fixed01 = false) {
  range.conditionalFormats.add("colorScale", fixed01 ? {
    colors: ["#F8696B", "#FFEB84", "#63BE7B"],
    thresholds: [{ type: "num", value: 0 }, { type: "num", value: 0.5 }, { type: "num", value: 1 }],
  } : {
    criteria: [
      { type: "lowestValue", color: "#F8696B" },
      { type: "percentile", value: 50, color: "#FFEB84" },
      { type: "highestValue", color: "#63BE7B" },
    ],
  });
}

function statusFormatting(range) {
  for (const [text, fill, color] of [
    ["PASS", C.paleGreen, C.green], ["FORMAL", C.paleGreen, C.green],
    ["SECONDARY", C.paleAmber, C.amber], ["DIAGNOSTIC", C.paleAmber, C.amber],
    ["NA", C.gray, C.text], ["QC FAIL", C.paleRed, C.red],
    ["NOT CANDIDATE", C.paleRed, C.red],
  ]) {
    range.conditionalFormats.add("containsText", {
      text, format: { fill, font: { bold: true, color } },
    });
  }
}

function metric(key, system, family, name, value, unit, role, note) {
  return [key, system, family, name, value, unit, role, note];
}

function matchCount(ndjson) {
  let total = 0;
  for (const line of String(ndjson ?? "").split("\n").filter(Boolean)) {
    try {
      const record = JSON.parse(line);
      if (Array.isArray(record.matches)) total += record.matches.length;
      else if (record.address) total += 1;
    } catch { /* future records may not be JSON */ }
  }
  return total;
}

await fs.mkdir(QA, { recursive: true });
const runManifest = JSON.parse(await fs.readFile(RUN + "/run_manifest.json", "utf8"));
const summaryMetrics = JSON.parse(await fs.readFile(RUN + "/summary_metrics.json", "utf8"));
const methodConfig = JSON.parse(await fs.readFile(RUN + "/method_config.json", "utf8"));
const validationReport = JSON.parse(await fs.readFile(RUN + "/validation/validation_report.json", "utf8"));
const validationChecks = await readCsv(RUN + "/validation/validation_checks.csv");
const inputAudit = await readCsv(RUN + "/input_audit.csv");
const artifactIndex = await readCsv(RUN + "/analysis_output_manifest.csv");
const singleTrack = await readCsv(RUN + "/single_crystal/per_peak/track_summary.csv");
const powderTrack = await readCsv(RUN + "/powder/per_peak/track_summary.csv");
const singleFeatures = await readCsv(RUN + "/single_crystal/per_peak/frame_track_features.csv");
const powderFeatures = await readCsv(RUN + "/powder/per_peak/frame_track_features.csv");
const powderAreaGroups = await readCsv(RUN + "/powder/per_peak/area_same_pressure_repeatability_groups.csv");
const powderLocationGroups = await readCsv(RUN + "/powder/per_peak/location_same_pressure_repeatability_groups.csv");
const boundaryCandidates = await readCsv(RUN + "/window_to_window/boundary_candidates.csv");
const controlScans = await readCsv(RUN + "/whole_pattern/powder_control_adjustment/scan_level_control_metrics.csv");
const presenceStates = await readCsv(RUN + "/presence/presence_state_long.csv");

const windowLabels = ["single_0deg", "single_10deg", "powder_spots", "powder_fit"];
const windowSummaryRows = [["Series", ...(await readCsv(RUN + "/windows/single_0deg/window_summary.csv"))[0]]];
const windowSensitivityRows = [["Series", ...(await readCsv(RUN + "/windows/single_0deg/shift_bound_sensitivity.csv"))[0]]];
for (const label of windowLabels) {
  const rows = await readCsv(RUN + "/windows/" + label + "/window_summary.csv");
  for (const row of rows.slice(1)) windowSummaryRows.push([label, ...row]);
  const sensitivity = await readCsv(RUN + "/windows/" + label + "/shift_bound_sensitivity.csv");
  for (const row of sensitivity.slice(1)) windowSensitivityRows.push([label, ...row]);
}

const matrices = {
  singleLocation: await readCsv(RUN + "/single_crystal/per_peak/aggregate_location_profile_matrix.csv"),
  singleArea: await readCsv(RUN + "/single_crystal/per_peak/aggregate_area_matrix.csv"),
  powderLocation: await readCsv(RUN + "/powder/per_peak/aggregate_location_profile_matrix.csv"),
  powderArea: await readCsv(RUN + "/powder/per_peak/aggregate_area_matrix.csv"),
  sc0Whole: await readCsv(RUN + "/whole_pattern/single_0deg/aggregate_matrix.csv"),
  sc10Whole: await readCsv(RUN + "/whole_pattern/single_10deg/aggregate_matrix.csv"),
  powderSpotsWhole: await readCsv(RUN + "/whole_pattern/powder_spots/aggregate_matrix.csv"),
  powderFitWhole: await readCsv(RUN + "/whole_pattern/powder_fit/aggregate_matrix.csv"),
  powderAdjusted: await readCsv(RUN + "/whole_pattern/powder_control_adjustment/aggregate_adjusted_residual_matrix.csv"),
};

const sc = summaryMetrics.per_peak.single_crystal;
const pw = summaryMetrics.per_peak.powder;
const win = summaryMetrics.windows;
const whole = summaryMetrics.whole_pattern;
const control = summaryMetrics.whole_pattern_control;
const boundary = summaryMetrics.window_boundaries;
const presence = summaryMetrics.presence;

const metricRows = [
  metric("sc0_raw_whole_r", "Single", "Whole pattern 0°", "r(Pearson score, |ΔP|)", whole.single_0deg.raw_r_correlation_vs_pressure_gap, "r", "QC", "Exact parity with 2026-07-16 baseline"),
  metric("sc10_raw_whole_r", "Single", "Whole pattern 10°", "r(Pearson score, |ΔP|)", whole.single_10deg.raw_r_correlation_vs_pressure_gap, "r", "QC", "Exact parity with baseline"),
  metric("powder_spots_raw_r", "Powder", "Whole spots", "r(Pearson score, |ΔP|)", whole.powder_spots.raw_r_correlation_vs_pressure_gap, "r", "QC", "Sample-sensitive channel"),
  metric("powder_fit_raw_r", "Powder", "Whole fit", "r(Pearson score, |ΔP|)", whole.powder_fit.raw_r_correlation_vs_pressure_gap, "r", "CONTROL", "W/background-dominated; stronger than spots"),
  metric("sc_obs", "Single", "Peak position", "Raw observations", sc.raw_observations, "rows", "METHOD DEV", "75 tracks; 263 collapsed track-frame features"),
  metric("sc_tracks", "Single", "Peak position", "Tracks", sc.tracks, "tracks", "METHOD DEV", "26 tracks are singletons in legacy audit"),
  metric("sc_width", "Single", "Peak position", "Median second-moment FWHM", sc.median_fwhm_two_theta_deg, "degree", "SECONDARY", sc.profile_width_status),
  metric("sc_loc_near", "Single", "Peak position", "Near-pair median", sc.location_near_median, "similarity", "SECONDARY", "Far set absent; AUC NA"),
  metric("pw_obs", "Powder", "Peak position", "Raw observations", pw.raw_observations, "rows", "METHOD DEV", "10 tracks; 164 collapsed features"),
  metric("pw_width", "Powder", "Peak position", "Median FWHM-like source width", pw.median_fwhm_two_theta_deg, "degree", "SECONDARY", pw.profile_width_status),
  metric("pw_loc_near", "Powder", "Peak position", "Near median", pw.location_near_median, "similarity", "SECONDARY", "Width and center precision are approximate"),
  metric("pw_loc_far", "Powder", "Peak position", "Far median", pw.location_far_median, "similarity", "SECONDARY", "Far ≥15 GPa"),
  metric("pw_loc_auc", "Powder", "Peak position", "Near-vs-far AUC", pw.location_near_far_auc, "probability", "SECONDARY", "New method is more physical; not higher performance than legacy"),
  metric("pw_area_tau", "Powder", "ROI area", "Track-balanced log-area pair tau", pw.area_repeatability.extra_pair_repeatability_tau, "log units", "SECONDARY", "29 groups; 5 independent tracks"),
  metric("pw_area_factor", "Powder", "ROI area", "Equivalent multiplicative repeat scale", Math.exp(pw.area_repeatability.extra_pair_repeatability_tau), "factor", "SECONDARY", "exp(tau); wide track-bootstrap CI"),
  metric("pw_area_auc", "Powder", "ROI area", "Near-vs-far AUC", pw.area_near_far_auc, "probability", "SECONDARY", "No per-observation area SE"),
  metric("sc_area_formal", "Single", "ROI area", "Formal calibrated metric available", 0, "boolean", "NA", "No same-pressure repeats; diagnostic matrix only"),
  metric("win_spots_auc", "Powder", "Same window spots", "Aligned NCC near-vs-far AUC", win.powder_spots.near_far_auc, "probability", "SECONDARY", "Fixed 5° core; same-pressure repeat weighting"),
  metric("win_fit_auc", "Powder", "Same window fit", "Aligned NCC near-vs-far AUC", win.powder_fit.near_far_auc, "probability", "CONTROL", "Control remains stronger than sample"),
  metric("partial_protocol", "Powder", "Whole control", "Median strict partial r with fit/order/protocol", control.median_scan_partial_r_given_fit_order_protocol, "r", "SENSITIVITY", "Descriptive, not causal"),
  metric("partial_no_protocol", "Powder", "Whole control", "Median strict partial r without protocol", control.median_scan_partial_r_without_protocol_control, "r", "SENSITIVITY", "Shows model sensitivity"),
  metric("partial_fisher", "Powder", "Whole control", "Median Fisher-z partial r with protocol", control.median_scan_partial_r_fisher_z_given_fit_z_order_protocol, "r", "SENSITIVITY", "Bounded-correlation sensitivity"),
  metric("partial_same_protocol", "Powder", "Whole control", "Median same-protocol-pairs partial r", control.median_scan_partial_r_same_protocol_pairs_given_fit_order, "r", "SENSITIVITY", "Protocol-restricted sensitivity"),
  metric("eligible_candidates", "Powder", "Boundary QC", "Eligible transition candidates", boundary.candidate_intervals, "intervals", "FORMAL", "Protocol/support/effect screened"),
  metric("stat_positive", "Powder", "Boundary QC", "Statistical-positive intervals", boundary.statistical_positive_intervals, "intervals", "DIAGNOSTIC", "Includes protocol artifact and sub-threshold effect"),
  metric("top_raw_edge", "Powder", "Boundary QC", "Strongest raw high-pressure edge", boundary.top_raw_high_pressure_GPa, "GPa", "QC FAIL", "3.50→3.75 switches D1w/duplicate to D1s/longscan"),
  metric("present_states", "Both", "Presence", "Present states", presence.present, "states", "OBSERVED", "Curated kept rows"),
  metric("unknown_states", "Both", "Presence", "Unknown states", presence.unknown, "states", "NA", "Not observed is not absent"),
  metric("confirmed_absent", "Both", "Presence", "Confirmed absent states", presence.confirmed_absent, "states", "NA", "Birth/death not identifiable"),
  metric("validation_checks", "Run", "Computational validation", "Checks passed", validationReport.passed, "checks", "PASS", "Computational invariants and baseline parity, not hypothesis validation"),
];

const workbook = Workbook.create();
const sheetNames = [
  "Summary", "Before vs After", "Core Metrics", "Peak Position", "Area Reliability",
  "Same Window", "Boundary QC", "Whole Control", "Presence", "Validation & Provenance",
  "Matrix Peak", "Matrix Whole", "Artifact Index",
];
const S = Object.fromEntries(sheetNames.map((name) => [name, workbook.worksheets.add(name)]));

const core = S["Core Metrics"];
setTitle(core, "UOTe uncertainty-aware core metrics", "每个数字都带 role/status。SECONDARY/SENSITIVITY 不应写成正式 transition 证据。", 8);
const coreTable = writeTable(core, 4, 1, [["Key", "System", "Family", "Metric", "Value", "Unit", "Role / status", "Interpretation"], ...metricRows], { wrapBody: true, bodyHeight: 31 });
applyFormats(core, coreTable); statusFormatting(core.getRange("G5:G" + coreTable.endRow));
core.freezePanes.freezeRows(4); core.freezePanes.freezeColumns(2);
setWidths(core, coreTable.endRow, { 0: 28, 1: 14, 2: 23, 3: 39, 4: 15, 5: 14, 6: 18, 7: 62 });
const metricMap = new Map(metricRows.map((row, i) => [row[0], 5 + i]));
const metricFormula = (key) => "='Core Metrics'!E" + metricMap.get(key);

const summary = S.Summary;
setTitle(summary, "UOTe correlation reanalysis — uncertainty-aware method development", "正式 baseline 仍是 2026-07-16；本工作簿是更保守的方法开发结果，不是独立 phase-transition proof。", 8);
setSection(summary, 4, "先看结论", 8);
const decision = writeTable(summary, 5, 1, [
  ["Question", "Answer", "Status", "Evidence / interpretation", "Value", "Unit", "Baseline relation", "Do not claim"],
  ["是否找到合格 transition candidate?", "没有", "FORMAL", "Protocol、scan support、effect-size 三重筛选后为 0", null, "intervals", "New method only", "没有信号 ≠ 证明没有 transition"],
  ["最强 raw window jump 是什么?", "3.50→3.75 GPa", "QC FAIL", "恰好 D1w/duplicate → D1s/longscan；按 acquisition artifact 处理", null, "GPa edge", "Not in legacy", "不可作为 UOTe candidate"],
  ["Control 是否仍比 sample 更强?", "是", "CONTROL", "Powder fit same-window AUC > spots AUC；raw whole fit |r| > spots |r|", null, "AUC", "Consistent with baseline caution", "不能单独归因 UOTe"],
  ["Whole-pattern adjustment 是否唯一?", "不是", "SENSITIVITY", "Protocol / Fisher-z / same-protocol 处理会改变数值；只看方向与敏感性", null, "partial r", "New sensitivity analysis", "不能挑一个数叫 corrected truth"],
  ["Birth/death 是否可识别?", "不可", "NA", "0 confirmed absent；missing 保持 unknown", null, "states", "Semantics preserved", "不能写成 0 births / 0 deaths"],
], { wrapBody: true, bodyHeight: 50 });
const summaryFormulas = [
  [6, "E", metricFormula("eligible_candidates"), "0"],
  [7, "E", metricFormula("top_raw_edge"), "0.00"],
  [8, "E", metricFormula("win_fit_auc"), "0.000"],
  [9, "E", metricFormula("partial_protocol"), "0.000"],
  [10, "E", metricFormula("confirmed_absent"), "0"],
];
for (const [row, col, formula, numberFormat] of summaryFormulas) {
  summary.getRange(col + row).formulas = [[formula]];
  summary.getRange(col + row).format.numberFormat = numberFormat;
}
statusFormatting(summary.getRange("C6:C10"));
setSection(summary, decision.endRow + 2, "关键数字（公式联动 Core Metrics）", 8);
const kpiStart = decision.endRow + 3;
const kpis = [
  ["Single 0° raw whole-pattern r", "sc0_raw_whole_r", "QC", "Exact baseline parity"],
  ["Single 10° raw whole-pattern r", "sc10_raw_whole_r", "QC", "Exact baseline parity"],
  ["Powder spots raw whole-pattern r", "powder_spots_raw_r", "QC", "Sample channel"],
  ["Powder fit raw whole-pattern r", "powder_fit_raw_r", "CONTROL", "W/background control"],
  ["Powder peak-position AUC", "pw_loc_auc", "SECONDARY", "FWHM-like width; not documented fit covariance"],
  ["Powder same-window spots AUC", "win_spots_auc", "SECONDARY", "Fixed-core direct NCC"],
  ["Powder same-window fit AUC", "win_fit_auc", "CONTROL", "Control stronger than spots"],
  ["Protocol-aware strict partial r", "partial_protocol", "SENSITIVITY", "Compare no-protocol/Fisher/same-protocol sheets"],
];
const kpiTable = writeTable(summary, kpiStart, 1, [["Metric", "Value", "Role", "How to read"], ...kpis.map((row) => [row[0], null, row[2], row[3]])], { wrapBody: true, bodyHeight: 35 });
kpis.forEach((row, index) => {
  const excelRow = kpiStart + 1 + index;
  summary.getRange("B" + excelRow).formulas = [[metricFormula(row[1])]];
  summary.getRange("B" + excelRow).format.numberFormat = "0.000";
});
statusFormatting(summary.getRange("C" + (kpiStart + 1) + ":C" + kpiTable.endRow));
const noteRow = kpiTable.endRow + 2;
summary.getRange("A" + noteRow + ":H" + noteRow).merge();
summary.getRange("A" + noteRow).values = [["32/32 PASS 是 computational validation：范围、NaN 语义、fixed-core shift、partial-r 定义、protocol screen、baseline parity 与 provenance；不是物理假设验证。"]];
summary.getRange("A" + noteRow).format = { fill: C.paleAmber, font: { bold: true, color: C.amber }, wrapText: true, rowHeight: 44, borders: { preset: "all", style: "thin", color: C.grid } };
summary.freezePanes.freezeRows(5);
setWidths(summary, noteRow, { 0: 35, 1: 20, 2: 17, 3: 60, 4: 15, 5: 15, 6: 28, 7: 40 });

const before = S["Before vs After"];
setTitle(before, "Before vs after — exactly what changed", "旧/新结果不一定能直接比较性能；本页强调公式、population、改进目标和剩余限制。", 6);
const beforeTable = writeTable(before, 4, 1, [
  ["Metric", "Before", "Now", "What it addresses", "Comparability", "Remaining limit"],
  ["Peak position", "clip(1−|Δ2θ|/0.06°,0,1)", "FWHM-scaled Gaussian center-distance; separate centroid z-consistency", "No universal hard cutoff", "Legacy/new AUC are descriptive, not performance contest", "Powder width is FWHM-like; single uses second moment, not fit covariance"],
  ["ROI area", "min/max of per-pixel mean excess", "Gaussian Δlog(integrated ROI excess/exposure), with repeatability when available", "Uses integrated signal and expected scatter", "Not directly comparable", "Single formal NA; powder only 29 groups / 5 tracks"],
  ["Same window", "ACF; neighboring-window maximum", "Direct NCC on fixed 5° core; zero/aligned/shift separate", "Avoids neighbor maximization", "Old .723 vs new value are different metrics", "Powder repeat weights; single uniform fallback"],
  ["Window-to-window", "Static ACF resemblance", "ΔP-adjusted change trajectories; independent-scan bootstrap", "Targets coordinated pressure changes", "New analysis", "Protocol/support/effect screens required"],
  ["Whole pattern", "Raw Pearson", "Raw Pearson QC + fit/order/protocol adjustment + Fisher/same-protocol sensitivities", "Makes system/protocol sensitivity visible", "Raw QC exactly comparable; adjusted is new", "Fit is not pure nuisance; result is noncausal"],
  ["Presence", "Mixed with missing", "Present / unknown / confirmed absent; birth/death separate", "Missing is never zero similarity", "New table", "No confirmed absence, so rates are NA"],
  ["Missing", "NaN", "NaN", "Preserves unknown", "Exact", "None"],
], { wrapBody: true, bodyHeight: 62 });
before.freezePanes.freezeRows(4);
setWidths(before, beforeTable.endRow, { 0: 23, 1: 39, 2: 49, 3: 39, 4: 39, 5: 49 });

const peak = S["Peak Position"];
setTitle(peak, "Peak position — width-scaled center distance and centroid consistency", "The primary location score uses FWHM as a distance scale; it is not a full profile-overlap integral. Powder width/SE is approximate and secondary.", 13);
setSection(peak, 4, "Method status", 13);
const peakOverview = writeTable(peak, 5, 1, [
  ["Dataset", "Raw obs", "Features", "Tracks", "Width scale", "Median width (°)", "Median center SE (°)", "Repeat groups", "Repeat tracks", "Repeat tau (°)", "Near median", "Far median", "AUC / status"],
  ["Single crystal", sc.raw_observations, sc.collapsed_features, sc.tracks, sc.profile_width_status, sc.median_fwhm_two_theta_deg, sc.median_centroid_se_two_theta_deg, sc.location_repeatability.same_pressure_groups, sc.location_repeatability.independent_tracks, sc.location_repeatability.extra_pair_repeatability_tau, sc.location_near_median, sc.location_far_median, "AUC NA; max ΔP <15 GPa"],
  ["Powder", pw.raw_observations, pw.collapsed_features, pw.tracks, pw.profile_width_status, pw.median_fwhm_two_theta_deg, pw.median_centroid_se_two_theta_deg, pw.location_repeatability.same_pressure_groups, pw.location_repeatability.independent_tracks, pw.location_repeatability.extra_pair_repeatability_tau, pw.location_near_median, pw.location_far_median, pw.location_near_far_auc],
], { wrapBody: true, bodyHeight: 49 });
applyFormats(peak, peakOverview);
setSection(peak, peakOverview.endRow + 2, "Track summaries", 13);
const trackRows = [["Dataset", "Track", "Frames", "Observations", "P min", "P max", "Median width", "Median center SE", "Location pairs", "Area pairs", "Missing semantics"]];
for (const [dataset, rows] of [["Single", singleTrack], ["Powder", powderTrack]]) {
  for (const row of pick(rows, ["track", "frames", "observations", "pressure_min_GPa", "pressure_max_GPa", "median_fwhm_two_theta_deg", "median_centroid_se_two_theta_deg", "location_pair_scores", "area_pair_scores", "missing_semantics"]).slice(1)) trackRows.push([dataset, ...row]);
}
const trackTable = writeTable(peak, peakOverview.endRow + 3, 1, trackRows, { wrapBody: true, bodyHeight: 25 });
applyFormats(peak, trackTable);
setSection(peak, trackTable.endRow + 2, "Collapsed track-frame features", 13);
const featureRows = [["Dataset", "Scan", "Track", "Pressure", "Frame", "Orientation", "Branch", "2theta", "Center SE", "Width", "Observations", "Location source", "Duplicate"]];
for (const rows of [singleFeatures, powderFeatures]) {
  for (const row of pick(rows, ["dataset", "scan", "track", "pressure_GPa", "frame", "orientation_base", "branch", "two_theta_deg", "centroid_se_two_theta_deg", "fwhm_two_theta_deg", "n_observations", "location_uncertainty_source", "duplicate_observation_flag"]).slice(1)) featureRows.push(row);
}
const featureTable = writeTable(peak, trackTable.endRow + 3, 1, featureRows, { wrapBody: true, bodyHeight: 24 });
applyFormats(peak, featureTable);
peak.freezePanes.freezeRows(5); peak.freezePanes.freezeColumns(3);
setWidths(peak, featureTable.endRow, { 0: 18, 1: 16, 2: 11, 3: 16, 4: 10, 5: 19, 6: 19, 7: 16, 8: 16, 9: 16, 10: 15, 11: 55, 12: 12 });

const area = S["Area Reliability"];
setTitle(area, "ROI area reliability", "Area means integrated background-subtracted ROI excess / exposure. Single is uncalibrated and formal summary is NA; powder is secondary.", 10);
setSection(area, 4, "Formal status", 10);
const areaOverview = writeTable(area, 5, 1, [
  ["Dataset", "Metric", "Per-observation SE", "Repeat groups", "Independent tracks", "Pair tau (log)", "Equivalent factor", "Near median", "Far median", "AUC / status"],
  ["Single crystal", "Integrated ROI excess counts/s; same orientation only", "Raw 2D sideband/counting propagation", sc.area_repeatability.same_pressure_groups, sc.area_repeatability.independent_tracks, sc.area_repeatability.extra_pair_repeatability_tau, null, sc.area_near_median, sc.area_far_median, "NA — diagnostic matrix only"],
  ["Powder", "Integrated connected-component area / filename exposure", "Unavailable; not fabricated", pw.area_repeatability.same_pressure_groups, pw.area_repeatability.independent_tracks, pw.area_repeatability.extra_pair_repeatability_tau, Math.exp(pw.area_repeatability.extra_pair_repeatability_tau), pw.area_near_median, pw.area_far_median, pw.area_near_far_auc],
], { wrapBody: true, bodyHeight: 52 });
applyFormats(area, areaOverview);
setSection(area, areaOverview.endRow + 2, "Powder repeatability calibration by track×pressure group", 10);
const areaGroups = pick(powderAreaGroups, ["group", "track", "pressure_GPa", "group_observations", "group_pairs", "group_absolute_difference_q68", "group_median_measurement_variance", "group_extra_pair_tau"]);
const areaGroupTable = writeTable(area, areaOverview.endRow + 3, 1, areaGroups, { wrapBody: true, bodyHeight: 25 });
applyFormats(area, areaGroupTable);
setSection(area, areaGroupTable.endRow + 2, "Powder location repeatability groups (for centroid consistency diagnostic)", 10);
const locGroups = pick(powderLocationGroups, ["group", "track", "pressure_GPa", "group_observations", "group_pairs", "group_absolute_difference_q68", "group_median_measurement_variance", "group_extra_pair_tau"]);
const locGroupTable = writeTable(area, areaGroupTable.endRow + 3, 1, locGroups, { wrapBody: true, bodyHeight: 25 });
applyFormats(area, locGroupTable);
area.freezePanes.freezeRows(5); area.freezePanes.freezeColumns(3);
setWidths(area, locGroupTable.endRow, { 0: 39, 1: 13, 2: 16, 3: 18, 4: 15, 5: 23, 6: 25, 7: 21, 8: 18, 9: 24 });

const same = S["Same Window"];
setTitle(same, "Same-window pressure comparison", "Direct NCC on a fixed 5° core. Powder: same-pressure cross-scan repeatability weighting. Single: uniform fallback. Shift is reported separately.", 20);
setSection(same, 4, "Series overview", 20);
const sameOverview = writeTable(same, 5, 1, [
  ["Series", "Channel", "Frames", "Scans", "Pressures", "Windows", "Weighting", "Width", "Step", "Max shift", "Near median", "Far median", "AUC", "r(score,ΔP)", "Role", "Fixed core", "Sensitivity bounds", "Old metric comparable?", "Limit", "Status"],
  ...windowLabels.map((label) => {
    const value = win[label];
    const role = label === "powder_fit" ? "CONTROL" : (label.startsWith("powder") ? "SECONDARY" : "SECONDARY / no far set");
    return [label, value.channel, value.frames, value.scans, value.pressures, value.windows, value.noise_weighting, value.window_width_deg, value.window_step_deg, value.max_shift_deg, value.near_median, value.far_median, value.near_far_auc, value.r_score_vs_pressure_gap, role, value.fixed_core_for_all_lags, "0.06 / 0.12 / 0.18°", "No — old used ACF/neighbor max", label.startsWith("single") ? "No ≥15 GPa far set" : "Control stronger than sample", "METHOD DEV" ];
  }),
], { wrapBody: true, bodyHeight: 52 });
applyFormats(same, sameOverview); statusFormatting(same.getRange("O6:O9"));
setSection(same, sameOverview.endRow + 2, "Shift-bound sensitivity", 20);
const sensitivityTable = writeTable(same, sameOverview.endRow + 3, 1, windowSensitivityRows, { wrapBody: true, bodyHeight: 24 });
applyFormats(same, sensitivityTable);
setSection(same, sensitivityTable.endRow + 2, "All fixed-core windows", 20);
const windowTable = writeTable(same, sensitivityTable.endRow + 3, 1, windowSummaryRows, { bodyHeight: 22 });
applyFormats(same, windowTable);
same.freezePanes.freezeRows(5); same.freezePanes.freezeColumns(3);
setWidths(same, windowTable.endRow, { 0: 17, 1: 16, 2: 11, 3: 12, 4: 12, 5: 14, 6: 17, 7: 17, 8: 17, 9: 17, 10: 17, 11: 18, 12: 18, 13: 18, 14: 18, 15: 18, 16: 20, 17: 19, 18: 19, 19: 22 });

const boundarySheet = S["Boundary QC"];
setTitle(boundarySheet, "Window trajectory boundary QC", "0 eligible candidates. Statistical-positive rows remain visible; protocol/support/practical-effect screens decide eligibility.", 17);
setSection(boundarySheet, 4, "Decision rules", 17);
const ruleTable = writeTable(boundarySheet, 5, 1, [
  ["Rule", "Value", "Why", "Result", "Status", "Primary windows", "Bootstrap", "Protocol source", "Effect threshold", "Min scans", "Eligible", "Stat-positive", "Confounded", "Top raw edge", "Top raw excess", "Interpretation", "Publication role"],
  ["Candidate eligibility", "CI low >0; sign≥0.60; robust outlier; protocol same; scan support", "Avoid tiny, sparse, or protocol-driven jumps", "No eligible intervals", "PASS", boundary.primary_windows, "Independent scan median", "Filename signature", boundary.practical_positive_outlier_threshold, boundary.minimum_independent_scan_support, boundary.candidate_intervals, boundary.statistical_positive_intervals, boundary.protocol_confounded_intervals, boundary.top_raw_high_pressure_GPa, boundary.top_raw_sample_specific_excess, boundary.status, "QC / hypothesis generation only"],
], { wrapBody: true, bodyHeight: 58 });
applyFormats(boundarySheet, ruleTable); statusFormatting(boundarySheet.getRange("E6:E6"));
setSection(boundarySheet, ruleTable.endRow + 2, "All adjacent-pressure intervals", 17);
const boundaryDisplay = pick(boundaryCandidates,
  ["candidate_rank", "p_low_GPa", "p_high_GPa", "delta_p_GPa", "independent_scans", "nonoverlap_windows", "median_spots_adjusted_change", "median_fit_adjusted_change", "median_sample_specific_excess", "sample_specific_excess_ci95_low", "sample_specific_excess_ci95_high", "statistical_positive_excess", "practical_positive_outlier", "protocol_low", "protocol_high", "protocol_confounded", "candidate_boundary", "interpretation"],
  ["Rank", "P low", "P high", "ΔP", "Scans", "Primary windows", "Spots adjusted", "Fit adjusted", "Sample excess", "CI low", "CI high", "Stat positive", "Practical outlier", "Protocol low", "Protocol high", "Protocol confounded", "Eligible", "Interpretation"],
);
const boundaryTable = writeTable(boundarySheet, ruleTable.endRow + 3, 1, boundaryDisplay, { wrapBody: true, bodyHeight: 36 });
applyFormats(boundarySheet, boundaryTable);
boundarySheet.getRange("A" + (boundaryTable.startRow + 1) + ":R" + (boundaryTable.startRow + 1)).format.fill = C.paleRed;
boundarySheet.freezePanes.freezeRows(boundaryTable.startRow); boundarySheet.freezePanes.freezeColumns(4);
setWidths(boundarySheet, boundaryTable.endRow, { 0: 9, 1: 12, 2: 12, 3: 11, 4: 11, 5: 17, 6: 18, 7: 18, 8: 18, 9: 15, 10: 15, 11: 15, 12: 17, 13: 19, 14: 19, 15: 19, 16: 12, 17: 55 });

const wholeSheet = S["Whole Control"];
setTitle(wholeSheet, "Whole-pattern QC and control sensitivity", "Raw Pearson exactly reproduces baseline. Adjusted results vary by protocol/Fisher handling; use direction and sensitivity, not a single corrected truth.", 25);
setSection(wholeSheet, 4, "Run-level control summary", 25);
const controlOverview = writeTable(wholeSheet, 5, 1, [
  ["Model", "Median partial r", "CI low", "CI high", "Protocol handled", "Correlation transform", "Population", "Interpretation", "Raw spots r", "Raw fit r", "Spots-fit r", "Protocol-changed pairs", "Scans", "Pairs", "Causal?", "Fit pure nuisance?", "Order control", "Protocol source", "Primary role", "Status", "Beta pressure", "Beta CI low", "Beta CI high", "Condition number", "Note"],
  ["Raw-r primary sensitivity", control.median_scan_partial_r_given_fit_order_protocol, control.median_scan_partial_r_given_fit_order_protocol_ci95[0], control.median_scan_partial_r_given_fit_order_protocol_ci95[1], "Indicator control", "Raw Pearson r", "All pairs", "Protocol-aware descriptive", control.raw_spots_pooled_r_vs_pressure_gap, control.raw_fit_pooled_r_vs_pressure_gap, control.spots_vs_fit_pair_r, control.protocol_changed_pairs, control.scans, control.pairs, "No", "No", "Frame-index gap", "Filename", "Sensitivity", "MODEL-SENSITIVE", control.median_standardized_pressure_gap_coefficient, control.median_standardized_pressure_gap_coefficient_ci95[0], control.median_standardized_pressure_gap_coefficient_ci95[1], control.median_design_condition_number, control.interpretation],
  ["No-protocol sensitivity", control.median_scan_partial_r_without_protocol_control, control.median_scan_partial_r_without_protocol_control_ci95[0], control.median_scan_partial_r_without_protocol_control_ci95[1], "No", "Raw Pearson r", "All pairs", "Shows protocol sensitivity", null, null, null, control.protocol_changed_pairs, control.scans, control.pairs, "No", "No", "Frame-index gap", "None", "Sensitivity", "MODEL-SENSITIVE", null, null, null, null, "Do not treat as corrected truth"],
  ["Fisher-z sensitivity", control.median_scan_partial_r_fisher_z_given_fit_z_order_protocol, control.median_scan_partial_r_fisher_z_given_fit_z_order_protocol_ci95[0], control.median_scan_partial_r_fisher_z_given_fit_z_order_protocol_ci95[1], "Indicator control", "atanh(r)", "All pairs", "Bounded-score sensitivity", null, null, null, control.protocol_changed_pairs, control.scans, control.pairs, "No", "No", "Frame-index gap", "Filename", "Sensitivity", "MODEL-SENSITIVE", control.median_fisher_z_standardized_pressure_gap_coefficient, control.median_fisher_z_standardized_pressure_gap_coefficient_ci95[0], control.median_fisher_z_standardized_pressure_gap_coefficient_ci95[1], null, "Direction remains negative; magnitude differs"],
  ["Same-protocol pairs", control.median_scan_partial_r_same_protocol_pairs_given_fit_order, control.median_scan_partial_r_same_protocol_pairs_given_fit_order_ci95[0], control.median_scan_partial_r_same_protocol_pairs_given_fit_order_ci95[1], "Pairs restricted", "Raw Pearson r", "Same-signature pairs", "Protocol-restricted sensitivity", null, null, null, 0, control.scans, null, "No", "No", "Frame-index gap", "Filename", "Sensitivity", "MODEL-SENSITIVE", null, null, null, null, "Still observational"],
], { wrapBody: true, bodyHeight: 58 });
applyFormats(wholeSheet, controlOverview); statusFormatting(wholeSheet.getRange("T6:T9"));
setSection(wholeSheet, controlOverview.endRow + 2, "Per-scan model outputs", 25);
const controlTable = writeTable(wholeSheet, controlOverview.endRow + 3, 1, controlScans, { bodyHeight: 23 });
applyFormats(wholeSheet, controlTable);
wholeSheet.freezePanes.freezeRows(controlTable.startRow); wholeSheet.freezePanes.freezeColumns(3);
setWidths(wholeSheet, controlTable.endRow, { 0: 14, 1: 10, 2: 19, 3: 19, 4: 17, 5: 18, 6: 17, 7: 23, 8: 22, 9: 23, 10: 25, 11: 22, 12: 20, 13: 19, 14: 19, 15: 23, 16: 22, 17: 19, 18: 19, 19: 23, 20: 20, 21: 18, 22: 18, 23: 18, 24: 20 });

const stateObjects = objects(presenceStates);
const presenceMap = new Map();
for (const row of stateObjects) {
  const key = row.dataset + "|" + row.track;
  if (!presenceMap.has(key)) presenceMap.set(key, { dataset: row.dataset, track: row.track, present: 0, unknown: 0, absent: 0, frames: 0 });
  const item = presenceMap.get(key); item.frames += 1;
  if (row.state === "present") item.present += 1;
  else if (row.state === "unknown") item.unknown += 1;
  else item.absent += 1;
}
const presenceAgg = [["Dataset", "Track", "Axis states", "Present", "Unknown", "Confirmed absent", "Birth rate", "Death rate"]];
for (const item of [...presenceMap.values()].sort((a, b) => String(a.dataset).localeCompare(String(b.dataset)) || Number(a.track) - Number(b.track))) {
  presenceAgg.push([item.dataset, item.track, item.frames, item.present, item.unknown, item.absent, null, null]);
}
const presenceSheet = S.Presence;
setTitle(presenceSheet, "Presence / birth / death semantics", "Curated kept row = present. Missing curated row = unknown. With no confirmed absence, birth/death rates are NA, not zero.", 8);
setSection(presenceSheet, 4, "Run summary", 8);
const presenceOverview = writeTable(presenceSheet, 5, 1, [
  ["States", "Present", "Unknown", "Confirmed absent", "Single transition rows", "Powder transition rows", "Birth/death identifiable", "Reason"],
  [presence.states, presence.present, presence.unknown, presence.confirmed_absent, presence.transition_rows_single_crystal, presence.transition_rows_powder, presence.birth_death_identifiable, presence.reason],
], { wrapBody: true, bodyHeight: 50 });
applyFormats(presenceSheet, presenceOverview);
setSection(presenceSheet, presenceOverview.endRow + 2, "Per-track state counts", 8);
const presenceTable = writeTable(presenceSheet, presenceOverview.endRow + 3, 1, presenceAgg, { bodyHeight: 23 });
applyFormats(presenceSheet, presenceTable);
presenceSheet.freezePanes.freezeRows(presenceTable.startRow); presenceSheet.freezePanes.freezeColumns(2);
setWidths(presenceSheet, presenceTable.endRow, { 0: 18, 1: 11, 2: 14, 3: 12, 4: 12, 5: 20, 6: 16, 7: 16 });

const validationSheet = S["Validation & Provenance"];
setTitle(validationSheet, "Computational validation and provenance", "32 checks verify implementation invariants, missing semantics, protocol screens, fixed-core shifts, strict partial-r definitions, and baseline parity—not a physical hypothesis.", 8);
setSection(validationSheet, 4, "Validation checks", 8);
const validationTable = writeTable(validationSheet, 5, 1, validationChecks, { wrapBody: true, bodyHeight: 35 });
applyFormats(validationSheet, validationTable);
const passedColumn = validationChecks[0].map(String).indexOf("passed");
if (passedColumn >= 0) {
  const range = validationSheet.getRangeByIndexes(validationTable.startRow, passedColumn, validationTable.endRow - validationTable.startRow, 1);
  range.conditionalFormats.add("cellIs", { operator: "equal", formula: 1, format: { fill: C.paleGreen, font: { bold: true, color: C.green } } });
  range.conditionalFormats.add("cellIs", { operator: "equal", formula: 0, format: { fill: C.paleRed, font: { bold: true, color: C.red } } });
}
setSection(validationSheet, validationTable.endRow + 2, "Input file-set audit", 8);
const inputTable = writeTable(validationSheet, validationTable.endRow + 3, 1, inputAudit, { wrapBody: true, bodyHeight: 29 });
applyFormats(validationSheet, inputTable);
setSection(validationSheet, inputTable.endRow + 2, "Runner and output identity", 8);
const identityTable = writeTable(validationSheet, inputTable.endRow + 3, 1, [
  ["Item", "Value", "Item", "Value", "Item", "Value", "Status", "Meaning"],
  ["Profile", runManifest.profile, "Date tag", runManifest.date_tag, "Baseline preserved", runManifest.baseline_preserved, "PASS", "New independent result directory"],
  ["Runner SHA256", runManifest.script_sha256, "Legacy runner SHA256", runManifest.legacy_preprocessing_runner_sha256, "Method config SHA256", runManifest.method_config_sha256, "PASS", "Code/result identity"],
  ["Input manifest SHA256", runManifest.input_file_manifest_sha256, "Analysis output manifest SHA256", runManifest.analysis_output_manifest_sha256, "Summary SHA256", runManifest.summary_metrics_sha256, "PASS", "File-level reproducibility"],
], { wrapBody: true, bodyHeight: 46 });
statusFormatting(validationSheet.getRange("G" + (identityTable.startRow + 1) + ":G" + identityTable.endRow));
validationSheet.freezePanes.freezeRows(5); validationSheet.freezePanes.freezeColumns(2);
setWidths(validationSheet, identityTable.endRow, { 0: 48, 1: 34, 2: 30, 3: 34, 4: 30, 5: 34, 6: 14, 7: 48 });

function matrixSection(sheet, row, title, note, rows, fixed01) {
  const data = strictLower(rows); const columns = data[0].length;
  setSection(sheet, row, title, columns);
  const noteRange = sheet.getRangeByIndexes(row, 0, 1, columns);
  noteRange.merge(); noteRange.values = [[note]];
  noteRange.format.fill = C.paleBlue; noteRange.format.font = { italic: true, color: C.text };
  noteRange.format.wrapText = true; noteRange.format.rowHeight = 36;
  const table = writeTable(sheet, row + 2, 1, data, { bodyHeight: 24 });
  if (data.length > 1 && data[0].length > 1) {
    const body = sheet.getRangeByIndexes(table.startRow, 1, data.length - 1, data[0].length - 1);
    body.format.numberFormat = "0.000"; colorScale(body, fixed01);
  }
  sheet.getRangeByIndexes(table.startRow, 0, data.length - 1, 1).format.fill = C.teal;
  return table.endRow + 2;
}

const matrixPeak = S["Matrix Peak"];
setTitle(matrixPeak, "Peak-level aggregate similarity matrices", "This sheet contains track-median aggregate matrices only. Individual location/area heatmaps and CSVs for every track are in per_peak_heatmaps/. Strict lower triangle display; single area is diagnostic/uncalibrated.", 20);
let peakRow = 4;
peakRow = matrixSection(matrixPeak, peakRow, "Single FWHM-scaled center-distance", "Second-moment width scale; no ≥15 GPa far set.", matrices.singleLocation, true);
peakRow = matrixSection(matrixPeak, peakRow, "Single integrated-area diagnostic", "Measurement-noise-only; same orientation pairs only; formal summary NA.", matrices.singleArea, true);
peakRow = matrixSection(matrixPeak, peakRow, "Powder FWHM-like center-distance", "Source q_width is not documented fitted FWHM; sensitivity/secondary.", matrices.powderLocation, true);
peakRow = matrixSection(matrixPeak, peakRow, "Powder track-balanced log-area similarity", "29 repeat groups from 5 tracks; no per-observation area SE.", matrices.powderArea, true);
matrixPeak.freezePanes.freezeRows(2); matrixPeak.freezePanes.freezeColumns(1);
setWidths(matrixPeak, peakRow, Object.fromEntries(Array.from({ length: 20 }, (_, i) => [i, i === 0 ? 28 : 11])));

const matrixWhole = S["Matrix Whole"];
setTitle(matrixWhole, "Whole-pattern QC and adjusted residual matrices", "Raw Pearson matrices exactly reproduce baseline. Adjusted powder residual is descriptive and protocol/model-sensitive.", 20);
let wholeRow = 4;
wholeRow = matrixSection(matrixWhole, wholeRow, "Single 0° raw Pearson QC", "Exact 2026-07-16 baseline parity.", matrices.sc0Whole, false);
wholeRow = matrixSection(matrixWhole, wholeRow, "Single 10° raw Pearson QC", "Exact 2026-07-16 baseline parity.", matrices.sc10Whole, false);
wholeRow = matrixSection(matrixWhole, wholeRow, "Powder spots raw Pearson QC", "Sample channel; median across 56 scans.", matrices.powderSpotsWhole, false);
wholeRow = matrixSection(matrixWhole, wholeRow, "Powder fit raw Pearson control", "W/background-dominated control; stronger pressure trend than spots.", matrices.powderFitWhole, false);
wholeRow = matrixSection(matrixWhole, wholeRow, "Powder spots residual after fit/order/protocol", "Descriptive raw-r residual matrix; compare Fisher/no-protocol/same-protocol sensitivity metrics.", matrices.powderAdjusted, false);
matrixWhole.freezePanes.freezeRows(2); matrixWhole.freezePanes.freezeColumns(1);
setWidths(matrixWhole, wholeRow, Object.fromEntries(Array.from({ length: 20 }, (_, i) => [i, i === 0 ? 28 : 11])));

const artifact = S["Artifact Index"];
setTitle(artifact, "Analysis artifact index", "All analysis outputs present before workbook generation. Full input files are separately recorded in input_file_manifest.csv.", 3);
const artifactTable = writeTable(artifact, 4, 1, artifactIndex, { wrapBody: true, bodyHeight: 28 });
applyFormats(artifact, artifactTable);
artifact.freezePanes.freezeRows(4); artifact.freezePanes.freezeColumns(1);
setWidths(artifact, artifactTable.endRow, { 0: 80, 1: 18, 2: 68 });

const previewRanges = {
  "Summary": "A1:H32", "Before vs After": "A1:F13", "Core Metrics": "A1:H35",
  "Peak Position": "A1:M36", "Area Reliability": "A1:J40", "Same Window": "A1:T38",
  "Boundary QC": "A1:R28", "Whole Control": "A1:Y35", "Presence": "A1:H35",
  "Validation & Provenance": "A1:H48", "Matrix Peak": "A1:T82", "Matrix Whole": "A1:T115",
  "Artifact Index": "A1:C35",
};

const rendered = [];
for (let index = 0; index < sheetNames.length; index += 1) {
  const sheetName = sheetNames[index];
  const image = await workbook.render({
    sheetName, range: previewRanges[sheetName],
    scale: sheetName.startsWith("Matrix") ? 0.82 : 1.0, format: "png",
  });
  const file = path.join(QA, String(index + 1).padStart(2, "0") + "_" + sheetName.toLowerCase().replace(/[^a-z0-9]+/g, "_") + ".png");
  await fs.writeFile(file, new Uint8Array(await image.arrayBuffer())); rendered.push(file);
}

const summaryInspection = await workbook.inspect({ kind: "table", range: "Summary!A1:H32", include: "values,formulas", tableMaxRows: 32, tableMaxCols: 8, maxChars: 24000 });
const preErrors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!", options: { useRegex: true, maxResults: 400 }, summary: "pre-export formula errors", maxChars: 16000 });
const preErrorCount = matchCount(preErrors.ndjson);
if (preErrorCount !== 0) throw new Error("Pre-export formula errors: " + preErrorCount);

const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(OUT);
const imported = await SpreadsheetFile.importXlsx(await FileBlob.load(OUT));
const importedSheets = await imported.inspect({ kind: "sheet", include: "id,name", maxChars: 16000 });
const importedSummary = await imported.inspect({ kind: "table", range: "Summary!A1:H32", include: "values,formulas", tableMaxRows: 32, tableMaxCols: 8, maxChars: 24000 });
const postErrors = await imported.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!", options: { useRegex: true, maxResults: 400 }, summary: "post-import formula errors", maxChars: 16000 });
const postErrorCount = matchCount(postErrors.ndjson);
if (postErrorCount !== 0) throw new Error("Post-import formula errors: " + postErrorCount);

for (const [name, range] of [["Summary", "A1:H32"], ["Boundary QC", "A1:R28"], ["Whole Control", "A1:Y35"], ["Validation & Provenance", "A1:H48"], ["Matrix Peak", "A1:T82"]]) {
  const image = await imported.render({ sheetName: name, range, scale: name.startsWith("Matrix") ? 0.82 : 1.0, format: "png" });
  await fs.writeFile(path.join(QA, "reimport_" + name.toLowerCase().replace(/[^a-z0-9]+/g, "_") + ".png"), new Uint8Array(await image.arrayBuffer()));
}

await fs.writeFile(QA + "/summary_inspection.ndjson", summaryInspection.ndjson ?? String(summaryInspection));
await fs.writeFile(QA + "/pre_export_formula_errors.ndjson", preErrors.ndjson ?? String(preErrors));
await fs.writeFile(QA + "/reimported_sheets.ndjson", importedSheets.ndjson ?? String(importedSheets));
await fs.writeFile(QA + "/reimported_summary.ndjson", importedSummary.ndjson ?? String(importedSummary));
await fs.writeFile(QA + "/reimported_formula_errors.ndjson", postErrors.ndjson ?? String(postErrors));
await fs.writeFile(OUT + ".inspect.ndjson", importedSummary.ndjson ?? String(importedSummary));
await fs.writeFile(QA + "/workbook_verification.json", JSON.stringify({
  workbook: OUT, sheets: sheetNames, sheet_count: sheetNames.length,
  rendered_sheet_count: rendered.length, reimport_rendered_key_sheets: 5,
  pre_export_formula_error_count: preErrorCount, post_import_formula_error_count: postErrorCount,
  computational_validation: validationReport,
  runner_sha256: runManifest.script_sha256,
  summary_metrics_sha256: runManifest.summary_metrics_sha256,
}, null, 2));

process.stdout.write(JSON.stringify({
  ok: true, workbook: OUT, sheets: sheetNames.length, rendered_sheets: rendered.length,
  pre_export_formula_errors: preErrorCount, post_import_formula_errors: postErrorCount, qa: QA,
}, null, 2) + "\n");
