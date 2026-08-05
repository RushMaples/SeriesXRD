#!/usr/bin/env node

/**
 * Build the uniform-correlation-v2 Excel report from the public result files.
 *
 * Workbook authoring intentionally uses @oai/artifact-tool only.  CSV/JSON
 * parsing is handled with the Node standard library so that the builder has no
 * hidden analytical dependency and remains easy to audit.
 */

import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const SHEET_NAMES = [
  "Summary",
  "Run & QC",
  "Per Peak",
  "Across Frames",
  "Within Frame",
  "Uncertainty",
  "Methods",
  "File Index",
];

const COLORS = {
  navy: "#17324D",
  teal: "#2D6F73",
  blue: "#4C78A8",
  paleBlue: "#DDEBF2",
  paleTeal: "#E2F0EE",
  paleGold: "#FFF3D6",
  paleGreen: "#E1F1E6",
  paleRed: "#F8E1E1",
  light: "#F6F8FA",
  line: "#CCD5DD",
  text: "#1F2933",
  muted: "#52616B",
  white: "#FFFFFF",
};

function parseArguments(argv) {
  const positional = [];
  let previewDir = null;
  for (let index = 0; index < argv.length; index += 1) {
    const item = argv[index];
    if (item === "--preview-dir") {
      if (index + 1 >= argv.length) {
        throw new Error("--preview-dir requires a directory path");
      }
      previewDir = argv[index + 1];
      index += 1;
    } else if (item.startsWith("--preview-dir=")) {
      previewDir = item.slice("--preview-dir=".length);
    } else if (item.startsWith("--")) {
      throw new Error(`Unknown option: ${item}`);
    } else {
      positional.push(item);
    }
  }
  if (positional.length !== 2) {
    throw new Error(
      "Usage: build_uniform_correlation_workbook.mjs RESULT_ROOT OUTPUT_XLSX [--preview-dir DIR]",
    );
  }
  const resultRoot = path.resolve(positional[0]);
  const outputXlsx = path.resolve(positional[1]);
  return {
    resultRoot,
    outputXlsx,
    previewDir: path.resolve(
      previewDir ?? path.join(path.dirname(outputXlsx), "previews"),
    ),
  };
}

function parseCsv(text) {
  const source = text.replace(/^\uFEFF/, "");
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let index = 0; index < source.length; index += 1) {
    const character = source[index];
    if (quoted) {
      if (character === '"' && source[index + 1] === '"') {
        field += '"';
        index += 1;
      } else if (character === '"') {
        quoted = false;
      } else {
        field += character;
      }
    } else if (character === '"') {
      quoted = true;
    } else if (character === ",") {
      row.push(field);
      field = "";
    } else if (character === "\n") {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += character;
    }
  }
  if (field.length || row.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  while (rows.length && rows.at(-1).every((value) => value === "")) {
    rows.pop();
  }
  if (!rows.length) return { headers: [], rows: [] };
  const rawHeaders = rows[0].map((value, index) => value.trim() || `column_${index + 1}`);
  const used = new Map();
  const headers = rawHeaders.map((header) => {
    const seen = used.get(header) ?? 0;
    used.set(header, seen + 1);
    return seen === 0 ? header : `${header}_${seen + 1}`;
  });
  const objects = rows.slice(1).map((values) => {
    const result = {};
    headers.forEach((header, index) => {
      result[header] = values[index] ?? "";
    });
    return result;
  });
  return { headers, rows: objects };
}

async function readJsonRequired(filePath) {
  const text = await fs.readFile(filePath, "utf8");
  return JSON.parse(text);
}

async function readCsvRequired(filePath) {
  return parseCsv(await fs.readFile(filePath, "utf8"));
}

async function listFilesRecursively(root, predicate) {
  const found = [];
  async function visit(directory) {
    let entries;
    try {
      entries = await fs.readdir(directory, { withFileTypes: true });
    } catch (error) {
      if (error?.code === "ENOENT") return;
      throw error;
    }
    entries.sort((first, second) => first.name.localeCompare(second.name));
    for (const entry of entries) {
      const target = path.join(directory, entry.name);
      if (entry.isDirectory()) await visit(target);
      else if (entry.isFile() && predicate(target)) found.push(target);
    }
  }
  await visit(root);
  return found;
}

function normalizeName(value) {
  return String(value ?? "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function coerceScalar(value, header = "") {
  if (value === null || value === undefined) return null;
  if (typeof value === "number" || typeof value === "boolean") {
    return Number.isFinite(value) || typeof value === "boolean" ? value : null;
  }
  if (typeof value !== "string") {
    return sanitizeText(JSON.stringify(value));
  }
  const text = value.trim();
  if (!text || /^(nan|na|n\/a|null|none)$/i.test(text)) return null;
  if (/^(true|yes)$/i.test(text)) return true;
  if (/^(false|no)$/i.test(text)) return false;
  const key = normalizeName(header);
  const numericHint = /(^|_)(count|index|frame|bytes|support|required|finite|nan|auc|similarity|median|low|high|width|area|pressure|gpa|deg|q|fwhm|score|fraction|percent|iterations|resamples|levels)(_|$)/.test(
    key,
  );
  const forceText =
    !numericHint &&
    (key === "id" ||
      /_id$/.test(key) ||
      /(^|_)(name|label|path|file|sha|hash|reason|note|method|status|state|profile|channel|family|metric|window|scan|hkl|shape|extension|analysis|validator|section|field|rule|formula)(_|$)/.test(
        key,
      ));
  if (!forceText && /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(text)) {
    const number = Number(text);
    if (Number.isFinite(number)) return number;
  }
  return sanitizeText(text);
}

function sanitizeText(value) {
  const text = String(value ?? "");
  const limited = text.length > 30000 ? `${text.slice(0, 29980)} …[truncated]` : text;
  return limited.startsWith("=") ? `'${limited}` : limited;
}

function asNumber(value) {
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (value === null || value === undefined) return null;
  const text = String(value).trim();
  if (!text || /^(nan|na|n\/a|null|none)$/i.test(text)) return null;
  const parsed = Number(text);
  return Number.isFinite(parsed) ? parsed : null;
}

function asBoolean(value) {
  if (typeof value === "boolean") return value;
  const text = String(value ?? "").trim().toLowerCase();
  if (["1", "true", "yes", "y", "pass", "passed", "reliable", "official"].includes(text)) {
    return true;
  }
  if (["0", "false", "no", "n", "fail", "failed", "unreliable"].includes(text)) {
    return false;
  }
  return null;
}

function median(values) {
  const finite = values.filter((value) => typeof value === "number" && Number.isFinite(value));
  if (!finite.length) return null;
  finite.sort((first, second) => first - second);
  const middle = Math.floor(finite.length / 2);
  return finite.length % 2 ? finite[middle] : (finite[middle - 1] + finite[middle]) / 2;
}

function flattenJson(value, prefix = "", output = []) {
  if (value === null || typeof value !== "object" || value instanceof Date) {
    output.push({ field: prefix || "value", value: coerceScalar(value, prefix) });
    return output;
  }
  if (Array.isArray(value)) {
    const scalarArray = value.every((item) => item === null || typeof item !== "object");
    if (scalarArray) {
      output.push({ field: prefix || "value", value: sanitizeText(JSON.stringify(value)) });
    } else {
      value.forEach((item, index) => flattenJson(item, `${prefix}[${index}]`, output));
    }
    return output;
  }
  for (const key of Object.keys(value).sort()) {
    flattenJson(value[key], prefix ? `${prefix}.${key}` : key, output);
  }
  return output;
}

function deepValues(value, requestedNames, output = []) {
  const aliases = new Set(requestedNames.map(normalizeName));
  if (Array.isArray(value)) {
    value.forEach((item) => deepValues(item, requestedNames, output));
  } else if (value && typeof value === "object") {
    for (const [key, item] of Object.entries(value)) {
      if (aliases.has(normalizeName(key))) output.push(item);
      deepValues(item, requestedNames, output);
    }
  }
  return output;
}

function firstDeepValue(value, names, fallback = null) {
  const values = deepValues(value, names);
  return values.length ? values[0] : fallback;
}

function valuesForMetric(rows, metric, fieldToken = "auc") {
  const metricName = normalizeName(metric);
  const values = [];
  for (const row of rows) {
    const rowMetric = normalizeName(
      row.metric ?? row.similarity ?? row.measure ?? row.type ?? row.score_type ?? "",
    );
    for (const [key, raw] of Object.entries(row)) {
      const normalized = normalizeName(key);
      const fieldMatches = normalized.includes(fieldToken);
      const metricMatches = normalized.includes(metricName) || rowMetric === metricName;
      if (fieldMatches && metricMatches) {
        const number = asNumber(raw);
        if (number !== null) values.push(number);
      }
    }
  }
  return values;
}

function numberFromRow(row, aliases) {
  const lookup = new Map(Object.keys(row).map((key) => [normalizeName(key), key]));
  for (const alias of aliases) {
    const exact = lookup.get(normalizeName(alias));
    if (exact) {
      const number = asNumber(row[exact]);
      if (number !== null) return number;
    }
  }
  return null;
}

function valueFromRow(row, aliases, fallback = null) {
  const lookup = new Map(Object.keys(row).map((key) => [normalizeName(key), key]));
  for (const alias of aliases) {
    const exact = lookup.get(normalizeName(alias));
    if (exact && row[exact] !== "") return row[exact];
  }
  return fallback;
}

function rowsForChannel(rows, channel) {
  return rows.filter((row) => normalizeName(row.channel) === normalizeName(channel));
}

function rowsForFamily(rows, channel, family) {
  return rowsForChannel(rows, channel).filter(
    (row) => normalizeName(row.family) === normalizeName(family),
  );
}

function aucValues(rows) {
  const result = [];
  for (const row of rows) {
    const value = numberFromRow(row, ["near_vs_far_auc", "near_far_auc", "auc"]);
    if (value !== null) result.push(value);
  }
  return result;
}

function distinctTrackCount(rows) {
  const identifiers = new Set();
  for (const row of rows) {
    const identifier = valueFromRow(row, ["track_id", "radial_peak_id", "peak_id", "track"]);
    if (identifier !== null && String(identifier).trim()) identifiers.add(String(identifier));
  }
  return identifiers.size || rows.length;
}

function officialTrackCount(rows) {
  const officialRows = rows.filter((row) => {
    const raw = valueFromRow(row, ["official", "is_official", "reliable", "accepted"], null);
    const parsed = asBoolean(raw);
    return parsed === null ? true : parsed;
  });
  return distinctTrackCount(officialRows);
}

function addProvenance(parsed, relativeSource, defaults = {}) {
  return parsed.rows.map((row) => ({ source_file: relativeSource, ...defaults, ...row }));
}

function validationSummary(report, relativeSource) {
  const checks = report?.checks && typeof report.checks === "object" ? report.checks : {};
  const checkValues = Object.values(checks).map(asBoolean).filter((value) => value !== null);
  const explicitPassed = asBoolean(
    report?.passed ?? report?.validation_passed ?? report?.all_checks_passed ?? null,
  );
  const passed = explicitPassed ?? (checkValues.length ? checkValues.every(Boolean) : null);
  const errors = Array.isArray(report?.errors) ? report.errors : [];
  return {
    source_file: relativeSource,
    validator: report?.validator ?? report?.profile ?? path.basename(relativeSource),
    passed,
    check_count: checkValues.length,
    failed_check_count: checkValues.filter((value) => !value).length,
    error_count: errors.length,
    error_summary: errors.length ? sanitizeText(errors.slice(0, 8).join(" | ")) : "",
  };
}

function columnLetter(index) {
  let value = index;
  let letters = "";
  while (value > 0) {
    value -= 1;
    letters = String.fromCharCode(65 + (value % 26)) + letters;
    value = Math.floor(value / 26);
  }
  return letters;
}

function styleHeader(range) {
  range.format = {
    fill: COLORS.teal,
    font: { bold: true, color: COLORS.white },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "outside", style: "thin", color: COLORS.navy },
  };
}

function setupSheet(sheet, title, subtitle, widthColumns = 8) {
  sheet.showGridLines = false;
  const end = columnLetter(widthColumns);
  sheet.getRange(`A1:${end}1`).merge();
  sheet.getRange(`A2:${end}2`).merge();
  sheet.getRange("A1").values = [[title]];
  sheet.getRange("A2").values = [[subtitle]];
  sheet.getRange(`A1:${end}1`).format = {
    fill: COLORS.navy,
    font: { bold: true, color: COLORS.white, size: 17 },
    verticalAlignment: "center",
  };
  sheet.getRange(`A2:${end}2`).format = {
    fill: COLORS.paleBlue,
    font: { color: COLORS.muted, italic: true, size: 10 },
    verticalAlignment: "center",
    wrapText: true,
  };
  sheet.getRange("A1").format.rowHeight = 31;
  sheet.getRange("A2").format.rowHeight = 25;
}

function writeKpis(sheet, startRow, kpis, { explanationColumn = true } = {}) {
  const columns = explanationColumn ? 3 : 2;
  const end = columnLetter(columns);
  sheet.getRange(`A${startRow}:${end}${startRow}`).values = [
    explanationColumn ? ["Headline indicator", "Value", "How to read it"] : ["Headline indicator", "Value"],
  ];
  styleHeader(sheet.getRange(`A${startRow}:${end}${startRow}`));
  kpis.forEach((item, offset) => {
    const row = startRow + 1 + offset;
    sheet.getRange(`A${row}`).values = [[item.label]];
    if (item.formula) sheet.getRange(`B${row}`).formulas = [[item.formula]];
    else sheet.getRange(`B${row}`).values = [[coerceScalar(item.value, item.label)]];
    if (item.numberFormat) sheet.getRange(`B${row}`).setNumberFormat(item.numberFormat);
    if (explanationColumn) sheet.getRange(`C${row}`).values = [[sanitizeText(item.explanation ?? "")]];
    const fill = offset % 2 === 0 ? COLORS.white : COLORS.light;
    sheet.getRange(`A${row}:${end}${row}`).format = {
      fill,
      font: { color: COLORS.text },
      verticalAlignment: "center",
      wrapText: explanationColumn,
      borders: {
        bottom: { style: "thin", color: COLORS.line },
      },
    };
    sheet.getRange(`A${row}:${end}${row}`).format.rowHeight = 48;
  });
  const lastRow = startRow + kpis.length;
  sheet.getRange(`A${startRow + 1}:A${lastRow}`).format.font = { bold: true, color: COLORS.text };
  sheet.getRange(`B${startRow + 1}:B${lastRow}`).format.horizontalAlignment = "right";
  sheet.getRange(`A${startRow}:A${lastRow}`).format.columnWidth = 31;
  sheet.getRange(`B${startRow}:B${lastRow}`).format.columnWidth = 18;
  if (explanationColumn) sheet.getRange(`C${startRow}:C${lastRow}`).format.columnWidth = 62;
  return lastRow;
}

function widenKpiExplanation(sheet, startRow, kpiCount, widthColumns) {
  const end = columnLetter(widthColumns);
  const header = sheet.getRange(`C${startRow}:${end}${startRow}`);
  header.merge();
  styleHeader(header);
  for (let offset = 0; offset < kpiCount; offset += 1) {
    const row = startRow + 1 + offset;
    const range = sheet.getRange(`C${row}:${end}${row}`);
    range.merge();
    range.format = {
      fill: offset % 2 === 0 ? COLORS.white : COLORS.light,
      font: { color: COLORS.text },
      verticalAlignment: "center",
      wrapText: true,
      borders: { bottom: { style: "thin", color: COLORS.line } },
    };
  }
}

function chooseColumns(rows, preferred = []) {
  const discovered = [];
  const seen = new Set();
  for (const row of rows) {
    for (const key of Object.keys(row)) {
      if (!seen.has(key)) {
        seen.add(key);
        discovered.push(key);
      }
    }
  }
  const ordered = [];
  for (const item of preferred) {
    if (seen.has(item) && !ordered.includes(item)) ordered.push(item);
  }
  for (const item of discovered) if (!ordered.includes(item)) ordered.push(item);
  return ordered.length ? ordered : ["status"];
}

function numberFormatForHeader(header) {
  const key = normalizeName(header);
  if (/(count|support|index|frame|files|rows|bytes|iterations|resamples|levels|windows)$/.test(key)) {
    return "#,##0";
  }
  if (/(fraction|percent|percentage|rate)$/.test(key)) return "0.0%";
  if (/(auc|similarity|median|ci_low|ci_high|ci_width|score|jaccard|correlation)/.test(key)) {
    return "0.000";
  }
  if (/(pressure|gpa|gap)/.test(key)) return "0.00";
  if (/(deg|theta|fwhm|prominence|area|q_)/.test(key)) return "0.0000";
  return null;
}

function widthForColumn(header, rows) {
  const key = normalizeName(header);
  if (/(path|file|artifact)/.test(key)) return 52;
  if (/(reason|note|description|summary|error|shape|method|rule|formula)/.test(key)) return 48;
  if (/(sha|hash)/.test(key)) return 23;
  const samples = rows.slice(0, 200).map((row) => String(row[header] ?? "").length);
  const maximum = Math.max(header.length, ...samples, 8);
  return Math.max(10, Math.min(30, maximum + 2));
}

function writeObjectTable(sheet, startRow, rows, tableName, preferred = []) {
  const displayRows = rows.length ? rows : [{ status: "No data available" }];
  const columns = chooseColumns(displayRows, preferred);
  const data = [
    columns,
    ...displayRows.map((row) => columns.map((header) => coerceScalar(row[header], header))),
  ];
  const endRow = startRow + data.length - 1;
  const endColumn = columnLetter(columns.length);
  const tableRange = sheet.getRange(`A${startRow}:${endColumn}${endRow}`);
  tableRange.values = data;
  styleHeader(sheet.getRange(`A${startRow}:${endColumn}${startRow}`));
  if (displayRows.length) {
    const table = sheet.tables.add(`A${startRow}:${endColumn}${endRow}`, true, tableName);
    table.style = "TableStyleMedium2";
    table.showHeaders = true;
    table.showFilterButton = true;
  }
  columns.forEach((header, index) => {
    const letter = columnLetter(index + 1);
    const range = sheet.getRange(`${letter}${startRow}:${letter}${endRow}`);
    range.format.columnWidth = widthForColumn(header, displayRows);
    const numberFormat = numberFormatForHeader(header);
    if (numberFormat && endRow > startRow) {
      sheet.getRange(`${letter}${startRow + 1}:${letter}${endRow}`).setNumberFormat(numberFormat);
    }
    if (/(reason|note|description|summary|error|shape|method|rule)/.test(normalizeName(header))) {
      range.format.wrapText = true;
      if (endRow > startRow) {
        sheet.getRange(`A${startRow + 1}:${endColumn}${endRow}`).format.rowHeight = 34;
      }
    }
  });
  return { startRow, endRow, endColumn, columns };
}

function appendChannelRows(parsed, channel, relativeSource) {
  return addProvenance(parsed, relativeSource, { channel });
}

function buildCiRecords(acrossRows, withinRows) {
  const records = [];
  for (const row of acrossRows) {
    const estimate = numberFromRow(row, ["near_vs_far_auc", "near_far_auc", "auc"]);
    const low = numberFromRow(row, ["auc_ci_low", "near_far_auc_ci_low", "ci_low"]);
    const high = numberFromRow(row, ["auc_ci_high", "near_far_auc_ci_high", "ci_high"]);
    records.push({
      source_file: row.source_file,
      channel: row.channel,
      analysis: "across_frames",
      family: row.family ?? "",
      item: valueFromRow(row, ["window_index", "window", "label"], ""),
      metric: "near_vs_far_auc",
      estimate,
      ci_low: low,
      ci_high: high,
      ci_width: low !== null && high !== null ? high - low : null,
      note: valueFromRow(row, ["auc_reason_if_na", "reason_if_na", "reason"], ""),
    });
  }
  for (const row of withinRows) {
    const estimate = numberFromRow(row, ["median_similarity", "similarity", "median"]);
    const low = numberFromRow(row, ["ci_low", "similarity_ci_low"]);
    const high = numberFromRow(row, ["ci_high", "similarity_ci_high"]);
    records.push({
      source_file: row.source_file,
      channel: row.channel,
      analysis: "within_frame",
      family: asBoolean(row.is_nonoverlap_control_pair) ? "nonoverlap_control" : "all_windows",
      item: `${row.window_a ?? ""} vs ${row.window_b ?? ""}`,
      metric: "median_similarity",
      estimate,
      ci_low: low,
      ci_high: high,
      ci_width: low !== null && high !== null ? high - low : null,
      note: asBoolean(row.sufficient_support) === false ? "insufficient support" : "",
    });
  }
  return records;
}

function scientificEvidenceText(validationPassed, strictAuc) {
  if (validationPassed !== true) {
    return "算法或结构验证没有明确全部通过；现在只能把这些数值当作诊断结果，不能做科学结论。";
  }
  if (strictAuc === null) {
    return "严格 same-window ACF 没有足够支持数据，当前证据不足，不能判断近压力是否更相似。";
  }
  if (strictAuc >= 0.7) {
    return "严格 ACF 对近/远压力有较强区分，但仍需同时看 95% CI、direct control 和 fit control；它本身不能单独证明相变。";
  }
  if (strictAuc >= 0.6) {
    return "严格 ACF 有一定区分信号，但强度中等；应把它看成线索，而不是相变证明。";
  }
  return "严格 ACF 对近/远压力的区分较弱；结果可以是正确的，但科学证据目前不充分。";
}

function compactInspect(ndjson, maximumRecords = 12) {
  const records = [];
  for (const line of String(ndjson ?? "").split(/\r?\n/).filter(Boolean)) {
    if (records.length >= maximumRecords) break;
    try {
      records.push(JSON.parse(line));
    } catch {
      records.push({ raw: line.slice(0, 500) });
    }
  }
  return records;
}

async function main() {
  const { resultRoot, outputXlsx, previewDir } = parseArguments(process.argv.slice(2));
  const relative = (target) => path.relative(resultRoot, target).split(path.sep).join("/");

  const manifestPath = path.join(resultRoot, "run_manifest.json");
  const configPath = path.join(resultRoot, "algorithm_config.json");
  const artifactIndexPath = path.join(resultRoot, "artifact_index.csv");
  const manifest = await readJsonRequired(manifestPath);
  const algorithmConfig = await readJsonRequired(configPath);
  const artifactIndexParsed = await readCsvRequired(artifactIndexPath);

  const peakRows = [];
  const acrossRows = [];
  const withinRows = [];
  for (const channel of ["spots", "fit"]) {
    const peakPath = path.join(resultRoot, channel, "per_peak", "peak_summary.csv");
    const acrossPath = path.join(resultRoot, channel, "across_frames", "all_window_summaries.csv");
    const withinPath = path.join(resultRoot, channel, "within_frame", "window_pair_summary.csv");
    peakRows.push(...appendChannelRows(await readCsvRequired(peakPath), channel, relative(peakPath)));
    acrossRows.push(...appendChannelRows(await readCsvRequired(acrossPath), channel, relative(acrossPath)));
    withinRows.push(...appendChannelRows(await readCsvRequired(withinPath), channel, relative(withinPath)));
  }

  const validationFiles = [
    ...(await listFilesRecursively(path.join(resultRoot, "validation"), (file) => file.endsWith(".json"))),
  ];
  const legacyValidation = path.join(resultRoot, "validation_report.json");
  try {
    await fs.access(legacyValidation);
    validationFiles.push(legacyValidation);
  } catch {
    // The official layout keeps reports in validation/; the legacy root file is optional.
  }
  if (!validationFiles.length) {
    throw new Error(`No validation report JSON files found under ${resultRoot}`);
  }
  const validationReports = [];
  for (const file of [...new Set(validationFiles)].sort()) {
    const report = await readJsonRequired(file);
    validationReports.push(validationSummary(report, relative(file)));
  }

  const robustnessFiles = await listFilesRecursively(
    path.join(resultRoot, "robustness"),
    (file) => file.toLowerCase().endsWith(".csv"),
  );
  if (!robustnessFiles.length) {
    throw new Error(`No robustness CSV tables found under ${path.join(resultRoot, "robustness")}`);
  }
  const robustnessRows = [];
  for (const file of robustnessFiles) {
    robustnessRows.push(...addProvenance(await readCsvRequired(file), relative(file)));
  }

  const artifactRows = addProvenance(artifactIndexParsed, relative(artifactIndexPath));
  const explicitValidation = validationReports
    .map((row) => asBoolean(row.passed))
    .filter((value) => value !== null);
  const validationPassed = explicitValidation.length
    ? explicitValidation.every(Boolean)
    : asBoolean(firstDeepValue(manifest, ["validation_passed", "all_checks_passed"], null));

  const spotsPeakRows = rowsForChannel(peakRows, "spots");
  const fitPeakRows = rowsForChannel(peakRows, "fit");
  const spotsTrackCount = officialTrackCount(spotsPeakRows);
  const fitTrackCount = officialTrackCount(fitPeakRows);
  const perPeakAucValues = (rows, metric) =>
    rows
      .map((row) => numberFromRow(row, [`${metric}_auc`]))
      .filter((value) => value !== null);
  const spotsAreaAuc = median(perPeakAucValues(spotsPeakRows, "area"));
  const spotsLocationAuc = median(perPeakAucValues(spotsPeakRows, "location"));
  const spotsPresenceAuc = median(perPeakAucValues(spotsPeakRows, "presence"));

  const spotsStrictRows = rowsForFamily(acrossRows, "spots", "acf_strict");
  const spotsDirectRows = rowsForFamily(acrossRows, "spots", "direct_strict");
  const spotsShiftRows = rowsForFamily(acrossRows, "spots", "shift_tolerant_secondary");
  const fitStrictRows = rowsForFamily(acrossRows, "fit", "acf_strict");
  const spotsStrictAuc = median(aucValues(spotsStrictRows));
  const spotsDirectAuc = median(aucValues(spotsDirectRows));
  const spotsShiftAuc = median(aucValues(spotsShiftRows));
  const fitStrictAuc = median(aucValues(fitStrictRows));
  const strictSupportedWindows = spotsStrictRows.filter(
    (row) => numberFromRow(row, ["near_vs_far_auc", "near_far_auc", "auc"]) !== null,
  ).length;

  const spotsWithin = rowsForChannel(withinRows, "spots");
  const fitWithin = rowsForChannel(withinRows, "fit");
  const nonoverlap = (rows) => rows.filter((row) => asBoolean(row.is_nonoverlap_control_pair) === true);
  const similarityValues = (rows) =>
    rows
      .map((row) => numberFromRow(row, ["median_similarity", "similarity", "median"]))
      .filter((value) => value !== null);
  const spotsNonoverlapMedian = median(similarityValues(nonoverlap(spotsWithin)));
  const spotsAllWindowMedian = median(similarityValues(spotsWithin));
  const fitNonoverlapMedian = median(similarityValues(nonoverlap(fitWithin)));
  const spotsSupportFlags = spotsWithin
    .map((row) => asBoolean(row.sufficient_support))
    .filter((value) => value !== null);
  const spotsSupportFraction = spotsSupportFlags.length
    ? spotsSupportFlags.filter(Boolean).length / spotsSupportFlags.length
    : null;

  const ciRecords = buildCiRecords(acrossRows, withinRows);
  const strictCiWidth = median(
    ciRecords
      .filter(
        (row) =>
          row.channel === "spots" && row.analysis === "across_frames" && row.family === "acf_strict",
      )
      .map((row) => row.ci_width),
  );
  const allCiWidth = median(ciRecords.map((row) => row.ci_width));

  const workbook = Workbook.create();
  const sheets = Object.fromEntries(
    SHEET_NAMES.map((name) => [name, workbook.worksheets.add(name)]),
  );

  // Run & QC: fixed headline cell B5 is referenced from Summary.
  setupSheet(
    sheets["Run & QC"],
    "Run & QC / 运行与质量控制",
    "记录算法版本、输入、验证结果与运行清单；数值好坏不影响 QC 是否通过。",
    8,
  );
  const runStatus = firstDeepValue(manifest, ["run_status", "status", "profile_status"], "unknown");
  const profile = firstDeepValue(manifest, ["profile", "profile_id", "algorithm_profile"], "uniform-correlation-v2");
  const algorithmVersion = firstDeepValue(
    manifest,
    ["algorithm_version", "version"],
    firstDeepValue(algorithmConfig, ["algorithm_version"], ""),
  );
  const wavelength = asNumber(
    firstDeepValue(
      manifest,
      ["wavelength", "wavelength_a", "wavelength_angstrom"],
      firstDeepValue(algorithmConfig, ["wavelength", "wavelength_a", "wavelength_angstrom"], null),
    ),
  );
  const bootstrapResamples = asNumber(
    firstDeepValue(
      manifest,
      ["bootstrap_resamples", "bootstrap_iterations", "n_bootstrap"],
      firstDeepValue(
        algorithmConfig,
        ["bootstrap_resamples", "bootstrap_iterations", "n_bootstrap"],
        null,
      ),
    ),
  );
  const randomSeed = asNumber(
    firstDeepValue(
      manifest,
      ["random_seed", "seed", "bootstrap_seed"],
      firstDeepValue(algorithmConfig, ["random_seed", "seed", "bootstrap_seed"], null),
    ),
  );
  const inputCount = asNumber(
    firstDeepValue(manifest, ["input_file_count", "file_count", "n_files", "frames"], null),
  );
  writeKpis(sheets["Run & QC"], 3, [
    {
      label: "Run status",
      value: runStatus,
      explanation: "运行清单写下的状态；它不等同于科学证据强弱。",
    },
    {
      label: "Validation reports all passed",
      value: validationPassed,
      explanation: "TRUE 表示现有 validation reports 的检查全部通过；FALSE 表示报告只能作为诊断。",
    },
    {
      label: "Validation checks",
      value: validationReports.reduce((sum, row) => sum + (asNumber(row.check_count) ?? 0), 0),
      numberFormat: "#,##0",
      explanation: "独立检查项目的总数。",
    },
    {
      label: "Failed checks",
      value: validationReports.reduce((sum, row) => sum + (asNumber(row.failed_check_count) ?? 0), 0),
      numberFormat: "#,##0",
      explanation: "应为 0；若不为 0，请先看下方 validation 表。",
    },
    { label: "Profile", value: profile, explanation: "冻结的统一算法配置名称。" },
    { label: "Algorithm version", value: algorithmVersion, explanation: "用于复现本次分析的软件算法版本。" },
    { label: "Wavelength (Å)", value: wavelength, numberFormat: "0.0000", explanation: "q 与 2θ 转换使用的显式波长。" },
    { label: "Bootstrap resamples", value: bootstrapResamples, numberFormat: "#,##0", explanation: "scan-level bootstrap 重采样次数。" },
    { label: "Random seed", value: randomSeed, numberFormat: "0", explanation: "固定为 0，使重复运行可复现。" },
    { label: "Input files / frames", value: inputCount, numberFormat: "#,##0", explanation: "清单记录的输入数量；完整逐文件哈希在 input inventory。" },
  ]);
  widenKpiExplanation(sheets["Run & QC"], 3, 10, 8);
  const validationTable = writeObjectTable(
    sheets["Run & QC"],
    16,
    validationReports,
    "ValidationReportsTable",
    ["source_file", "validator", "passed", "check_count", "failed_check_count", "error_count", "error_summary"],
  );
  const manifestRows = flattenJson(manifest).map((row) => ({ section: "run_manifest", ...row }));
  writeObjectTable(
    sheets["Run & QC"],
    validationTable.endRow + 3,
    manifestRows,
    "RunManifestTable",
    ["section", "field", "value"],
  );
  // Two audit tables share the same worksheet columns.  Preserve enough
  // width for both the validation source/validator names and manifest fields.
  const qcColumnWidths = { A: 44, B: 42, C: 32, D: 18, E: 20, F: 16, G: 42 };
  for (const [letter, width] of Object.entries(qcColumnWidths)) {
    sheets["Run & QC"].getRange(`${letter}:${letter}`).format.columnWidth = width;
  }
  sheets["Run & QC"].freezePanes.freezeRows(3);

  // Per Peak: fixed headline cell B5 is the spots official track count.
  setupSheet(
    sheets["Per Peak"],
    "Per Peak / 自动径向峰结果",
    "1D XY 只能得到 radial_peak；area/location 只在同 scan、两个压力都可靠检出时计算，缺失保持 NaN。",
    10,
  );
  writeKpis(sheets["Per Peak"], 3, [
    { label: "Summary rows", value: peakRows.length, numberFormat: "#,##0", explanation: "spots 与 fit 两个 channel 的 peak summary 总行数。" },
    { label: "Spots official radial tracks", value: spotsTrackCount, numberFormat: "#,##0", explanation: "主结果中满足 track support 且未被标记 ambiguous 的径向峰数；不是原来的 azimuth-specific 40 tracks。" },
    { label: "Fit official radial tracks", value: fitTrackCount, numberFormat: "#,##0", explanation: "fit control 独立自动追踪到的可靠径向峰数。" },
    { label: "Spots median area AUC", value: spotsAreaAuc, numberFormat: "0.000", explanation: "AUC≈0.5 表示近/远压力不可区分；越大表示近压力的相对峰面积通常更相似。" },
    { label: "Spots median location AUC", value: spotsLocationAuc, numberFormat: "0.000", explanation: "使用峰宽归一化后的 q-location similarity；只比较双方都 present 的 scan。" },
    { label: "Spots median presence AUC", value: spotsPresenceAuc, numberFormat: "0.000", explanation: "Presence 用 Jaccard 单独评估，双方都 absent 不会提高分数。" },
  ]);
  widenKpiExplanation(sheets["Per Peak"], 3, 6, 10);
  writeObjectTable(
    sheets["Per Peak"],
    12,
    peakRows,
    "PerPeakSummaryTable",
    ["source_file", "channel", "track_id", "official", "ambiguous", "metric", "near_median", "far_median", "near_vs_far_auc"],
  );
  sheets["Per Peak"].freezePanes.freezeRows(12);
  sheets["Per Peak"].freezePanes.freezeColumns(2);

  // Across Frames: fixed headline cell B5 is spots strict ACF median AUC.
  setupSheet(
    sheets["Across Frames"],
    "Across Frames / 同窗口跨压力",
    "主结果是 strict same-window ACF；direct strict 保留位置，shift-tolerant 只作为 secondary。",
    10,
  );
  writeKpis(sheets["Across Frames"], 3, [
    { label: "Window-summary rows", value: acrossRows.length, numberFormat: "#,##0", explanation: "两个 channel、三种 family 的 window summary 行数。" },
    { label: "Spots strict ACF median AUC", value: spotsStrictAuc, numberFormat: "0.000", explanation: "主结果。0.5 约等于随机区分；越接近 1，近压力 window pattern 越可能比远压力更相似。" },
    { label: "Spots direct strict median AUC", value: spotsDirectAuc, numberFormat: "0.000", explanation: "位置敏感 control。若它与 ACF 结论不同，说明绝对峰位置在影响结果。" },
    { label: "Spots shift-tolerant median AUC", value: spotsShiftAuc, numberFormat: "0.000", explanation: "允许左右一个 step 后取最大值，容易提高分数，因此只能作为 secondary。" },
    { label: "Fit strict ACF median AUC", value: fitStrictAuc, numberFormat: "0.000", explanation: "fit control；与 spots 比较可判断信号是否只来自某个处理 channel。" },
    { label: "Spots strict supported windows", value: strictSupportedWindows, numberFormat: "#,##0", explanation: "有有限 AUC、达到支持要求的 strict ACF windows。" },
  ]);
  widenKpiExplanation(sheets["Across Frames"], 3, 6, 10);
  writeObjectTable(
    sheets["Across Frames"],
    12,
    acrossRows,
    "AcrossWindowSummaryTable",
    ["source_file", "channel", "family", "window_index", "start_deg", "end_deg", "median_similarity", "near_median", "far_median", "near_vs_far_auc", "auc_ci_low", "auc_ci_high", "auc_reason_if_na"],
  );
  sheets["Across Frames"].freezePanes.freezeRows(12);
  sheets["Across Frames"].freezePanes.freezeColumns(3);

  // Within Frame: fixed headline cell B5 is spots non-overlap median similarity.
  setupSheet(
    sheets["Within Frame"],
    "Within Frame / 同一帧窗口之间",
    "科学解释以真正不重叠的 control 为主；重叠 windows 因共享数据会自然显得更相似。",
    10,
  );
  writeKpis(sheets["Within Frame"], 3, [
    { label: "Window-pair rows", value: withinRows.length, numberFormat: "#,##0", explanation: "spots 与 fit 的 window pair summary 总行数。" },
    { label: "Spots non-overlap median", value: spotsNonoverlapMedian, numberFormat: "0.000", explanation: "主解释值：没有共享 2θ 数据的 windows，其 ACF pattern 的中位 Pearson similarity。" },
    { label: "Spots all-window median", value: spotsAllWindowMedian, numberFormat: "0.000", explanation: "探索值；含重叠 windows，通常会被共享数据抬高。" },
    { label: "Fit non-overlap median", value: fitNonoverlapMedian, numberFormat: "0.000", explanation: "fit control 的不重叠 window similarity。" },
    { label: "Spots sufficient-support pairs", value: spotsSupportFraction, numberFormat: "0.0%", explanation: "达到 scan support 要求的 spots window pairs 比例。" },
  ]);
  widenKpiExplanation(sheets["Within Frame"], 3, 5, 10);
  // The beginner-facing explanations wrap to several Chinese lines.  Give
  // them enough vertical room so the headline block remains readable in both
  // Excel and the rendered QA preview.
  sheets["Within Frame"].getRange("A4:C8").format.rowHeight = 62;
  writeObjectTable(
    sheets["Within Frame"],
    11,
    withinRows,
    "WithinWindowPairTable",
    ["source_file", "channel", "window_a", "window_b", "overlap_deg", "is_nonoverlap_control_pair", "median_similarity", "ci_low", "ci_high", "scan_support", "support_required", "sufficient_support"],
  );
  sheets["Within Frame"].freezePanes.freezeRows(11);
  sheets["Within Frame"].freezePanes.freezeColumns(3);

  // Uncertainty: fixed headline cell B5 is the strict spots ACF median CI width.
  setupSheet(
    sheets["Uncertainty"],
    "Uncertainty / 不确定性与稳健性",
    "CI 越窄代表不同 scan 重采样下越稳定；空白通常表示 support 不足，不应填成 0。",
    10,
  );
  writeKpis(sheets["Uncertainty"], 3, [
    { label: "CI records", value: ciRecords.length, numberFormat: "#,##0", explanation: "从 across/within summary 中整理出的 estimate 与 95% CI 记录数。" },
    { label: "Spots strict ACF median CI width", value: strictCiWidth, numberFormat: "0.000", explanation: "主 across-frame AUC 的典型 95% CI 宽度；越小越稳定，但仍需看支持数。" },
    { label: "All median CI width", value: allCiWidth, numberFormat: "0.000", explanation: "所有可用 across/within CI 宽度的中位数，只用于整体 QC。" },
    { label: "Robustness rows", value: robustnessRows.length, numberFormat: "#,##0", explanation: "独立 robustness tables 的总行数。" },
    { label: "Validation reports", value: validationReports.length, numberFormat: "#,##0", explanation: "被本工作簿读取的验证报告数量。" },
  ]);
  widenKpiExplanation(sheets["Uncertainty"], 3, 5, 10);
  const ciTable = writeObjectTable(
    sheets["Uncertainty"],
    11,
    ciRecords,
    "ConfidenceIntervalTable",
    ["source_file", "channel", "analysis", "family", "item", "metric", "estimate", "ci_low", "ci_high", "ci_width", "note"],
  );
  writeObjectTable(
    sheets["Uncertainty"],
    ciTable.endRow + 3,
    robustnessRows,
    "RobustnessTable",
    ["source_file", "channel", "analysis", "metric", "status", "passed", "value", "difference", "tolerance", "note"],
  );
  sheets["Uncertainty"].freezePanes.freezeRows(11);
  sheets["Uncertainty"].freezePanes.freezeColumns(2);

  // Methods combines a concise human description with the complete frozen config.
  setupSheet(
    sheets.Methods,
    "Methods / 统一算法说明",
    "这里记录的是对所有同类 XY 一视同仁的固定规则；真实结果不会反向改变参数。",
    8,
  );
  const methodRows = [
    { analysis: "Preprocessing", uniform_method: "检查有限值/单调 2θ/重复坐标；使用 ≥90% frames 覆盖的最大连续范围；禁止外推。", beginner_meaning: "先把不同文件放到共同、真实存在的角度范围内。" },
    { analysis: "Peak detection", uniform_method: "AsLS baseline + σ=1 bin Gaussian；prominence≥5σnoise、height≥3σnoise、width≥2 bins。", beginner_meaning: "只有明显高于噪声的凸起才作为候选峰。" },
    { analysis: "Peak fitting", uniform_method: "linear background + multiple pseudo-Voigt，soft_l1；ΔBIC≥10、area/SE≥3、参数不撞边界。", beginner_meaning: "重叠峰必须真的能被统计上区分，否则记 unknown。" },
    { analysis: "Radial tracking", uniform_method: "pressure 内 complete-link q clustering；pressure 间 bidirectional Hungarian constant-velocity matching。", beginner_meaning: "从低压和高压两端都能一致连起来，才保留同一个 radial peak 身份。" },
    { analysis: "Per-peak scores", uniform_method: "Area=min/max；location 为峰宽归一化 Gaussian；presence=Jaccard；缺失为 NaN。", beginner_meaning: "峰强、峰位和有没有峰分开回答，缺数据不会伪装成负相关。" },
    { analysis: "Across-frame", uniform_method: "strict same-window ACF 为主；direct strict 为位置敏感验证；±1 step max 仅为 secondary。", beginner_meaning: "主要看相同角度窗口随压力是否保持内部 pattern。" },
    { analysis: "Within-frame", uniform_method: "同一 frame 的 window-to-window ACF Pearson；每隔 5 windows 形成 non-overlap control。", beginner_meaning: "判断不同角度段是否出现重复结构，主看不共享数据的窗口。" },
    { analysis: "Uncertainty", uniform_method: "2,000 次 scan-level bootstrap，seed=0；near/far 用非零压力差的 Q25/Q75。", beginner_meaning: "把 scan 当独立单位重抽，避免把同一 scan 的点误当成很多独立样本。" },
  ];
  const methodTable = writeObjectTable(
    sheets.Methods,
    4,
    methodRows,
    "MethodOverviewTable",
    ["analysis", "uniform_method", "beginner_meaning"],
  );
  sheets.Methods.getRange(`A5:C${methodTable.endRow}`).format.rowHeight = 60;
  sheets.Methods.getRange(`A5:C${methodTable.endRow}`).format.wrapText = true;
  sheets.Methods.getRange(`A4:A${methodTable.endRow}`).format.columnWidth = 25;
  sheets.Methods.getRange(`B4:B${methodTable.endRow}`).format.columnWidth = 64;
  sheets.Methods.getRange(`C4:C${methodTable.endRow}`).format.columnWidth = 68;
  const configRows = flattenJson(algorithmConfig).map((row) => ({ section: "algorithm_config", ...row }));
  writeObjectTable(
    sheets.Methods,
    methodTable.endRow + 3,
    configRows,
    "AlgorithmConfigTable",
    ["section", "field", "value"],
  );
  sheets.Methods.freezePanes.freezeRows(4);

  // File Index: fixed headline cell B4 is the artifact count.
  setupSheet(
    sheets["File Index"],
    "File Index / 结果文件索引",
    "每一行对应一个输出 artifact；relative_path、bytes 与 SHA256 用于复现和完整性检查。",
    10,
  );
  writeKpis(
    sheets["File Index"],
    3,
    [
      {
        label: "Indexed artifacts",
        value: artifactRows.length,
        numberFormat: "#,##0",
        explanation: "artifact_index.csv 中记录的输出文件数。",
      },
      {
        label: "Indexed bytes",
        value: artifactRows.reduce((sum, row) => sum + (asNumber(row.bytes) ?? 0), 0),
        numberFormat: "#,##0",
        explanation: "索引内文件大小之和；工作簿本身可能在索引更新前生成，因此以最终 artifact index 为准。",
      },
    ],
  );
  widenKpiExplanation(sheets["File Index"], 3, 2, 10);
  sheets["File Index"].getRange("A4:C5").format.rowHeight = 66;
  writeObjectTable(
    sheets["File Index"],
    7,
    artifactRows,
    "ArtifactIndexTable",
    ["relative_path", "extension", "bytes", "sha256", "shape", "finite_count", "nan_count", "source_file"],
  );
  sheets["File Index"].freezePanes.freezeRows(7);
  sheets["File Index"].freezePanes.freezeColumns(1);

  // Summary is written last so every quoted cross-sheet formula has a target.
  setupSheet(
    sheets.Summary,
    "Uniform XY Correlation v2 / 结果总览",
    "小白版：先看算法是否通过，再看科学证据强不强；相关性弱不代表程序失败。",
    6,
  );
  sheets.Summary.getRange("A4:C4").values = [["你想知道的问题", "本次结果", "怎么理解"]];
  styleHeader(sheets.Summary.getRange("A4:C4"));
  const summaryRows = [
    [
      "算法有没有按计划运行？",
      '=IF(\'Run & QC\'!B5,"通过","未通过")',
      validationPassed === true
        ? "现有 validation reports 全部通过。注意：这只说明算法和输出结构通过检查，不保证科学信号一定强。"
        : "至少一个 validation report 未通过或没有明确通过；请先修复 QC，再解释科学结果。",
      null,
    ],
    [
      "主数据自动找到多少个可靠 radial peaks？",
      "='Per Peak'!B5",
      "这是 spots 的 official radial tracks 数量。1D XY 已经没有方位角，因此不能把它当成原来 40 个 spot tracks。",
      "#,##0",
    ],
    [
      "近压力和远压力的窗口 pattern 能区分吗？",
      '=IF(\'Across Frames\'!B5="","",\'Across Frames\'!B5)',
      "这是主分析 strict ACF 的中位 AUC：约 0.5 表示分不出来；越接近 1，近压力通常越相似。它不是相变概率。",
      "0.000",
    ],
    [
      "同一 frame 的不同角度段有重复 pattern 吗？",
      '=IF(\'Within Frame\'!B5="","",\'Within Frame\'!B5)',
      "这是不重叠 windows 的中位 Pearson similarity。接近 1 为同向相似，0 为无线性关系，负值为反向变化。",
      "0.000",
    ],
    [
      "主 across-frame 结果稳定吗？",
      '=IF(\'Uncertainty\'!B5="","",\'Uncertainty\'!B5)',
      "这是 spots strict ACF AUC 的典型 95% CI 宽度；越小越稳定。空白常表示 support 不足，不能当成 0。",
      "0.000",
    ],
    [
      "报告索引了多少个结果文件？",
      "='File Index'!B4",
      "所有图、矩阵、support、trajectory 与表格都应能从 File Index 找到。",
      "#,##0",
    ],
  ];
  summaryRows.forEach(([question, formula, explanation, format], index) => {
    const row = 5 + index;
    sheets.Summary.getRange(`A${row}`).values = [[question]];
    sheets.Summary.getRange(`B${row}`).formulas = [[formula]];
    sheets.Summary.getRange(`C${row}`).values = [[explanation]];
    if (format) sheets.Summary.getRange(`B${row}`).setNumberFormat(format);
    sheets.Summary.getRange(`A${row}:C${row}`).format = {
      fill: index % 2 === 0 ? COLORS.white : COLORS.light,
      font: { color: COLORS.text },
      wrapText: true,
      verticalAlignment: "center",
      borders: { bottom: { style: "thin", color: COLORS.line } },
    };
    sheets.Summary.getRange(`A${row}`).format.font = { bold: true, color: COLORS.text };
    sheets.Summary.getRange(`B${row}`).format.horizontalAlignment = "right";
    sheets.Summary.getRange(`A${row}:C${row}`).format.rowHeight = 45;
  });
  sheets.Summary.getRange("A4:A10").format.columnWidth = 34;
  sheets.Summary.getRange("B4:B10").format.columnWidth = 17;
  sheets.Summary.getRange("C4:C10").format.columnWidth = 74;
  sheets.Summary.getRange("A12:C12").merge();
  sheets.Summary.getRange("A12").values = [["科学结论是否充分？"]];
  sheets.Summary.getRange("A12:C12").format = {
    fill: COLORS.paleGold,
    font: { bold: true, color: COLORS.navy, size: 12 },
    borders: { preset: "outside", style: "thin", color: COLORS.line },
  };
  sheets.Summary.getRange("A13:C14").merge();
  sheets.Summary.getRange("A13").values = [[scientificEvidenceText(validationPassed, spotsStrictAuc)]];
  sheets.Summary.getRange("A13:C14").format = {
    fill: validationPassed === false ? COLORS.paleRed : COLORS.paleGreen,
    font: { color: COLORS.text, size: 11 },
    wrapText: true,
    verticalAlignment: "center",
    borders: { preset: "outside", style: "thin", color: COLORS.line },
  };
  sheets.Summary.getRange("A13:C14").format.rowHeight = 34;
  sheets.Summary.getRange("A16:C16").merge();
  sheets.Summary.getRange("A16").values = [[
    "阅读顺序：Run & QC → Per Peak → Across Frames（主证据）→ Within Frame（non-overlap control）→ Uncertainty。",
  ]];
  sheets.Summary.getRange("A16:C16").format = {
    fill: COLORS.paleBlue,
    font: { italic: true, color: COLORS.muted },
    wrapText: true,
  };
  sheets.Summary.freezePanes.freezeRows(4);

  // Render a readable key range from every sheet.  Long audit tables are kept
  // complete in the workbook but capped in previews to avoid multi-megapixel
  // images that are not useful for visual QA.
  await fs.mkdir(previewDir, { recursive: true });
  const previewRanges = {
    Summary: "A1:F18",
    "Run & QC": "A1:H32",
    "Per Peak": "A1:J32",
    "Across Frames": "A1:J32",
    "Within Frame": "A1:J32",
    Uncertainty: "A1:J32",
    Methods: "A1:H32",
    "File Index": "A1:J32",
  };
  const previewFiles = [];
  for (let index = 0; index < SHEET_NAMES.length; index += 1) {
    const sheetName = SHEET_NAMES[index];
    const preview = await workbook.render({
      sheetName,
      range: previewRanges[sheetName],
      scale: 1.25,
      format: "png",
    });
    const previewPath = path.join(
      previewDir,
      `${String(index + 1).padStart(2, "0")}_${normalizeName(sheetName)}.png`,
    );
    await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
    previewFiles.push(previewPath);
  }

  const keyInspections = {};
  for (const [sheetName, range] of Object.entries(previewRanges)) {
    const inspected = await workbook.inspect({
      kind: "table",
      range: `'${sheetName}'!${range}`,
      include: "values,formulas",
      tableMaxRows: 18,
      tableMaxCols: 10,
      maxChars: 3500,
    });
    keyInspections[sheetName] = compactInspect(inspected.ndjson, 4);
  }
  const formulaErrorInspection = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!",
    options: { useRegex: true, maxResults: 100 },
    summary: "final formula error scan",
    maxChars: 4000,
  });

  await fs.mkdir(path.dirname(outputXlsx), { recursive: true });
  const exported = await SpreadsheetFile.exportXlsx(workbook);
  await exported.save(outputXlsx);

  const output = {
    ok: true,
    output_xlsx: outputXlsx,
    preview_dir: previewDir,
    sheets: SHEET_NAMES,
    source_counts: {
      per_peak_rows: peakRows.length,
      across_rows: acrossRows.length,
      within_rows: withinRows.length,
      confidence_interval_rows: ciRecords.length,
      robustness_rows: robustnessRows.length,
      validation_reports: validationReports.length,
      artifact_rows: artifactRows.length,
    },
    inspected_ranges: previewRanges,
    formula_error_scan: compactInspect(formulaErrorInspection.ndjson, 20),
    key_inspection_record_counts: Object.fromEntries(
      Object.entries(keyInspections).map(([name, records]) => [name, records.length]),
    ),
    previews: previewFiles,
  };
  process.stdout.write(`${JSON.stringify(output)}\n`);
}

main().catch((error) => {
  process.stderr.write(
    `${JSON.stringify({ ok: false, error: error?.message ?? String(error), stack: error?.stack ?? "" })}\n`,
  );
  process.exitCode = 1;
});
