import type {
  CorrelationFamily,
  CorrelationTransform,
  DisplayProfileDomain,
  PlotRecord,
  SampleType,
  ValidationStatus,
  VisualizationType,
} from "./types";

const LABELS: Record<string, string> = {
  current_formal: "Current formal",
  baseline: "Baseline",
  legacy: "Legacy",
  exploratory: "Exploratory",
  validation: "Validation",
  powder: "Powder",
  single_crystal: "Single crystal",
  mixed: "Mixed",
  not_applicable: "N/A",
  original: "Original",
  log_squared: "Log² (Log-squared)",
  exp_squared: "Exp² (Exp-squared)",
  roi_area: "ROI area",
  location: "Peak location",
  window_across_frames: "Across frames — same 2θ window",
  window_to_window_across_frames: "Across frames — same 2θ window",
  window_within_frame: "Within one frame — window pairs",
  window_to_window_within_same_frame: "Within one frame — window pairs",
  transition_tracking: "Transition tracking",
  validation_diagnostic: "Validation / diagnostic",
  heatmap: "Heatmap",
  waterfall_shaded: "Waterfall · shaded",
  waterfall_unshaded: "Waterfall · unshaded",
  matrix: "Matrix",
  line_plot: "Line plot",
  transition_plot: "Transition plot",
  "3d_plot": "3D plot",
  validation_report: "Validation report",
  correlation_transform: "Correlation-transform profile",
  original_positive: "Original XY-derived (pre-denoise)",
  source_spots_channel_xy_pre_nonlinear_transform: "Source spots-channel XY, before Log transform",
  formal_correlation_transform_profile: "Nonlinear-transformed formal profile",
  spots: "Spots",
  fit_control: "Fit control",
  acf_strict: "ACF strict",
  direct_strict: "Direct strict",
  shift_tolerant_secondary: "Shift-tolerant secondary",
  aggregate: "Aggregate",
  by_pressure: "By pressure",
};

export function humanize(value: string | null | undefined): string {
  if (!value) return "N/A";
  return LABELS[value] ?? value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function sampleLabel(value: SampleType): string {
  return humanize(value);
}

export function transformLabel(value: CorrelationTransform): string {
  return humanize(value);
}

export function familyLabel(value: CorrelationFamily): string {
  return humanize(value);
}

export function visualizationLabel(value: VisualizationType): string {
  return humanize(value);
}

export function displayDomainLabel(value: DisplayProfileDomain): string {
  return humanize(value);
}

export function validationTone(status: ValidationStatus): "pass" | "fail" | "warning" | "unknown" {
  const normalized = String(status).toUpperCase();
  if (normalized === "PASS") return "pass";
  if (normalized === "FAIL") return "fail";
  if (normalized.includes("WARN")) return "warning";
  return "unknown";
}

export function formatPressure(value: number | null | undefined): string {
  return Number.isFinite(value) ? `${Number(value).toLocaleString(undefined, { maximumFractionDigits: 3 })} GPa` : "N/A";
}

export function formatNumber(value: number | null | undefined, digits = 4): string {
  return Number.isFinite(value)
    ? Number(value).toLocaleString(undefined, { maximumFractionDigits: digits })
    : "N/A";
}

export function localPeak(record: PlotRecord): number | null {
  return record.anchor_peak_number ?? record.anchor_local_peak_index ?? null;
}

export function runLabel(record: PlotRecord): string {
  return String(record.run ?? record.result_run ?? "N/A");
}

export function compactTitle(record: PlotRecord): string {
  const anchor = record.anchor_uid || record.frame_id || record.plot_id;
  const peak = localPeak(record);
  return peak == null ? String(anchor) : `${anchor} · Peak ${peak}`;
}

export function cardTitle(record: PlotRecord): string {
  const waterfall = String(record.visualization_type).startsWith("waterfall");
  const family = familyLabel(record.correlation_family);
  return waterfall ? `${family} (waterfall)` : family;
}

export function altText(record: PlotRecord): string {
  const pieces = [
    sampleLabel(record.sample),
    familyLabel(record.correlation_family),
    transformLabel(record.correlation_transform),
    record.anchor_pressure_gpa != null ? formatPressure(record.anchor_pressure_gpa) : null,
    localPeak(record) != null ? `local peak ${localPeak(record)}` : null,
  ].filter(Boolean);
  return `${pieces.join(", ")} correlation plot`;
}

export function isWaterfall(record: PlotRecord): boolean {
  return String(record.visualization_type).startsWith("waterfall");
}

export function windowLabel(record: PlotRecord): string {
  if (record.window_start_deg == null || record.window_end_deg == null) return "N/A";
  return `${formatNumber(record.window_start_deg, 2)}–${formatNumber(record.window_end_deg, 2)}°`;
}

export function sameSemanticCoordinates(records: PlotRecord[]): boolean {
  if (records.length < 2) return true;
  const first = records[0];
  return records.every(
    (record) =>
      record.correlation_family === first.correlation_family &&
      record.visualization_type === first.visualization_type &&
      record.sample === first.sample,
  );
}
