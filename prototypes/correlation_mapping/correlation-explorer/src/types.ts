export type ResultStatus =
  | "current_formal"
  | "baseline"
  | "legacy"
  | "exploratory"
  | "validation"
  | string;

export type ValidationStatus = "PASS" | "FAIL" | "WARNING" | "UNKNOWN" | string;
export type SampleType = "powder" | "single_crystal" | "mixed" | "not_applicable" | string;
export type CorrelationTransform =
  | "original"
  | "log_squared"
  | "exp_squared"
  | "not_applicable"
  | string;
export type CorrelationFamily =
  | "roi_area"
  | "location"
  | "window_across_frames"
  | "window_within_frame"
  | "transition_tracking"
  | "validation_diagnostic"
  | string;
export type VisualizationType =
  | "heatmap"
  | "waterfall_shaded"
  | "waterfall_unshaded"
  | "matrix"
  | "line_plot"
  | "transition_plot"
  | "3d_plot"
  | "validation_report"
  | string;
export type DisplayProfileDomain =
  | "correlation_transform"
  | "original_positive"
  | "not_applicable"
  | string;

export interface CompanionMap {
  csv?: string | string[];
  json?: string | string[];
  image?: string | string[];
  [key: string]: string | string[] | undefined;
}

export interface CompanionFile {
  kind: string;
  path: string;
  url?: string;
}

export interface PlotRecord {
  plot_id: string;
  title: string;
  image_path: string | null;
  image_url?: string | null;
  companions?: CompanionMap | Array<string | CompanionFile> | null;
  csv_path?: string | null;
  json_path?: string | null;
  run?: string | null;
  result_run?: string | null;
  result_status: ResultStatus;
  validation_status: ValidationStatus;
  sample: SampleType;
  correlation_transform: CorrelationTransform;
  correlation_family: CorrelationFamily;
  visualization_type: VisualizationType;
  display_profile_domain: DisplayProfileDomain;
  display_profile_source?: string | null;
  display_profile_construction?: string | null;
  anchor_uid?: string | null;
  anchor_pressure_gpa?: number | null;
  anchor_peak_number?: number | null;
  anchor_local_peak_index?: number | null;
  anchor_two_theta_deg?: number | null;
  anchor_q?: number | null;
  anchor_q_width?: number | null;
  half_width_factor?: number | null;
  track_id?: string | number | null;
  frame_id?: string | number | null;
  window_start_deg?: number | null;
  window_end_deg?: number | null;
  window_index?: number | null;
  frame_scope?: string | null;
  pressure_gpa?: number | null;
  strict_lower_triangle?: boolean | null;
  signal_channel?: string | null;
  algorithm_variant?: string | null;
  aggregation_level?: string | null;
  classification_source?: string | null;
  classification_warning?: string | null;
  missing_value_semantics?: string | Record<string, unknown> | null;
  asset_id?: string | null;
  sha256?: string | null;
  aliases?: string[];
  logical_contexts?: Array<Record<string, unknown>>;
  indexed_at?: string | null;
  [key: string]: unknown;
}

export interface FacetEntry {
  value: string;
  label?: string;
  count: number;
}

export type FacetResponse = Record<string, FacetEntry[] | Record<string, number>>;

export interface PlotPage {
  items: PlotRecord[];
  total: number;
  indexTotal: number;
  page: number;
  pageSize: number;
  facets: FacetResponse;
  updatedAt?: string | null;
}

export type ViewMode = "grid" | "table";
export type AppMode = "library" | "compare";

export interface Filters {
  query: string;
  selected: Record<string, string[]>;
  pressureMin: string;
  pressureMax: string;
  savedMode: "all" | "favorites" | "collection";
  collectionName: string;
}

export interface UrlState {
  mode: AppMode;
  filters: Filters;
  sort: string;
  view: ViewMode;
  page: number;
  pageSize: number;
  selectedPlotId: string | null;
  compareIds: string[];
}

export interface SavedLibrary {
  version: 1;
  favorites: string[];
  collections: Record<string, string[]>;
}
