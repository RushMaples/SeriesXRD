import type { Filters, UrlState } from "./types";

export const FACET_PARAM_MAP: Record<string, string> = {
  result_status: "status",
  validation_status: "validation",
  sample: "sample",
  correlation_transform: "transform",
  correlation_family: "family",
  visualization_type: "viz",
  display_profile_domain: "display",
  signal_channel: "channel",
  algorithm_variant: "algorithm",
  aggregation_level: "aggregation",
};

export const DEFAULT_FILTERS: Filters = {
  query: "",
  selected: { result_status: ["current_formal"], validation_status: ["PASS"] },
  pressureMin: "",
  pressureMax: "",
  savedMode: "all",
  collectionName: "",
};

export const DEFAULT_STATE: UrlState = {
  mode: "library",
  filters: DEFAULT_FILTERS,
  sort: "pressure_desc",
  view: "grid",
  page: 1,
  pageSize: 24,
  selectedPlotId: null,
  compareIds: [],
};

function readMulti(params: URLSearchParams, key: string): string[] {
  return params
    .getAll(key)
    .flatMap((value) => value.split(","))
    .map((value) => value.trim())
    .filter(Boolean);
}

export function readUrlState(search = window.location.search): UrlState {
  const params = new URLSearchParams(search);
  const hasExplicitFilters = params.has("all") || Object.values(FACET_PARAM_MAP).some((key) => params.has(key));
  const selected: Record<string, string[]> = {};
  for (const [field, parameter] of Object.entries(FACET_PARAM_MAP)) {
    const values = readMulti(params, parameter);
    if (values.length) selected[field] = values;
  }
  if (!hasExplicitFilters) {
    selected.result_status = ["current_formal"];
    selected.validation_status = ["PASS"];
  }

  const saved = params.get("saved") ?? "all";
  const collectionName = params.get("collection") ?? "";
  const savedMode = saved === "favorites" ? "favorites" : collectionName ? "collection" : "all";
  const page = Math.max(1, Number(params.get("page") ?? 1) || 1);
  const pageSize = [12, 24, 48, 96].includes(Number(params.get("page_size")))
    ? Number(params.get("page_size"))
    : 24;

  return {
    mode: params.get("mode") === "compare" ? "compare" : "library",
    filters: {
      query: params.get("q") ?? "",
      selected,
      pressureMin: params.get("pressure_min") ?? "",
      pressureMax: params.get("pressure_max") ?? "",
      savedMode,
      collectionName,
    },
    sort: params.get("sort") ?? "pressure_desc",
    view: params.get("view") === "table" ? "table" : "grid",
    page,
    pageSize,
    selectedPlotId: params.get("plot"),
    compareIds: readMulti(params, "compare").slice(0, 4),
  };
}

export function writeUrlState(state: UrlState): void {
  const params = new URLSearchParams();
  if (state.mode === "compare") params.set("mode", "compare");
  if (state.filters.query.trim()) params.set("q", state.filters.query.trim());
  const selectedEntries = Object.entries(state.filters.selected).filter(([, values]) => values.length);
  if (!selectedEntries.length) params.set("all", "1");
  for (const [field, values] of selectedEntries) {
    const parameter = FACET_PARAM_MAP[field] ?? field;
    for (const value of values) params.append(parameter, value);
  }
  if (state.filters.pressureMin) params.set("pressure_min", state.filters.pressureMin);
  if (state.filters.pressureMax) params.set("pressure_max", state.filters.pressureMax);
  if (state.filters.savedMode === "favorites") params.set("saved", "favorites");
  if (state.filters.savedMode === "collection" && state.filters.collectionName) {
    params.set("collection", state.filters.collectionName);
  }
  if (state.sort !== "pressure_desc") params.set("sort", state.sort);
  if (state.view !== "grid") params.set("view", state.view);
  if (state.page !== 1) params.set("page", String(state.page));
  if (state.pageSize !== 24) params.set("page_size", String(state.pageSize));
  if (state.selectedPlotId) params.set("plot", state.selectedPlotId);
  for (const plotId of state.compareIds) params.append("compare", plotId);
  const query = params.toString();
  const next = `${window.location.pathname}${query ? `?${query}` : ""}`;
  window.history.replaceState(null, "", next);
}
