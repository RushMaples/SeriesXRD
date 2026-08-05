import type { FacetResponse, PlotPage, PlotRecord } from "./types";

const API_BASE = String(import.meta.env.VITE_API_BASE ?? "/api").replace(/\/$/, "");

type FetchPlotsInput = {
  query: string;
  selected: Record<string, string[]>;
  pressureMin: string;
  pressureMax: string;
  sort: string;
  page: number;
  pageSize: number;
  ids?: string[];
};

export interface IndexMeta {
  generated_at?: string | null;
  summary?: {
    plot_records?: number;
    assets?: number;
    audit_status?: string;
    [key: string]: unknown;
  };
  audit_status?: string;
}

const SERVER_FIELD_NAMES: Record<string, string> = {
  signal_channel: "channel",
  algorithm_variant: "method",
  aggregation_level: "scope",
};

function coerceItems(payload: unknown): PlotRecord[] {
  if (Array.isArray(payload)) return payload as PlotRecord[];
  if (!payload || typeof payload !== "object") return [];
  const object = payload as Record<string, unknown>;
  const candidate = object.items ?? object.plots ?? object.data ?? object.results;
  return Array.isArray(candidate) ? (candidate as PlotRecord[]) : [];
}

function coerceFacets(payload: unknown): FacetResponse {
  if (!payload || typeof payload !== "object") return {};
  const object = payload as Record<string, unknown>;
  const candidate = object.facets ?? object.aggregations ?? {};
  return candidate && typeof candidate === "object" ? (candidate as FacetResponse) : {};
}

export async function fetchPlots(input: FetchPlotsInput, signal?: AbortSignal): Promise<PlotPage> {
  const params = new URLSearchParams();
  if (input.query.trim()) params.set("q", input.query.trim());
  params.set("page", String(input.page));
  params.set("page_size", String(input.pageSize));
  params.set("sort", input.sort);
  if (input.pressureMin) params.set("pressure_min", input.pressureMin);
  if (input.pressureMax) params.set("pressure_max", input.pressureMax);
  if (input.ids?.length) params.set("ids", input.ids.join(","));
  for (const [field, values] of Object.entries(input.selected)) {
    const serverField = SERVER_FIELD_NAMES[field] ?? field;
    for (const value of values) params.append(serverField, value);
  }

  const response = await fetch(`${API_BASE}/plots?${params.toString()}`, {
    signal,
    headers: { Accept: "application/json" },
  });
  if (!response.ok) throw new Error(`Plot index request failed (${response.status})`);
  const payload = (await response.json()) as Record<string, unknown> | PlotRecord[];
  const items = coerceItems(payload);
  const object: Record<string, unknown> = Array.isArray(payload) ? {} : payload;
  const total = Number(object.total ?? object.count ?? items.length);
  const indexTotal = Number(object.index_total ?? object.indexTotal ?? total);
  const page = Number(object.page ?? input.page);
  const pageSize = Number(object.page_size ?? object.pageSize ?? input.pageSize);
  const updatedAt = (object.updated_at ?? object.updatedAt ?? null) as string | null;
  return { items, total, indexTotal, page, pageSize, facets: coerceFacets(payload), updatedAt };
}

export async function fetchMeta(signal?: AbortSignal): Promise<IndexMeta> {
  const response = await fetch(`${API_BASE}/meta`, { signal, headers: { Accept: "application/json" } });
  if (!response.ok) throw new Error(`Index metadata request failed (${response.status})`);
  return (await response.json()) as IndexMeta;
}

export async function fetchPlot(plotId: string, signal?: AbortSignal): Promise<PlotRecord | null> {
  const direct = await fetch(`${API_BASE}/plots/${encodeURIComponent(plotId)}`, {
    signal,
    headers: { Accept: "application/json" },
  });
  if (direct.ok) {
    const payload = (await direct.json()) as unknown;
    if (payload && typeof payload === "object" && "item" in payload) {
      return (payload as { item?: PlotRecord }).item ?? null;
    }
    return payload as PlotRecord;
  }
  if (direct.status !== 404 && direct.status !== 405) {
    throw new Error(`Plot detail request failed (${direct.status})`);
  }
  const page = await fetchPlots(
    {
      query: "",
      selected: {},
      pressureMin: "",
      pressureMax: "",
      sort: "pressure_desc",
      page: 1,
      pageSize: 1,
      ids: [plotId],
    },
    signal,
  );
  return page.items[0] ?? null;
}

export function fileUrl(path: string | null | undefined): string {
  if (!path) return "";
  if (/^(https?:|data:|blob:)/.test(path) || path.startsWith(`${API_BASE}/`)) return path;
  return "";
}

export function imageUrl(record: PlotRecord): string {
  return fileUrl(record.image_url || record.image_path || "");
}
