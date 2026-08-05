#!/usr/bin/env node

import { createReadStream, existsSync, readFileSync, statSync } from "node:fs";
import { createServer } from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const SERVER_DIR = path.dirname(fileURLToPath(import.meta.url));
const APP_ROOT = path.resolve(SERVER_DIR, "..");
const CORRELATIONS_ROOT = path.resolve(APP_ROOT, "..");
const RESULTS_ROOT = path.resolve(
  process.env.CORRELATION_RESULTS_ROOT || path.join(CORRELATIONS_ROOT, "results"),
);
const INDEX_PATH = path.join(APP_ROOT, "data", "plot-index.json");
const AUDIT_PATH = path.join(APP_ROOT, "data", "classification-audit.json");
const DIST_ROOT = path.join(APP_ROOT, "dist");
const PORT = Number(process.env.PORT || 4312);
const HOST = process.env.HOST || "127.0.0.1";

if (!existsSync(INDEX_PATH)) {
  throw new Error(`Missing ${INDEX_PATH}. Run npm run index first.`);
}

const index = JSON.parse(readFileSync(INDEX_PATH, "utf8"));
const audit = existsSync(AUDIT_PATH) ? JSON.parse(readFileSync(AUDIT_PATH, "utf8")) : null;
const records = index.records;
const recordsById = new Map(records.map((record) => [record.id, record]));
const assetsById = new Map(index.assets.map((asset) => [asset.id, asset]));

const FACET_FIELDS = [
  "result_status",
  "validation_status",
  "sample",
  "correlation_transform",
  "correlation_family",
  "visualization_type",
  "display_profile_domain",
  "channel",
  "method",
  "scope",
  "run_id",
];

const MIME = {
  ".css": "text/css; charset=utf-8",
  ".csv": "text/csv; charset=utf-8",
  ".gz": "application/gzip",
  ".html": "text/html; charset=utf-8",
  ".ico": "image/x-icon",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".map": "application/json; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml",
};

function isUnder(parent, candidate) {
  const relative = path.relative(parent, candidate);
  return relative !== ".." && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
}

function json(response, status, payload) {
  const body = JSON.stringify(payload);
  response.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(body),
    "Cache-Control": "no-store",
  });
  response.end(body);
}

function pressureOf(record) {
  return record.anchor?.pressure_gpa ?? record.pressure_gpa ?? record.window?.pressure_gpa ?? null;
}

function enrich(record) {
  const asset = assetsById.get(record.asset_id);
  const companionLinks = (record.companion_paths || []).map((item, companionIndex) => ({
    ...item,
    url: `/api/plots/${encodeURIComponent(record.id)}/companion/${encodeURIComponent(item.kind)}?index=${companionIndex}`,
  }));
  return {
    ...record,
    plot_id: record.id,
    result_run: record.run_id,
    image_url: `/api/plots/${encodeURIComponent(record.id)}/image`,
    companions: companionLinks,
    csv_path: companionLinks.find((item) => item.path.endsWith(".csv") || item.path.endsWith(".csv.gz"))?.path ?? null,
    json_path: companionLinks.find((item) => item.path.endsWith(".json"))?.path ?? null,
    anchor_uid: record.anchor?.point_uid ?? record.anchor?.token ?? null,
    anchor_pressure_gpa: record.anchor?.pressure_gpa ?? null,
    anchor_peak_number: record.anchor?.local_peak_index ?? null,
    anchor_local_peak_index: record.anchor?.local_peak_index ?? null,
    anchor_two_theta_deg: record.anchor?.two_theta_deg ?? null,
    anchor_q: record.anchor?.q ?? null,
    anchor_q_width: record.anchor?.observation_q_width_summary?.median ?? null,
    anchor_q_width_summary: record.anchor?.observation_q_width_summary ?? null,
    track_id: record.track?.track_id ?? record.track?.id ?? null,
    window_start_deg: record.window?.start_deg ?? null,
    window_end_deg: record.window?.end_deg ?? null,
    window_index: record.window?.index_0based ?? record.window?.index ?? null,
    frame_scope: record.scope ?? null,
    signal_channel: record.channel ?? null,
    algorithm_variant: record.method ?? null,
    aggregation_level: record.scope ?? null,
    classification_source: record.classification_source ?? "manifest_and_allowlisted_path",
    classification_warning: record.classification_warnings?.join("; ") || null,
    missing_value_semantics: record.matrix_semantics ?? null,
    strict_lower_triangle: record.triangle_policy === "strict_lower_no_diagonal",
    sha256: asset?.sha256 ?? null,
    aliases: asset?.alias_paths_relative ?? [],
    asset: asset
      ? {
          width: asset.width,
          height: asset.height,
          size_bytes: asset.size_bytes,
          alias_count: asset.aliases.length,
        }
      : null,
  };
}

function queryValues(params, field) {
  return [...params.getAll(field), ...params.getAll(field === "run_id" ? "result_run" : `__none_${field}`)]
    .flatMap((value) => value.split(","))
    .filter(Boolean);
}

function normalizeSearch(value) {
  return value
    .toLowerCase()
    .replaceAll("log²", "log squared")
    .replaceAll("q-width", "qwidth")
    .replaceAll("single-crystal", "single crystal")
    .trim();
}

function searchHaystack(record) {
  const aliases = {
    log_squared: "log log2 log squared logarithmic",
    roi_area: "roi roi area integrated iou peak shape intensity",
    location: "location peak center 2theta",
    window_to_window_across_frames: "window across frames same 2theta window",
    window_to_window_within_same_frame: "window within frame within one frame different window pairs",
    original_positive: "original profile original positive original xy xy files pre denoise",
    correlation_transform: "transformed profile denoised",
    single_crystal: "single crystal",
  };
  const anchor = record.anchor || {};
  const window = record.window || {};
  return normalizeSearch(
    [
      record.id,
      record.title,
      record.run_id,
      record.result_status,
      record.validation_status,
      record.sample,
      record.correlation_transform,
      record.correlation_family,
      record.visualization_type,
      record.display_profile_domain,
      record.channel,
      record.method,
      record.scope,
      anchor.token,
      anchor.point_uid,
      anchor.local_peak_index != null ? `peak ${anchor.local_peak_index}` : null,
      anchor.q != null ? `q ${anchor.q}` : null,
      anchor.two_theta_deg != null ? `2theta ${anchor.two_theta_deg}` : null,
      record.half_width_factor != null ? `qwidth ${record.half_width_factor}` : null,
      window.start_deg != null && window.end_deg != null ? `window ${window.start_deg}-${window.end_deg}` : null,
      ...(record.tags || []),
      aliases[record.correlation_transform],
      aliases[record.correlation_family],
      aliases[record.display_profile_domain],
      aliases[record.sample],
      record.visualization_type === "waterfall_shaded" ? "waterfall colors colour colored shading shaded" : null,
    ]
      .filter(Boolean)
      .join(" "),
  );
}

function buildFilter(params) {
  const rawSearch = params.get("search") || params.get("q") || "";
  let search = normalizeSearch(rawSearch);
  const pressureMatch = search.match(/(?:^|\s)(\d+(?:\.\d+)?)\s*gpa(?:\s|$)/);
  const exactSearchPressure = pressureMatch ? Number(pressureMatch[1]) : null;
  if (pressureMatch) search = search.replace(pressureMatch[0], " ").trim();
  const peakMatch = search.match(/(?:^|\s)(?:local\s+)?peak\s+(\d+)(?:\s|$)/);
  const exactSearchPeak = peakMatch ? Number(peakMatch[1]) : null;
  if (peakMatch) search = search.replace(peakMatch[0], " ").trim();
  const words = search.split(/\s+/).filter(Boolean);
  const pressureMinValue = params.get("pressureMin") ?? params.get("pressure_min");
  const pressureMaxValue = params.get("pressureMax") ?? params.get("pressure_max");
  const pressureMin = pressureMinValue !== null && pressureMinValue !== "" ? Number(pressureMinValue) : null;
  const pressureMax = pressureMaxValue !== null && pressureMaxValue !== "" ? Number(pressureMaxValue) : null;
  const ids = (params.get("ids") || "").split(",").filter(Boolean);
  const selected = Object.fromEntries(FACET_FIELDS.map((field) => [field, queryValues(params, field)]));

  const matches = (record, omittedFacet = null) => {
    if (ids.length && !ids.includes(record.id)) return false;
    const pressure = pressureOf(record);
    if (exactSearchPressure !== null && pressure !== exactSearchPressure) return false;
    if (exactSearchPeak !== null && record.anchor?.local_peak_index !== exactSearchPeak) return false;
    if (pressureMin !== null && (pressure === null || pressure < pressureMin)) return false;
    if (pressureMax !== null && (pressure === null || pressure > pressureMax)) return false;
    if (words.length) {
      const haystack = searchHaystack(record);
      if (!words.every((word) => haystack.includes(word))) return false;
    }
    for (const field of FACET_FIELDS) {
      if (field === omittedFacet || !selected[field].length) continue;
      const value = record[field] ?? "not_applicable";
      if (!selected[field].includes(String(value))) return false;
    }
    return true;
  };
  return { matches, selected };
}

function facetsFor(filter) {
  const facets = {};
  for (const field of FACET_FIELDS) {
    const counts = new Map();
    for (const record of records) {
      if (!filter.matches(record, field)) continue;
      const value = String(record[field] ?? "not_applicable");
      counts.set(value, (counts.get(value) || 0) + 1);
    }
    facets[field] = [...counts.entries()]
      .map(([value, count]) => ({ value, count }))
      .sort((a, b) => b.count - a.count || a.value.localeCompare(b.value));
  }
  return facets;
}

function sortRecords(items, sort) {
  const [field, direction = "asc"] = sort.split("_").reduce(
    (result, part, index, all) => (index === all.length - 1 && ["asc", "desc"].includes(part) ? [all.slice(0, -1).join("_"), part] : result),
    [sort, "asc"],
  );
  const multiplier = direction === "desc" ? -1 : 1;
  return [...items].sort((a, b) => {
    const valueFor = (record) => {
      if (field === "pressure") return pressureOf(record);
      if (field === "peak") return record.anchor?.local_peak_index ?? null;
      if (field === "two_theta") return record.anchor?.two_theta_deg ?? null;
      if (field === "title") return record.title;
      if (field === "run") return record.run_id;
      return record.id;
    };
    const left = valueFor(a);
    const right = valueFor(b);
    if (left == null && right == null) return a.id.localeCompare(b.id);
    if (left == null) return 1;
    if (right == null) return -1;
    const compared = typeof left === "number" && typeof right === "number" ? left - right : String(left).localeCompare(String(right));
    return compared === 0 ? a.id.localeCompare(b.id) : compared * multiplier;
  });
}

function contentDisposition(filePath) {
  const safeName = path.basename(filePath).replace(/["\r\n]/g, "_");
  return `inline; filename="${safeName}"`;
}

function serveAllowedFile(request, response, filePath, options = {}) {
  const absolute = path.resolve(filePath);
  if (!isUnder(RESULTS_ROOT, absolute) || !existsSync(absolute) || !statSync(absolute).isFile()) {
    return json(response, 404, { error: "Indexed file is unavailable." });
  }
  const stats = statSync(absolute);
  const headers = {
    "Content-Type": MIME[path.extname(absolute).toLowerCase()] || "application/octet-stream",
    "Content-Length": stats.size,
    "Cache-Control": options.cache || "private, max-age=3600",
    "Content-Disposition": contentDisposition(absolute),
    "X-Content-Type-Options": "nosniff",
  };
  if (options.etag) headers.ETag = `"${options.etag}"`;
  response.writeHead(200, headers);
  if (request.method === "HEAD") return response.end();
  createReadStream(absolute).pipe(response);
}

function plotRoute(pathname) {
  const prefix = "/api/plots/";
  if (!pathname.startsWith(prefix)) return null;
  const remainder = pathname.slice(prefix.length);
  const companionMarker = "/companion/";
  if (remainder.includes(companionMarker)) {
    const [encodedId, encodedKind] = remainder.split(companionMarker);
    return { id: decodeURIComponent(encodedId), kind: decodeURIComponent(encodedKind), type: "companion" };
  }
  if (remainder.endsWith("/image")) {
    return { id: decodeURIComponent(remainder.slice(0, -6)), type: "image" };
  }
  return { id: decodeURIComponent(remainder), type: "detail" };
}

function handleApi(request, response, url) {
  if (url.pathname === "/api/meta") {
    return json(response, 200, {
      schema_version: index.schema_version,
      generated_at: index.generated_at,
      summary: index.summary,
      audit_status: audit?.status || "UNKNOWN",
      audit_summary: audit?.summary || null,
      result_roots: index.result_roots,
    });
  }
  if (url.pathname === "/api/audit") return json(response, 200, audit || { status: "UNKNOWN" });
  if (url.pathname === "/api/plots") {
    const filter = buildFilter(url.searchParams);
    const matching = records.filter((record) => filter.matches(record));
    const sort = url.searchParams.get("sort") || "pressure_desc";
    const requestedPage = Number(url.searchParams.get("page") || 1);
    const requestedPageSize = Number(url.searchParams.get("pageSize") || url.searchParams.get("page_size") || 24);
    const page = Number.isFinite(requestedPage) ? Math.max(1, Math.trunc(requestedPage)) : 1;
    const pageSize = Number.isFinite(requestedPageSize) ? Math.min(100, Math.max(1, Math.trunc(requestedPageSize))) : 24;
    const ordered = sortRecords(matching, sort);
    const start = (page - 1) * pageSize;
    return json(response, 200, {
      items: ordered.slice(start, start + pageSize).map(enrich),
      total: matching.length,
      index_total: records.length,
      page,
      pageSize,
      page_size: pageSize,
      facets: facetsFor(filter),
      updatedAt: index.generated_at,
      updated_at: index.generated_at,
    });
  }

  const route = plotRoute(url.pathname);
  if (!route) return json(response, 404, { error: "Unknown API endpoint." });
  const record = recordsById.get(route.id);
  if (!record) return json(response, 404, { error: "Unknown plot id." });
  if (route.type === "detail") return json(response, 200, enrich(record));
  if (route.type === "image") {
    const asset = assetsById.get(record.asset_id);
    return serveAllowedFile(request, response, record.image_path, {
      cache: "private, max-age=86400, immutable",
      etag: asset?.sha256,
    });
  }
  const companionIndex = Number(url.searchParams.get("index") || 0);
  const all = record.companion_paths || [];
  const selected = all[companionIndex]?.kind === route.kind ? all[companionIndex] : all.find((item) => item.kind === route.kind);
  if (!selected) return json(response, 404, { error: "Unknown companion for this plot." });
  return serveAllowedFile(request, response, selected.path, { cache: "private, max-age=3600" });
}

function serveStatic(request, response, url) {
  if (!existsSync(DIST_ROOT)) {
    return json(response, 503, { error: "Frontend build is missing. Run npm run build." });
  }
  const requested = url.pathname === "/" ? "index.html" : decodeURIComponent(url.pathname.slice(1));
  let candidate = path.resolve(DIST_ROOT, requested);
  if (!isUnder(DIST_ROOT, candidate) || !existsSync(candidate) || !statSync(candidate).isFile()) {
    candidate = path.join(DIST_ROOT, "index.html");
  }
  const stats = statSync(candidate);
  response.writeHead(200, {
    "Content-Type": MIME[path.extname(candidate).toLowerCase()] || "application/octet-stream",
    "Content-Length": stats.size,
    "Cache-Control": candidate.endsWith("index.html") ? "no-cache" : "public, max-age=31536000, immutable",
  });
  if (request.method === "HEAD") return response.end();
  createReadStream(candidate).pipe(response);
}

const server = createServer((request, response) => {
  try {
    if (!["GET", "HEAD"].includes(request.method || "")) return json(response, 405, { error: "Read-only API: GET and HEAD only." });
    const url = new URL(request.url || "/", `http://${request.headers.host || `${HOST}:${PORT}`}`);
    if (url.pathname.startsWith("/api/")) return handleApi(request, response, url);
    return serveStatic(request, response, url);
  } catch (error) {
    console.error(error);
    return json(response, 500, { error: "Internal explorer error." });
  }
});

server.listen(PORT, HOST, () => {
  console.log(`XRD Correlation Atlas API: http://${HOST}:${PORT}`);
  console.log(`Loaded ${records.length} logical plots and ${index.assets.length} assets.`);
});
