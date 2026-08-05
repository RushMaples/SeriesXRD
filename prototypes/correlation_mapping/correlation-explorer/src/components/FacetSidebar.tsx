import { ChevronDown, ChevronUp, Folder, Star } from "lucide-react";
import { memo, useMemo, useState } from "react";

import { humanize } from "../format";
import type { FacetEntry, FacetResponse, Filters, SavedLibrary } from "../types";

type FacetSidebarProps = {
  facets: FacetResponse;
  filters: Filters;
  library: SavedLibrary;
  total: number;
  onToggleFacet: (field: string, value: string) => void;
  onPressureChange: (minimum: string, maximum: string) => void;
  onSavedModeChange: (mode: Filters["savedMode"], collectionName?: string) => void;
  onClear: () => void;
};

type FacetSpec = { field: string; label: string; preferred?: string[] };

const FACET_SPECS: FacetSpec[] = [
  {
    field: "result_status",
    label: "Result status",
    preferred: ["current_formal", "baseline", "legacy", "exploratory", "validation"],
  },
  { field: "validation_status", label: "Validation", preferred: ["PASS", "WARNING", "FAIL", "UNKNOWN"] },
  { field: "sample", label: "Sample", preferred: ["powder", "single_crystal", "mixed", "not_applicable"] },
  {
    field: "correlation_transform",
    label: "Correlation calculation",
    preferred: ["log_squared", "original", "not_applicable"],
  },
  {
    field: "correlation_family",
    label: "Correlation family",
    preferred: [
      "roi_area",
      "location",
      "window_to_window_across_frames",
      "window_to_window_within_same_frame",
      "transition_tracking",
      "validation_diagnostic",
    ],
  },
  {
    field: "visualization_type",
    label: "Visualization",
    preferred: [
      "heatmap",
      "waterfall_shaded",
      "waterfall_unshaded",
      "matrix",
      "line_plot",
      "transition_plot",
      "3d_plot",
      "validation_report",
    ],
  },
  {
    field: "display_profile_domain",
    label: "Curve displayed",
    preferred: ["correlation_transform", "original_positive", "not_applicable"],
  },
  { field: "signal_channel", label: "Signal channel", preferred: ["spots", "fit_control"] },
  {
    field: "algorithm_variant",
    label: "Algorithm",
    preferred: ["acf_strict", "direct_strict", "shift_tolerant_secondary"],
  },
  { field: "aggregation_level", label: "Aggregation", preferred: ["by_pressure", "aggregate"] },
];

const SERVER_FACET_FIELDS: Record<string, string> = {
  signal_channel: "channel",
  algorithm_variant: "method",
  aggregation_level: "scope",
};

function normalizeFacet(facets: FacetResponse, spec: FacetSpec): FacetEntry[] {
  const raw = facets[SERVER_FACET_FIELDS[spec.field] ?? spec.field];
  const entries: FacetEntry[] = Array.isArray(raw)
    ? raw.map((entry) => ({ value: String(entry.value), label: entry.label, count: Number(entry.count) || 0 }))
    : raw && typeof raw === "object"
      ? Object.entries(raw).map(([value, count]) => ({ value, count: Number(count) || 0 }))
      : [];
  const known = new Map(entries.map((entry) => [entry.value, entry]));
  for (const value of spec.preferred ?? []) {
    if (!known.has(value)) known.set(value, { value, count: 0 });
  }
  const preference = new Map((spec.preferred ?? []).map((value, index) => [value, index]));
  return [...known.values()].sort((left, right) => {
    const leftIndex = preference.get(left.value) ?? Number.MAX_SAFE_INTEGER;
    const rightIndex = preference.get(right.value) ?? Number.MAX_SAFE_INTEGER;
    return leftIndex - rightIndex || left.value.localeCompare(right.value);
  });
}

const FacetGroup = memo(function FacetGroup({
  spec,
  entries,
  selected,
  onToggle,
  initiallyOpen = true,
}: {
  spec: FacetSpec;
  entries: FacetEntry[];
  selected: string[];
  onToggle: (field: string, value: string) => void;
  initiallyOpen?: boolean;
}) {
  const [open, setOpen] = useState(initiallyOpen);
  return (
    <section className="facet-group">
      <button className="facet-heading" type="button" onClick={() => setOpen((value) => !value)} aria-expanded={open}>
        <span>{spec.label}</span>
        {open ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
      </button>
      {open ? (
        <div className="facet-options">
          {entries.map((entry) => {
            const checked = selected.includes(entry.value);
            return (
              <label className={entry.count === 0 && !checked ? "facet-option disabled-count" : "facet-option"} key={entry.value}>
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() => onToggle(spec.field, entry.value)}
                  disabled={entry.count === 0 && !checked}
                />
                <span>{entry.label ?? humanize(entry.value)}</span>
                <span className="facet-count">{entry.count.toLocaleString()}</span>
              </label>
            );
          })}
        </div>
      ) : null}
    </section>
  );
});

export function FacetSidebar({
  facets,
  filters,
  library,
  total,
  onToggleFacet,
  onPressureChange,
  onSavedModeChange,
  onClear,
}: FacetSidebarProps) {
  const normalized = useMemo(
    () => Object.fromEntries(FACET_SPECS.map((spec) => [spec.field, normalizeFacet(facets, spec)])),
    [facets],
  );
  const collectionNames = Object.keys(library.collections).sort((a, b) => a.localeCompare(b));

  return (
    <aside className="facet-sidebar" aria-label="Plot filters">
      <section className="saved-filters">
        <button
          className={filters.savedMode === "all" ? "saved-filter active" : "saved-filter"}
          type="button"
          onClick={() => onSavedModeChange("all")}
        >
          <span>All indexed plots</span><strong>{total.toLocaleString()}</strong>
        </button>
        <button
          className={filters.savedMode === "favorites" ? "saved-filter active" : "saved-filter"}
          type="button"
          onClick={() => onSavedModeChange("favorites")}
        >
          <span><Star size={14} /> Favorites</span><strong>{library.favorites.length}</strong>
        </button>
        {collectionNames.length ? (
          <label className="collection-filter">
            <span><Folder size={14} /> Collection</span>
            <select
              value={filters.savedMode === "collection" ? filters.collectionName : ""}
              onChange={(event) =>
                event.target.value ? onSavedModeChange("collection", event.target.value) : onSavedModeChange("all")
              }
            >
              <option value="">Choose collection</option>
              {collectionNames.map((name) => (
                <option key={name} value={name}>{name} ({library.collections[name].length})</option>
              ))}
            </select>
          </label>
        ) : null}
      </section>

      {FACET_SPECS.map((spec, index) => (
        <FacetGroup
          key={spec.field}
          spec={spec}
          entries={normalized[spec.field]}
          selected={filters.selected[spec.field] ?? []}
          onToggle={onToggleFacet}
          initiallyOpen={index < 7}
        />
      ))}

      <section className="facet-group pressure-facet">
        <div className="facet-heading static"><span>Pressure (GPa)</span></div>
        <div className="pressure-row">
          <label>
            <span>Minimum</span>
            <input
              type="number"
              inputMode="decimal"
              value={filters.pressureMin}
              onChange={(event) => onPressureChange(event.target.value, filters.pressureMax)}
              placeholder="0.00"
            />
          </label>
          <span aria-hidden="true">–</span>
          <label>
            <span>Maximum</span>
            <input
              type="number"
              inputMode="decimal"
              value={filters.pressureMax}
              onChange={(event) => onPressureChange(filters.pressureMin, event.target.value)}
              placeholder="60.00"
            />
          </label>
        </div>
      </section>

      <button className="clear-filters" type="button" onClick={onClear}>Clear all filters</button>
    </aside>
  );
}
