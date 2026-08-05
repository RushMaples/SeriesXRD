import { Grid2X2, List, LoaderCircle, RefreshCw } from "lucide-react";
import {
  startTransition,
  useCallback,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { fetchMeta, fetchPlot, fetchPlots, type IndexMeta } from "./api";
import { CompareTray } from "./components/CompareTray";
import { CompareView } from "./components/CompareView";
import { DetailPanel } from "./components/DetailPanel";
import { FacetSidebar } from "./components/FacetSidebar";
import { Header } from "./components/Header";
import { PlotCard } from "./components/PlotCard";
import { PlotTable } from "./components/PlotTable";
import {
  addToCollection,
  loadSavedLibrary,
  persistSavedLibrary,
  removeFromCollection,
  toggleFavorite,
} from "./storage";
import type { PlotPage, PlotRecord, SavedLibrary, UrlState } from "./types";
import { DEFAULT_FILTERS, readUrlState, writeUrlState } from "./urlState";

const EMPTY_PAGE: PlotPage = {
  items: [],
  total: 0,
  indexTotal: 0,
  page: 1,
  pageSize: 24,
  facets: {},
  updatedAt: null,
};

const SORT_OPTIONS = [
  ["pressure_desc", "Pressure (desc)"],
  ["pressure_asc", "Pressure (asc)"],
  ["peak_asc", "Local peak (asc)"],
  ["two_theta_asc", "2θ center (asc)"],
  ["title_asc", "Title (A–Z)"],
] as const;

function mergeRecordCache(cache: Map<string, PlotRecord>, records: PlotRecord[]): Map<string, PlotRecord> {
  for (const record of records) cache.set(record.plot_id, record);
  return cache;
}

function App() {
  const [state, setState] = useState<UrlState>(() => readUrlState());
  const [page, setPage] = useState<PlotPage>(EMPTY_PAGE);
  const [indexMeta, setIndexMeta] = useState<IndexMeta | null>(null);
  const [library, setLibrary] = useState<SavedLibrary>(() => loadSavedLibrary());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [cacheVersion, setCacheVersion] = useState(0);
  const recordCache = useRef(new Map<string, PlotRecord>());
  const deferredQuery = useDeferredValue(state.filters.query);

  useEffect(() => {
    writeUrlState(state);
  }, [state]);

  useEffect(() => {
    const onPopState = () => setState(readUrlState());
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  useEffect(() => {
    const narrowViewport = window.matchMedia("(max-width: 1120px)");
    const closeDockedDetail = () => {
      if (narrowViewport.matches) {
        setState((current) => current.selectedPlotId ? { ...current, selectedPlotId: null } : current);
      }
    };
    closeDockedDetail();
    narrowViewport.addEventListener("change", closeDockedDetail);
    return () => narrowViewport.removeEventListener("change", closeDockedDetail);
  }, []);

  useEffect(() => {
    persistSavedLibrary(library);
  }, [library]);

  useEffect(() => {
    const controller = new AbortController();
    fetchMeta(controller.signal).then(setIndexMeta).catch(() => undefined);
    return () => controller.abort();
  }, []);

  const savedIds = useMemo(() => {
    if (state.filters.savedMode === "favorites") return library.favorites;
    if (state.filters.savedMode === "collection") return library.collections[state.filters.collectionName] ?? [];
    return undefined;
  }, [library, state.filters.collectionName, state.filters.savedMode]);

  const requestKey = JSON.stringify({
    query: deferredQuery,
    selected: state.filters.selected,
    pressureMin: state.filters.pressureMin,
    pressureMax: state.filters.pressureMax,
    sort: state.sort,
    page: state.page,
    pageSize: state.pageSize,
    ids: savedIds,
  });

  const loadPage = useCallback(() => {
    const controller = new AbortController();
    if (savedIds && savedIds.length === 0) {
      setPage((current) => ({ ...current, items: [], total: 0, page: 1 }));
      setLoading(false);
      setError(null);
      return () => controller.abort();
    }
    setLoading(true);
    setError(null);
    fetchPlots(
      {
        query: deferredQuery,
        selected: state.filters.selected,
        pressureMin: state.filters.pressureMin,
        pressureMax: state.filters.pressureMax,
        sort: state.sort,
        page: state.page,
        pageSize: state.pageSize,
        ids: savedIds,
      },
      controller.signal,
    )
      .then((result) => {
        mergeRecordCache(recordCache.current, result.items);
        setPage(result);
        setCacheVersion((version) => version + 1);
        if (result.items[0]) {
          setState((current) => {
            const selectedIsVisible = result.items.some((record) => record.plot_id === current.selectedPlotId);
            if (selectedIsVisible) return current;
            if (window.matchMedia("(max-width: 1120px)").matches) {
              return current.selectedPlotId ? { ...current, selectedPlotId: null } : current;
            }
            return { ...current, selectedPlotId: result.items[0].plot_id };
          });
        }
      })
      .catch((reason: unknown) => {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        setError(reason instanceof Error ? reason.message : "Unable to read the plot index.");
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  // requestKey deliberately captures the complete server query as one primitive dependency.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [requestKey]);

  useEffect(() => loadPage(), [loadPage]);

  useEffect(() => {
    const ids = [state.selectedPlotId, ...state.compareIds]
      .filter((value): value is string => Boolean(value))
      .filter((plotId) => !recordCache.current.has(plotId));
    if (!ids.length) return;
    const controller = new AbortController();
    Promise.all(ids.map((plotId) => fetchPlot(plotId, controller.signal)))
      .then((records) => {
        mergeRecordCache(recordCache.current, records.filter((record): record is PlotRecord => Boolean(record)));
        setCacheVersion((version) => version + 1);
      })
      .catch((reason: unknown) => {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        setNotice(reason instanceof Error ? reason.message : "Some selected plots could not be loaded.");
      });
    return () => controller.abort();
  }, [state.compareIds, state.selectedPlotId]);

  const selectedRecord = state.selectedPlotId ? recordCache.current.get(state.selectedPlotId) ?? null : null;
  const compareRecords = useMemo(
    () => state.compareIds.map((plotId) => recordCache.current.get(plotId)).filter((record): record is PlotRecord => Boolean(record)),
    // cacheVersion signals in-place Map updates without cloning the entire record index.
    [cacheVersion, state.compareIds],
  );
  const favoriteIds = useMemo(() => new Set(library.favorites), [library.favorites]);

  const updateFilters = useCallback((updater: (filters: UrlState["filters"]) => UrlState["filters"]) => {
    startTransition(() => setState((current) => ({ ...current, filters: updater(current.filters), page: 1 })));
  }, []);

  const toggleFacet = useCallback((field: string, value: string) => {
    updateFilters((filters) => {
      const values = new Set(filters.selected[field] ?? []);
      values.has(value) ? values.delete(value) : values.add(value);
      const selected = { ...filters.selected };
      values.size ? (selected[field] = [...values]) : delete selected[field];
      return { ...filters, selected };
    });
  }, [updateFilters]);

  const toggleCompareRecord = useCallback((record: PlotRecord) => {
    recordCache.current.set(record.plot_id, record);
    setCacheVersion((version) => version + 1);
    setState((current) => {
      if (current.compareIds.includes(record.plot_id)) {
        return { ...current, compareIds: current.compareIds.filter((plotId) => plotId !== record.plot_id) };
      }
      if (current.compareIds.length >= 4) {
        setNotice("The comparison workspace accepts up to four plots.");
        return current;
      }
      return { ...current, compareIds: [...current.compareIds, record.plot_id] };
    });
  }, []);

  const removeCompare = useCallback((plotId: string) => {
    setState((current) => ({ ...current, compareIds: current.compareIds.filter((id) => id !== plotId) }));
  }, []);

  const enterCompare = useCallback(() => {
    if (state.compareIds.length < 2) return;
    window.history.pushState(null, "", window.location.href);
    setState((current) => ({ ...current, mode: "compare" }));
  }, [state.compareIds.length]);

  const setLibraryAndPersist = useCallback((updater: (current: SavedLibrary) => SavedLibrary) => {
    setLibrary((current) => updater(current));
  }, []);

  const queryRelated = useCallback((record: PlotRecord) => {
    const term = record.anchor_uid || record.frame_id || record.plot_id;
    setState((current) => ({
      ...current,
      mode: "library",
      page: 1,
      selectedPlotId: record.plot_id,
      filters: { ...current.filters, query: String(term), savedMode: "all", collectionName: "" },
    }));
  }, []);

  const totalPages = Math.max(1, Math.ceil(page.total / state.pageSize));

  return (
    <div className={`app-shell${state.compareIds.length && state.mode === "library" ? " has-tray" : ""}`}>
      <Header
        query={state.filters.query}
        onQueryChange={(query) => updateFilters((filters) => ({ ...filters, query }))}
        total={Number(indexMeta?.summary?.plot_records ?? page.indexTotal ?? page.total)}
        updatedAt={indexMeta?.generated_at ?? page.updatedAt}
        loading={loading}
      />

      {state.mode === "compare" ? (
        <CompareView
          records={compareRecords}
          onBack={() => setState((current) => ({ ...current, mode: "library" }))}
          onRemove={removeCompare}
          onClear={() => setState((current) => ({ ...current, compareIds: [], mode: "library" }))}
        />
      ) : (
        <>
          <div className="library-layout">
            <FacetSidebar
              facets={page.facets}
              filters={state.filters}
              library={library}
              total={Number(indexMeta?.summary?.plot_records ?? page.indexTotal ?? page.total)}
              onToggleFacet={toggleFacet}
              onPressureChange={(pressureMin, pressureMax) => updateFilters((filters) => ({ ...filters, pressureMin, pressureMax }))}
              onSavedModeChange={(savedMode, collectionName = "") => updateFilters((filters) => ({ ...filters, savedMode, collectionName }))}
              onClear={() => setState((current) => ({ ...current, filters: { ...DEFAULT_FILTERS, selected: {} }, page: 1 }))}
            />

            <main className="results-panel">
              <div className="results-toolbar">
                <div><h1>{page.total.toLocaleString()} matching plot{page.total === 1 ? "" : "s"}</h1>{deferredQuery ? <span>for “{deferredQuery}”</span> : null}</div>
                <div className="results-controls">
                  <label>Sort by
                    <select value={state.sort} onChange={(event) => setState((current) => ({ ...current, sort: event.target.value, page: 1 }))}>
                      {SORT_OPTIONS.map(([value, label]) => <option value={value} key={value}>{label}</option>)}
                    </select>
                  </label>
                  <div className="view-toggle" role="group" aria-label="Results view">
                    <button type="button" className={state.view === "grid" ? "active" : ""} onClick={() => setState((current) => ({ ...current, view: "grid" }))} aria-label="Grid view"><Grid2X2 size={18} /></button>
                    <button type="button" className={state.view === "table" ? "active" : ""} onClick={() => setState((current) => ({ ...current, view: "table" }))} aria-label="Table view"><List size={19} /></button>
                  </div>
                </div>
              </div>

              {notice ? <div className="notice-banner" role="status">{notice}<button type="button" onClick={() => setNotice(null)}>Dismiss</button></div> : null}
              {error ? (
                <div className="error-state" role="alert"><RefreshCw size={28} /><h2>Plot index unavailable</h2><p>{error}</p><button className="secondary-button" type="button" onClick={() => loadPage()}>Try again</button></div>
              ) : loading && !page.items.length ? (
                <div className="loading-state"><LoaderCircle className="spin" size={28} /><span>Reading the classified plot index…</span></div>
              ) : !page.items.length ? (
                <div className="empty-state"><h2>No plots match these filters</h2><p>Clear one or more scientific facets, or search a different anchor or window.</p><button className="secondary-button" type="button" onClick={() => setState((current) => ({ ...current, filters: { ...DEFAULT_FILTERS, selected: {} }, page: 1 }))}>Clear filters</button></div>
              ) : state.view === "grid" ? (
                <div className={loading ? "plot-grid refreshing" : "plot-grid"}>
                  {page.items.map((record) => (
                    <PlotCard
                      key={record.plot_id}
                      record={record}
                      selected={state.selectedPlotId === record.plot_id}
                      compared={state.compareIds.includes(record.plot_id)}
                      favorite={favoriteIds.has(record.plot_id)}
                      onSelect={(selected) => {
                        recordCache.current.set(selected.plot_id, selected);
                        setState((current) => ({ ...current, selectedPlotId: selected.plot_id }));
                      }}
                      onToggleCompare={toggleCompareRecord}
                      onToggleFavorite={(plotId) => setLibraryAndPersist((current) => toggleFavorite(current, plotId))}
                    />
                  ))}
                </div>
              ) : (
                <PlotTable
                  records={page.items}
                  selectedId={state.selectedPlotId}
                  compareIds={state.compareIds}
                  favoriteIds={favoriteIds}
                  onSelect={(selected) => setState((current) => ({ ...current, selectedPlotId: selected.plot_id }))}
                  onToggleCompare={toggleCompareRecord}
                  onToggleFavorite={(plotId) => setLibraryAndPersist((current) => toggleFavorite(current, plotId))}
                />
              )}

              {!error && page.total > 0 ? (
                <nav className="pagination" aria-label="Plot result pages">
                  <button type="button" disabled={state.page <= 1} onClick={() => setState((current) => ({ ...current, page: current.page - 1 }))}>Previous</button>
                  <span>Page <strong>{state.page.toLocaleString()}</strong> of {totalPages.toLocaleString()}</span>
                  <button type="button" disabled={state.page >= totalPages} onClick={() => setState((current) => ({ ...current, page: current.page + 1 }))}>Next</button>
                  <label>Rows
                    <select value={state.pageSize} onChange={(event) => setState((current) => ({ ...current, pageSize: Number(event.target.value), page: 1 }))}>
                      {[12, 24, 48, 96].map((value) => <option value={value} key={value}>{value}</option>)}
                    </select>
                  </label>
                </nav>
              ) : null}
            </main>

            <DetailPanel
              record={selectedRecord}
              library={library}
              onClose={() => setState((current) => ({ ...current, selectedPlotId: null }))}
              onToggleFavorite={(plotId) => setLibraryAndPersist((current) => toggleFavorite(current, plotId))}
              onAddToCollection={(name, plotId) => setLibraryAndPersist((current) => addToCollection(current, name, plotId))}
              onRemoveFromCollection={(name, plotId) => setLibraryAndPersist((current) => removeFromCollection(current, name, plotId))}
              onFindRelated={queryRelated}
            />
          </div>

          <CompareTray
            records={compareRecords}
            onRemove={removeCompare}
            onClear={() => setState((current) => ({ ...current, compareIds: [] }))}
            onCompare={enterCompare}
          />
        </>
      )}
    </div>
  );
}

export default App;
