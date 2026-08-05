import { Search, Settings, X } from "lucide-react";

type HeaderProps = {
  query: string;
  onQueryChange: (value: string) => void;
  total: number;
  updatedAt?: string | null;
  loading?: boolean;
};

function AtlasMark() {
  return (
    <svg className="atlas-mark" viewBox="0 0 36 36" aria-hidden="true">
      <path d="M5 18 14 8l10 4 7-7M14 8l3 20m7-16 5 14M5 18l12 10 12-2" />
      <circle cx="5" cy="18" r="2.2" />
      <circle cx="14" cy="8" r="2.2" />
      <circle cx="24" cy="12" r="2.2" />
      <circle cx="31" cy="5" r="2.2" />
      <circle cx="17" cy="28" r="2.2" />
      <circle cx="29" cy="26" r="2.2" />
    </svg>
  );
}

export function Header({ query, onQueryChange, total, updatedAt, loading = false }: HeaderProps) {
  const updated = updatedAt
    ? new Date(updatedAt).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })
    : "Index status unavailable";
  return (
    <header className="app-header">
      <div className="brand" aria-label="XRD Correlation Atlas">
        <AtlasMark />
        <span>XRD Correlation Atlas</span>
      </div>
      <label className="global-search">
        <Search size={21} aria-hidden="true" />
        <span className="sr-only">Search plots</span>
        <input
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          placeholder="Search pressure, anchor peak, q-width, window, or method"
          spellCheck={false}
        />
        {query ? (
          <button className="icon-button" type="button" onClick={() => onQueryChange("")} aria-label="Clear search">
            <X size={20} />
          </button>
        ) : null}
      </label>
      <div className="index-summary" aria-live="polite">
        <strong>{loading ? "Updating…" : `Index ${total.toLocaleString()} plots`}</strong>
        <span>Last updated: {updated}</span>
      </div>
      <a className="icon-button header-settings" href="/api/audit" target="_blank" rel="noreferrer" aria-label="Open index audit" title="Open index audit">
        <Settings size={23} />
      </a>
    </header>
  );
}
