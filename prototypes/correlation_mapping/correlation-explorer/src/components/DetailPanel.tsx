import {
  AlertTriangle,
  Braces,
  ExternalLink,
  FileImage,
  FileSpreadsheet,
  FolderPlus,
  Search,
  Star,
  X,
} from "lucide-react";
import { useMemo, useState } from "react";

import { fileUrl, imageUrl } from "../api";
import {
  altText,
  compactTitle,
  displayDomainLabel,
  familyLabel,
  formatNumber,
  formatPressure,
  humanize,
  isWaterfall,
  localPeak,
  runLabel,
  sampleLabel,
  transformLabel,
  validationTone,
  visualizationLabel,
  windowLabel,
} from "../format";
import type { PlotRecord, SavedLibrary } from "../types";
import { PlotImage } from "./PlotImage";
import { SemanticLegend } from "./SemanticLegend";

type DetailPanelProps = {
  record: PlotRecord | null;
  library: SavedLibrary;
  onClose: () => void;
  onToggleFavorite: (plotId: string) => void;
  onAddToCollection: (name: string, plotId: string) => void;
  onRemoveFromCollection: (name: string, plotId: string) => void;
  onFindRelated: (record: PlotRecord) => void;
};

type DetailTab = "preview" | "metadata" | "files";

function flattenCompanions(record: PlotRecord): Array<{ kind: string; path: string; url?: string }> {
  const files: Array<{ kind: string; path: string; url?: string }> = [];
  if (record.csv_path) files.push({ kind: "CSV", path: record.csv_path });
  if (record.json_path) files.push({ kind: "JSON", path: record.json_path });
  if (Array.isArray(record.companions)) {
    for (const item of record.companions) {
      if (typeof item === "string") files.push({ kind: item.split(".").pop()?.toUpperCase() ?? "FILE", path: item });
      else files.push({ kind: item.kind.toUpperCase(), path: item.path, url: item.url });
    }
  } else if (record.companions && typeof record.companions === "object") {
    for (const [kind, value] of Object.entries(record.companions)) {
      for (const path of Array.isArray(value) ? value : value ? [value] : []) files.push({ kind: kind.toUpperCase(), path });
    }
  }
  const unique = new Map(files.map((file) => [file.path, file]));
  return [...unique.values()];
}

function MetadataRow({ label, value, highlighted = false }: { label: string; value: React.ReactNode; highlighted?: boolean }) {
  return (
    <div className={highlighted ? "metadata-row highlighted" : "metadata-row"}>
      <dt>{label}</dt><dd>{value}</dd>
    </div>
  );
}

export function DetailPanel({
  record,
  library,
  onClose,
  onToggleFavorite,
  onAddToCollection,
  onRemoveFromCollection,
  onFindRelated,
}: DetailPanelProps) {
  const [tab, setTab] = useState<DetailTab>("preview");
  const [newCollection, setNewCollection] = useState("");
  const companions = useMemo(() => (record ? flattenCompanions(record) : []), [record]);

  if (!record) {
    return (
      <aside className="detail-panel empty-detail">
        <div className="empty-detail-mark" aria-hidden="true">⌖</div>
        <h2>Select a plot</h2>
        <p>Choose a heatmap, matrix, or waterfall to inspect its scientific metadata and companion files.</p>
        <SemanticLegend compact />
      </aside>
    );
  }

  const favorite = library.favorites.includes(record.plot_id);
  const memberCollections = Object.entries(library.collections)
    .filter(([, ids]) => ids.includes(record.plot_id))
    .map(([name]) => name);
  const aliases = record.aliases ?? [];
  const directionality = record.correlation_family === "roi_area" ? "Anchor → targets (directional support)" : "As defined by companion matrix";

  return (
    <aside className="detail-panel" aria-label="Selected plot details">
      <div className="detail-heading">
        <div>
          <span>Selected plot</span>
          <h2>{compactTitle(record)}</h2>
        </div>
        <div className="detail-actions">
          <button
            className={favorite ? "icon-button active" : "icon-button"}
            type="button"
            onClick={() => onToggleFavorite(record.plot_id)}
            aria-label={favorite ? "Remove favorite" : "Add favorite"}
          >
            <Star size={18} fill={favorite ? "currentColor" : "none"} />
          </button>
          <button className="icon-button" type="button" onClick={onClose} aria-label="Close details"><X size={20} /></button>
        </div>
      </div>

      <div className="detail-tabs" role="tablist" aria-label="Plot detail sections">
        {(["preview", "metadata", "files"] as DetailTab[]).map((value) => (
          <button
            key={value}
            className={tab === value ? "active" : ""}
            type="button"
            role="tab"
            aria-selected={tab === value}
            onClick={() => setTab(value)}
          >
            {humanize(value)}
          </button>
        ))}
      </div>

      <div className="detail-scroll">
        {tab === "preview" ? (
          <div className="detail-preview">
            <a href={imageUrl(record)} target="_blank" rel="noreferrer" title="Open full-resolution image">
              <PlotImage src={imageUrl(record)} alt={altText(record)} loading="eager" />
              <span className="open-full"><ExternalLink size={14} /> Full resolution</span>
            </a>
            <div className="detail-summary">
              <div><span>Correlation calculation</span><strong>{transformLabel(record.correlation_transform)}</strong></div>
              <div><span>Curve displayed</span><strong>{isWaterfall(record) ? displayDomainLabel(record.display_profile_domain) : "N/A"}</strong></div>
              <div><span>Correlation family</span><strong>{familyLabel(record.correlation_family)}</strong></div>
              <div><span>Validation</span><strong><span className={`status-badge ${validationTone(record.validation_status)}`}>{record.validation_status}</span></strong></div>
            </div>
            <SemanticLegend compact />
          </div>
        ) : null}

        {tab === "metadata" ? (
          <>
            {record.classification_warning ? (
              <div className="classification-warning"><AlertTriangle size={17} /><span>{record.classification_warning}</span></div>
            ) : null}
            <dl className="metadata-list">
              <MetadataRow label="Sample type" value={sampleLabel(record.sample)} />
              <MetadataRow label="Result lifecycle" value={humanize(record.result_status)} />
              <MetadataRow label="Validation status" value={<span className={`status-badge ${validationTone(record.validation_status)}`}>{record.validation_status}</span>} />
              <MetadataRow label="Anchor pressure" value={formatPressure(record.anchor_pressure_gpa)} />
              <MetadataRow label="Local peak number" value={localPeak(record) ?? "N/A"} highlighted />
              <MetadataRow label="2θ peak center" value={record.anchor_two_theta_deg == null ? "N/A" : `${formatNumber(record.anchor_two_theta_deg)}°`} />
              <MetadataRow label="q" value={formatNumber(record.anchor_q, 6)} />
              <MetadataRow label="q-width" value={formatNumber(record.anchor_q_width, 6)} />
              <MetadataRow label="q-width factor" value={formatNumber(record.half_width_factor, 3)} />
              <MetadataRow label="Correlation calculation" value={transformLabel(record.correlation_transform)} highlighted />
              <MetadataRow label="Correlation family" value={familyLabel(record.correlation_family)} />
              <MetadataRow label="Directionality" value={directionality} />
              <MetadataRow label="Visualization" value={visualizationLabel(record.visualization_type)} />
              <MetadataRow label="Curve displayed" value={isWaterfall(record) ? displayDomainLabel(record.display_profile_domain) : "N/A"} highlighted={isWaterfall(record)} />
              {isWaterfall(record) ? <MetadataRow label="Curve source" value={humanize(record.display_profile_source)} /> : null}
              {isWaterfall(record) && record.display_profile_construction
                ? <MetadataRow label="Curve construction" value={record.display_profile_construction} />
                : null}
              <MetadataRow label="Window" value={windowLabel(record)} />
              <MetadataRow label="Frame scope" value={humanize(record.frame_scope)} />
              <MetadataRow label="Signal channel" value={humanize(record.signal_channel)} />
              <MetadataRow label="Algorithm" value={humanize(record.algorithm_variant)} />
              <MetadataRow label="Aggregation" value={humanize(record.aggregation_level)} />
              <MetadataRow label="Result run" value={runLabel(record)} />
              <MetadataRow label="Classification source" value={record.classification_source || "N/A"} />
            </dl>
            <button className="secondary-button full-width" type="button" onClick={() => onFindRelated(record)}>
              <Search size={16} /> Find related plots
            </button>
          </>
        ) : null}

        {tab === "files" ? (
          <div className="files-tab">
            {record.image_path ? (
              <a className="file-row" href={imageUrl(record)} target="_blank" rel="noreferrer">
                <FileImage size={18} /><span><strong>Image (PNG)</strong><small>{record.image_path}</small></span><ExternalLink size={15} />
              </a>
            ) : null}
            {companions.map((file) => (
              <a className="file-row" key={file.path} href={file.url || fileUrl(file.path)} target="_blank" rel="noreferrer">
                {file.kind.includes("JSON") ? <Braces size={18} /> : <FileSpreadsheet size={18} />}
                <span><strong>Companion {file.kind}</strong><small>{file.path}</small></span><ExternalLink size={15} />
              </a>
            ))}
            {!record.image_path && companions.length === 0 ? <p className="empty-copy">No readable companion paths were indexed.</p> : null}

            <section className="file-metadata-block">
              <h3>Asset identity</h3>
              <code>{record.asset_id || record.sha256 || "No asset id"}</code>
              <p>{aliases.length} alias path{aliases.length === 1 ? "" : "s"}</p>
              {aliases.length ? (
                <details><summary>Show aliases</summary>{aliases.map((path) => <code key={path}>{path}</code>)}</details>
              ) : null}
            </section>

            <section className="collections-editor">
              <h3><FolderPlus size={16} /> Collections</h3>
              {memberCollections.length ? (
                <div className="collection-chips">
                  {memberCollections.map((name) => (
                    <button key={name} type="button" onClick={() => onRemoveFromCollection(name, record.plot_id)} title="Remove from collection">
                      {name}<X size={12} />
                    </button>
                  ))}
                </div>
              ) : <p>This plot is not in a collection.</p>}
              <form
                onSubmit={(event) => {
                  event.preventDefault();
                  if (newCollection.trim()) {
                    onAddToCollection(newCollection, record.plot_id);
                    setNewCollection("");
                  }
                }}
              >
                <input value={newCollection} onChange={(event) => setNewCollection(event.target.value)} placeholder="New or existing collection" />
                <button type="submit" className="secondary-button">Add</button>
              </form>
            </section>
          </div>
        ) : null}
      </div>
    </aside>
  );
}
