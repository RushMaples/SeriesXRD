import {
  ArrowLeft,
  Braces,
  FileSpreadsheet,
  Info,
  Maximize2,
  Minus,
  Plus,
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
  sameSemanticCoordinates,
  sampleLabel,
  transformLabel,
  validationTone,
  visualizationLabel,
  windowLabel,
} from "../format";
import type { PlotRecord } from "../types";
import { PlotImage } from "./PlotImage";

type CompareViewProps = {
  records: PlotRecord[];
  onBack: () => void;
  onRemove: (plotId: string) => void;
  onClear: () => void;
};

type CompareRow = { label: string; value: (record: PlotRecord) => string; anchor?: boolean };

const COMPARE_ROWS: CompareRow[] = [
  { label: "Sample", value: (record) => sampleLabel(record.sample) },
  { label: "Correlation family", value: (record) => familyLabel(record.correlation_family) },
  { label: "Correlation calculation", value: (record) => transformLabel(record.correlation_transform) },
  { label: "Curve displayed", value: (record) => isWaterfall(record) ? displayDomainLabel(record.display_profile_domain) : "N/A" },
  { label: "Curve source", value: (record) => isWaterfall(record) ? humanize(record.display_profile_source) : "N/A" },
  { label: "Anchor pressure", value: (record) => formatPressure(record.anchor_pressure_gpa), anchor: true },
  { label: "Local peak number", value: (record) => String(localPeak(record) ?? "N/A"), anchor: true },
  { label: "2θ center", value: (record) => record.anchor_two_theta_deg == null ? "N/A" : `${formatNumber(record.anchor_two_theta_deg)}°`, anchor: true },
  { label: "q-width factor", value: (record) => formatNumber(record.half_width_factor, 3), anchor: true },
  { label: "Window", value: (record) => windowLabel(record) },
  { label: "Signal channel", value: (record) => humanize(record.signal_channel) },
  { label: "Algorithm", value: (record) => humanize(record.algorithm_variant) },
  { label: "Visualization", value: (record) => visualizationLabel(record.visualization_type) },
  { label: "Result run", value: (record) => runLabel(record) },
  { label: "Validation", value: (record) => record.validation_status },
];

function firstCompanion(record: PlotRecord, kind: "csv" | "json"): string | null {
  const direct = kind === "csv" ? record.csv_path : record.json_path;
  if (!record.companions || Array.isArray(record.companions)) {
    const match = Array.isArray(record.companions)
      ? record.companions.find((item) =>
          typeof item === "string"
            ? item.toLowerCase().endsWith(`.${kind}`)
            : item.path.toLowerCase().endsWith(`.${kind}`) || item.kind.toLowerCase().includes(kind),
        )
      : null;
    if (typeof match === "string") return fileUrl(match);
    if (match) return match.url || fileUrl(match.path);
    return direct ? fileUrl(direct) : null;
  }
  const value = record.companions[kind];
  const path = Array.isArray(value) ? value[0] ?? null : value ?? null;
  return path ? fileUrl(path) : direct ? fileUrl(direct) : null;
}

function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (checked: boolean) => void; label: string }) {
  return (
    <label className="toggle-row">
      <span>{label}</span>
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <span className="toggle-track" aria-hidden="true"><span /></span>
    </label>
  );
}

function CompareCard({
  record,
  index,
  zoom,
  onZoom,
  onRemove,
}: {
  record: PlotRecord;
  index: number;
  zoom: number;
  onZoom: (value: number) => void;
  onRemove: () => void;
}) {
  return (
    <article className="compare-card">
      <header>
        <div><span className="compare-number">{index + 1}</span><strong>{compactTitle(record)}</strong></div>
        <button className="icon-button" type="button" onClick={onRemove} aria-label="Remove comparison plot"><X size={18} /></button>
        <p>{sampleLabel(record.sample)} {familyLabel(record.correlation_family)}, {transformLabel(record.correlation_transform)}</p>
        <p>Anchor {formatPressure(record.anchor_pressure_gpa)} · Local peak {localPeak(record) ?? "N/A"}</p>
      </header>
      <div className="compare-image-stage">
        <div className="compare-image-scroll">
          <PlotImage
            src={imageUrl(record)}
            alt={altText(record)}
            loading="eager"
            className="compare-image"
          />
          <style>{`.compare-card:nth-of-type(${index + 1}) .compare-image{width:${zoom}%}`}</style>
        </div>
        <div className="zoom-controls">
          <select value={zoom} onChange={(event) => onZoom(Number(event.target.value))} aria-label="Zoom percentage">
            {[50, 75, 100, 125, 150, 200].map((value) => <option value={value} key={value}>{value}%</option>)}
          </select>
          <button type="button" onClick={() => onZoom(Math.max(50, zoom - 25))} aria-label="Zoom out"><Minus size={16} /></button>
          <input type="range" min="50" max="200" step="5" value={zoom} onChange={(event) => onZoom(Number(event.target.value))} aria-label="Image zoom" />
          <button type="button" onClick={() => onZoom(Math.min(200, zoom + 25))} aria-label="Zoom in"><Plus size={16} /></button>
          <a href={imageUrl(record)} target="_blank" rel="noreferrer" aria-label="Open full-resolution plot"><Maximize2 size={17} /></a>
        </div>
      </div>
      <footer><span>Result run</span><strong title={runLabel(record)}>{runLabel(record)}</strong><span>Validation</span><span className={`status-badge ${validationTone(record.validation_status)}`}>{record.validation_status}</span></footer>
    </article>
  );
}

export function CompareView({ records, onBack, onRemove, onClear }: CompareViewProps) {
  const [syncZoom, setSyncZoom] = useState(true);
  const [linkAnchor, setLinkAnchor] = useState(true);
  const [showOnlyDifferences, setShowOnlyDifferences] = useState(false);
  const [commonZoom, setCommonZoom] = useState(100);
  const [individualZoom, setIndividualZoom] = useState<Record<string, number>>({});
  const coordinateCompatible = sameSemanticCoordinates(records);
  const visibleRows = useMemo(
    () => COMPARE_ROWS.filter((row) => {
      if (!linkAnchor && row.anchor) return false;
      if (!showOnlyDifferences) return true;
      return new Set(records.map(row.value)).size > 1;
    }),
    [linkAnchor, records, showOnlyDifferences],
  );

  const setZoom = (record: PlotRecord, value: number) => {
    if (syncZoom) setCommonZoom(value);
    else setIndividualZoom((current) => ({ ...current, [record.plot_id]: value }));
  };

  return (
    <main className="compare-view">
      <div className="compare-toolbar">
        <button className="secondary-button" type="button" onClick={onBack}><ArrowLeft size={17} /> Back to Library</button>
        <h1>Compare plots</h1><span className="count-pill">{records.length} selected</span>
        <div className="toolbar-spacer" />
        <button className="secondary-button" type="button" onClick={onClear}>Remove all</button>
        <button className="primary-button" type="button" onClick={onClear}>Clear comparison</button>
      </div>

      {records.length < 2 ? (
        <section className="compare-empty"><h2>Select at least two plots</h2><p>Return to the library and add 2–4 scientifically comparable plots.</p><button className="primary-button" type="button" onClick={onBack}>Back to Library</button></section>
      ) : (
        <>
          <section className={`compare-cards count-${records.length}`}>
            {records.map((record, index) => (
              <CompareCard
                key={record.plot_id}
                record={record}
                index={index}
                zoom={syncZoom ? commonZoom : individualZoom[record.plot_id] ?? 100}
                onZoom={(value) => setZoom(record, value)}
                onRemove={() => onRemove(record.plot_id)}
              />
            ))}
          </section>

          <section className="comparison-lower">
            <div className="metadata-comparison-wrap">
              <table className="metadata-comparison">
                <tbody>
                  {visibleRows.map((row) => {
                    const values = records.map(row.value);
                    const differs = new Set(values).size > 1;
                    return (
                      <tr key={row.label}>
                        <th>{row.label}</th>
                        {values.map((value, index) => (
                          <td className={differs ? "different" : ""} key={`${records[index].plot_id}-${row.label}`}>
                            {row.label === "Validation" ? <span className={`status-badge ${validationTone(value)}`}>{value}</span> : value}
                          </td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <aside className="compare-controls">
              <section>
                <h2>Controls & actions</h2>
                <Toggle checked={syncZoom} onChange={setSyncZoom} label="Sync zoom (images only)" />
                <Toggle checked={linkAnchor} onChange={setLinkAnchor} label="Link anchor metadata" />
                <Toggle checked={showOnlyDifferences} onChange={setShowOnlyDifferences} label="Show only differences" />
                <div className="companion-actions">
                  {firstCompanion(records[0], "csv") ? <a className="secondary-button" href={firstCompanion(records[0], "csv")!} target="_blank" rel="noreferrer"><FileSpreadsheet size={16} /> Open CSV</a> : null}
                  {firstCompanion(records[0], "json") ? <a className="secondary-button" href={firstCompanion(records[0], "json")!} target="_blank" rel="noreferrer"><Braces size={16} /> Open JSON</a> : null}
                </div>
              </section>
              <section className="semantics-notice">
                <h2><Info size={17} /> Semantics notice</h2>
                <p>{coordinateCompatible
                  ? "These plots share a semantic coordinate family; image zoom can sync. Exact values remain defined by each companion matrix."
                  : "Different visualization semantics — image zoom can sync; matrix coordinates cannot. Across-frame and within-frame blanks must not be compared as zero."}</p>
              </section>
            </aside>
          </section>
        </>
      )}
    </main>
  );
}
