import { ArrowRight, X } from "lucide-react";

import { imageUrl } from "../api";
import { altText, compactTitle, familyLabel, formatPressure, transformLabel } from "../format";
import type { PlotRecord } from "../types";
import { PlotImage } from "./PlotImage";

type CompareTrayProps = {
  records: PlotRecord[];
  onRemove: (plotId: string) => void;
  onClear: () => void;
  onCompare: () => void;
};

export function CompareTray({ records, onRemove, onClear, onCompare }: CompareTrayProps) {
  if (!records.length) return null;
  return (
    <aside className="compare-tray" aria-label={`Compare tray with ${records.length} plots`}>
      <div className="tray-label"><strong>Compare tray ({records.length})</strong><span>Choose 2–4 plots</span></div>
      <div className="tray-records">
        {records.map((record) => (
          <article className="tray-card" key={record.plot_id}>
            <PlotImage src={imageUrl(record)} alt={altText(record)} />
            <div><strong>{compactTitle(record)}</strong><span>{familyLabel(record.correlation_family)}</span><span>{formatPressure(record.anchor_pressure_gpa)} · {transformLabel(record.correlation_transform)}</span></div>
            <button className="icon-button" type="button" onClick={() => onRemove(record.plot_id)} aria-label="Remove from comparison"><X size={15} /></button>
          </article>
        ))}
      </div>
      <div className="tray-actions">
        <button className="primary-button" type="button" onClick={onCompare} disabled={records.length < 2}>
          Compare {records.length} plots <ArrowRight size={17} />
        </button>
        <button className="text-button" type="button" onClick={onClear}>Clear tray</button>
      </div>
    </aside>
  );
}
