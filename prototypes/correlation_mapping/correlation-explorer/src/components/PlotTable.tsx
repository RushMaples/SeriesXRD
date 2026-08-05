import { Check, Star } from "lucide-react";

import { imageUrl } from "../api";
import {
  altText,
  compactTitle,
  familyLabel,
  formatPressure,
  localPeak,
  sampleLabel,
  transformLabel,
  validationTone,
  visualizationLabel,
} from "../format";
import type { PlotRecord } from "../types";
import { PlotImage } from "./PlotImage";

type PlotTableProps = {
  records: PlotRecord[];
  selectedId: string | null;
  compareIds: string[];
  favoriteIds: Set<string>;
  onSelect: (record: PlotRecord) => void;
  onToggleCompare: (record: PlotRecord) => void;
  onToggleFavorite: (plotId: string) => void;
};

export function PlotTable({
  records,
  selectedId,
  compareIds,
  favoriteIds,
  onSelect,
  onToggleCompare,
  onToggleFavorite,
}: PlotTableProps) {
  return (
    <div className="plot-table-wrap">
      <table className="plot-table">
        <thead>
          <tr>
            <th aria-label="Compare" />
            <th>Plot</th>
            <th>Sample</th>
            <th>Correlation calculation</th>
            <th>Family / visualization</th>
            <th>Anchor pressure</th>
            <th>Local peak</th>
            <th>Validation</th>
            <th aria-label="Favorite" />
          </tr>
        </thead>
        <tbody>
          {records.map((record) => {
            const compared = compareIds.includes(record.plot_id);
            const favorite = favoriteIds.has(record.plot_id);
            return (
              <tr
                key={record.plot_id}
                className={selectedId === record.plot_id ? "selected" : ""}
                onClick={() => onSelect(record)}
              >
                <td>
                  <button
                    type="button"
                    className={compared ? "compare-check checked" : "compare-check"}
                    onClick={(event) => { event.stopPropagation(); onToggleCompare(record); }}
                    aria-label={compared ? "Remove from comparison" : "Add to comparison"}
                  >
                    {compared ? <Check size={14} /> : null}
                  </button>
                </td>
                <td>
                  <div className="table-plot-cell">
                    <PlotImage src={imageUrl(record)} alt={altText(record)} />
                    <div><strong>{compactTitle(record)}</strong><span>{record.title}</span></div>
                  </div>
                </td>
                <td>{sampleLabel(record.sample)}</td>
                <td>{transformLabel(record.correlation_transform)}</td>
                <td><strong>{familyLabel(record.correlation_family)}</strong><span>{visualizationLabel(record.visualization_type)}</span></td>
                <td>{formatPressure(record.anchor_pressure_gpa ?? record.pressure_gpa)}</td>
                <td>{localPeak(record) ?? "N/A"}</td>
                <td><span className={`status-badge ${validationTone(record.validation_status)}`}>{record.validation_status}</span></td>
                <td>
                  <button
                    type="button"
                    className={favorite ? "favorite-button active" : "favorite-button"}
                    onClick={(event) => { event.stopPropagation(); onToggleFavorite(record.plot_id); }}
                    aria-label={favorite ? "Remove favorite" : "Add favorite"}
                  >
                    <Star size={15} fill={favorite ? "currentColor" : "none"} />
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
