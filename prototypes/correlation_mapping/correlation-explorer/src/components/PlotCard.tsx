import { AlertTriangle, Check, Star } from "lucide-react";
import { memo } from "react";

import { imageUrl } from "../api";
import {
  altText,
  cardTitle,
  compactTitle,
  displayDomainLabel,
  formatNumber,
  formatPressure,
  isWaterfall,
  localPeak,
  sampleLabel,
  transformLabel,
  validationTone,
  windowLabel,
} from "../format";
import type { PlotRecord } from "../types";
import { PlotImage } from "./PlotImage";

type PlotCardProps = {
  record: PlotRecord;
  selected: boolean;
  compared: boolean;
  favorite: boolean;
  onSelect: (record: PlotRecord) => void;
  onToggleCompare: (record: PlotRecord) => void;
  onToggleFavorite: (plotId: string) => void;
};

export const PlotCard = memo(function PlotCard({
  record,
  selected,
  compared,
  favorite,
  onSelect,
  onToggleCompare,
  onToggleFavorite,
}: PlotCardProps) {
  const pressure = record.anchor_pressure_gpa ?? record.pressure_gpa;
  const peak = localPeak(record);
  return (
    <article
      className={`plot-card${selected ? " selected" : ""}${compared ? " compared" : ""}`}
      onClick={() => onSelect(record)}
      aria-current={selected ? "true" : undefined}
    >
      <div className="card-controls">
        <button
          type="button"
          className={compared ? "compare-check checked" : "compare-check"}
          onClick={(event) => { event.stopPropagation(); onToggleCompare(record); }}
          aria-label={compared ? "Remove from comparison" : "Add to comparison"}
          aria-pressed={compared}
        >
          {compared ? <Check size={14} /> : null}
        </button>
        <button
          type="button"
          className={favorite ? "favorite-button active" : "favorite-button"}
          onClick={(event) => { event.stopPropagation(); onToggleFavorite(record.plot_id); }}
          aria-label={favorite ? "Remove favorite" : "Add favorite"}
          aria-pressed={favorite}
        >
          <Star size={15} fill={favorite ? "currentColor" : "none"} />
        </button>
      </div>
      <div className="card-visual">
        <PlotImage src={imageUrl(record)} alt={altText(record)} />
      </div>
      <div className="card-copy">
        <div className="card-badges">
          <span>{sampleLabel(record.sample)}</span>
          <span>{transformLabel(record.correlation_transform)}</span>
          <span className={`status-badge ${validationTone(record.validation_status)}`}>{record.validation_status}</span>
        </div>
        <h3>{cardTitle(record)}</h3>
        <p className="card-anchor">{compactTitle(record)}</p>
        {pressure != null ? <p>{formatPressure(pressure)}{peak != null ? ` · Local peak ${peak}` : ""}</p> : null}
        {record.anchor_two_theta_deg != null ? <p>2θ {formatNumber(record.anchor_two_theta_deg)}°</p> : null}
        {record.window_start_deg != null ? <p>Window {windowLabel(record)}</p> : null}
        {record.half_width_factor != null ? <p>q support factor {formatNumber(record.half_width_factor, 2)}</p> : null}
        {isWaterfall(record) ? <p>Curve: {displayDomainLabel(record.display_profile_domain)}</p> : null}
        {record.classification_warning ? (
          <p className="warning-line"><AlertTriangle size={13} /> Classification warning</p>
        ) : null}
      </div>
    </article>
  );
});
