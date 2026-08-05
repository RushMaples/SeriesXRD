export function SemanticLegend({ compact = false }: { compact?: boolean }) {
  return (
    <section className={compact ? "semantic-legend compact" : "semantic-legend"} aria-label="Matrix cell semantics">
      <h3>Semantic legend</h3>
      <div className="semantic-key">
        <span className="semantic-swatch measured-zero" aria-hidden="true" />
        <span><strong>0.000</strong> = measured zero</span>
      </div>
      <div className="semantic-key">
        <span className="semantic-swatch structural-blank" aria-hidden="true" />
        <span>blank = missing peak slot</span>
      </div>
      <div className="semantic-key">
        <span className="semantic-swatch anchor-omitted" aria-hidden="true" />
        <span>anchor row = intentionally omitted</span>
      </div>
      <div className="semantic-key">
        <span className="semantic-swatch triangle-omitted" aria-hidden="true" />
        <span>hatched = strict-triangle omission</span>
      </div>
    </section>
  );
}
