import type { ReactNode } from "react";
export function EquationDisplay({ children }: { children: ReactNode }) {
  return (
    <div className="math-equation panel">
      <span className="eyebrow">EQUATION → VALUES</span>
      <div>{children}</div>
    </div>
  );
}
export function InterpretationPanel({
  children,
  meaning,
}: {
  children: ReactNode;
  meaning: string;
}) {
  return (
    <section className="math-interpretation panel">
      <h3>What you’re seeing</h3>
      <p>{children}</p>
      <p>
        <b>ML connection</b> {meaning}
      </p>
    </section>
  );
}
export function VisualizationShell({
  title,
  subtitle,
  equation,
  controls,
  children,
  values,
  interpretation,
  meaning,
  timeline,
}: {
  title: string;
  subtitle: string;
  equation: ReactNode;
  controls: ReactNode;
  children: ReactNode;
  values: readonly (readonly [string, ReactNode])[];
  interpretation: ReactNode;
  meaning: string;
  timeline: ReactNode;
}) {
  return (
    <article className="math-shell">
      <div className="view-title">
        <div>
          <p className="eyebrow">INTERACTIVE / 2D SANDBOX</p>
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>
      </div>
      <EquationDisplay>{equation}</EquationDisplay>
      <div className="math-experiment">
        <section className="math-controls panel">
          <h3>Interactive controls</h3>
          {controls}
        </section>
        <section
          className="math-geometry panel"
          aria-label={`${title} geometry`}
        >
          {children}
        </section>
      </div>
      <section className="math-values" aria-label="Computed values">
        {values.map(([label, value]) => (
          <div className="metric" key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </section>
      <InterpretationPanel meaning={meaning}>
        {interpretation}
      </InterpretationPanel>
      {timeline}
    </article>
  );
}
