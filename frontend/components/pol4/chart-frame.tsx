'use client';

import type { ReactNode } from 'react';

/**
 * The frame every Pol 4 chart sits in.
 *
 * Enforces the parts a chart must have: a title that names what is plotted, the
 * unit, and - for anything with more than one series - a legend, so identity is
 * never colour alone. `source` names the artefact the numbers came from, which
 * is the traceability rule made visible rather than documented elsewhere.
 */
export function ChartFrame({
  title,
  unit,
  hint,
  source,
  legend,
  action,
  children,
  height = 320,
}: {
  title: string;
  unit?: string;
  hint?: string;
  source?: string;
  legend?: { label: string; color: string }[];
  action?: ReactNode;
  children: ReactNode;
  height?: number;
}) {
  return (
    <section className="pol4-card">
      <header className="pol4-card__head">
        <div className="pol4-card__titles">
          <h2 className="pol4-card__title">{title}</h2>
          {hint ? <p className="pol4-card__hint">{hint}</p> : null}
        </div>
        <div className="pol4-card__meta">
          {action}
          {unit ? <span className="pol4-card__unit">{unit}</span> : null}
        </div>
      </header>

      {legend && legend.length > 1 ? (
        <ul className="pol4-legend">
          {legend.map((item) => (
            <li key={item.label}>
              <span className="pol4-legend__swatch" style={{ background: item.color }} />
              {item.label}
            </li>
          ))}
        </ul>
      ) : null}

      <div className="pol4-card__body chart-ltr" style={{ height }} dir="ltr">
        {children}
      </div>

      {source ? <p className="pol4-card__source">منبع: {source}</p> : null}
    </section>
  );
}
