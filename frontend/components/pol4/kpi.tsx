'use client';

import type { ReactNode } from 'react';

/**
 * A single headline number.
 *
 * Compact by design: the KPI row is a summary, and the charts below it carry
 * the detail. `note` says where the number comes from, so no figure on the
 * dashboard is unexplained.
 */
export function Kpi({
  label,
  value,
  note,
  tone = 'default',
}: {
  label: string;
  value: ReactNode;
  note?: string;
  tone?: 'default' | 'good' | 'warn';
}) {
  return (
    <div className={`pol4-kpi pol4-kpi--${tone}`}>
      <p className="pol4-kpi__label">{label}</p>
      <p className="pol4-kpi__value" dir="auto">
        {value}
      </p>
      {note ? <p className="pol4-kpi__note">{note}</p> : null}
    </div>
  );
}

export function KpiRow({ children }: { children: ReactNode }) {
  return <div className="pol4-kpi-row">{children}</div>;
}
