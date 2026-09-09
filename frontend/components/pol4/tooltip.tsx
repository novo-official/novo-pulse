'use client';

import type { CSSProperties } from 'react';

import { num } from './theme';

interface Item {
  name?: string;
  dataKey?: string | number;
  value?: number;
  color?: string;
}

/**
 * The hover layer every Pol 4 chart ships.
 *
 * Values wear text tokens; the colour lives in the swatch beside them, never in
 * the number itself.
 */
export function ChartTooltip({
  active,
  payload,
  label,
  labelFormatter,
  valueFormatter = num,
  order,
}: {
  active?: boolean;
  payload?: Item[];
  label?: string | number;
  labelFormatter?: (value: string) => string;
  valueFormatter?: (value: number) => string;
  order?: string[];
}) {
  if (!active || !payload?.length) return null;

  const rows = order
    ? order
        .map((key) => payload.find((item) => item.dataKey === key))
        .filter((item): item is Item => Boolean(item))
    : payload;

  return (
    <div className="pol4-tooltip" dir="rtl">
      {label !== undefined ? (
        <p className="pol4-tooltip__label">
          {labelFormatter ? labelFormatter(String(label)) : String(label)}
        </p>
      ) : null}
      <ul>
        {rows.map((item) => (
          <li key={String(item.dataKey ?? item.name)}>
            <span
              className="pol4-tooltip__dot"
              style={{ '--dot': item.color } as CSSProperties}
            />
            <span className="pol4-tooltip__name">{item.name}</span>
            <strong dir="ltr">{valueFormatter(Number(item.value ?? 0))}</strong>
          </li>
        ))}
      </ul>
    </div>
  );
}
