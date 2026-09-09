/** Theme-aware CSS values allow Recharts SVGs to update without a rerender. */
export const CHART = {
  actual: 'rgb(var(--chart-actual))',
  forecast: 'rgb(var(--indigo-deep))',
  backtest: 'rgb(var(--violet-deep))',
  band: 'rgb(var(--indigo))',
  bandOpacity: 0.14,
  scenario: '#0d9488',
  positive: '#059669',
  negative: '#e11d48',
  neutral: '#94a3b8',
  grid: 'rgb(var(--chart-grid))',
  axis: 'rgb(var(--chart-axis))',
};

export const SERIES_COLORS = [
  '#4f46e5',
  '#0d9488',
  '#d97706',
  '#7c3aed',
  '#e11d48',
  '#0284c7',
  '#65a30d',
  '#c026d3',
];

export const axisProps = {
  stroke: CHART.axis,
  fontSize: 11,
  tickLine: false,
  axisLine: false,
} as const;
