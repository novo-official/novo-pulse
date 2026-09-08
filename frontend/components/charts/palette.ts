/** One colour system across every chart, tuned for light backgrounds. */
export const CHART = {
  actual: '#0f172a',
  forecast: '#4f46e5',
  backtest: '#7c3aed',
  band: '#6366f1',
  bandOpacity: 0.14,
  scenario: '#0d9488',
  positive: '#059669',
  negative: '#e11d48',
  neutral: '#94a3b8',
  grid: '#e7eaf1',
  axis: '#64748b',
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
