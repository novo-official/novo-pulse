/**
 * Chart palette for the Pol 4 dashboard.
 *
 * Three categorical slots, validated with the data-viz validator in both modes:
 * worst adjacent CVD ΔE 9.2 light / 9.4 dark (≥8 target), worst normal-vision
 * ΔE 27.6 light / 26.5 dark (≥15 floor). Aqua sits below 3:1 on the light
 * surface, so every chart that uses it also ships a legend and direct labels -
 * identity is never carried by colour alone.
 *
 * The hues are bound to meaning, not to series order: observed demand is always
 * orange, the forecast always blue, the historical expectation always aqua. A
 * filter that changes which cities are on screen never repaints them.
 */
export const SERIES = {
  /** Predicted final demand - the model's answer. */
  forecast: 'var(--pol4-forecast)',
  /** Demand already counted in evaluation.csv at the cutoff. */
  observed: 'var(--pol4-observed)',
  /** What this city's own history says to expect - a reference, not a forecast. */
  expected: 'var(--pol4-expected)',
  grid: 'var(--pol4-grid)',
  axis: 'var(--pol4-axis)',
  ink: 'var(--pol4-ink)',
} as const;

/** Single-hue sequential ramp for the heatmap. Light → dark, never a rainbow. */
export const HEAT_STEPS = [
  'var(--pol4-heat-0)',
  'var(--pol4-heat-1)',
  'var(--pol4-heat-2)',
  'var(--pol4-heat-3)',
  'var(--pol4-heat-4)',
] as const;

/** Pick a ramp step for `value` on a 0..max scale. */
export function heatStep(value: number, max: number): string {
  if (max <= 0 || value <= 0) return HEAT_STEPS[0];
  const ratio = Math.min(value / max, 1);
  // Square root spreads the low end, where most (city, date) cells sit.
  const index = Math.min(HEAT_STEPS.length - 1, Math.floor(Math.sqrt(ratio) * HEAT_STEPS.length));
  return HEAT_STEPS[index];
}

export const axisProps = {
  stroke: SERIES.axis,
  fontSize: 11,
  tickLine: false,
  axisLine: false,
} as const;

const fa = new Intl.NumberFormat('fa-IR');
const faCompact = new Intl.NumberFormat('fa-IR', { notation: 'compact', maximumFractionDigits: 1 });
const faDecimal = new Intl.NumberFormat('fa-IR', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const faDate = new Intl.DateTimeFormat('fa-IR-u-ca-persian', { day: 'numeric', month: 'long' });
const faDateFull = new Intl.DateTimeFormat('fa-IR-u-ca-persian', {
  day: 'numeric',
  month: 'long',
  year: 'numeric',
});

export const num = (value: number) => fa.format(Math.round(value));
export const compact = (value: number) => faCompact.format(value);
export const percent = (value: number, digits = 1) =>
  `${new Intl.NumberFormat('fa-IR', { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value * 100)}٪`;
export const ratio = (value: number) => faDecimal.format(value);

/** Azar dates are shown in Jalali - that is the calendar the window belongs to. */
export const jalali = (iso: string) => {
  const date = new Date(`${iso}T00:00:00`);
  return Number.isNaN(date.getTime()) ? iso : faDate.format(date);
};
export const jalaliFull = (iso: string) => {
  const date = new Date(`${iso}T00:00:00`);
  return Number.isNaN(date.getTime()) ? iso : faDateFull.format(date);
};
