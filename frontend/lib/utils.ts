import {
  GREGORIAN_MONTHS_FA,
  JALALI_MONTHS,
  toJalali,
  type CalendarName,
} from './calendar.ts';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const FA_DIGITS = ['۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹'];

/** Latin digits with thousands separators - kept Latin for numeric legibility. */
export function formatNumber(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return value.toLocaleString('en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function formatCompact(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  const abs = Math.abs(value);
  if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(1)} میلیون`;
  if (abs >= 10_000) return `${(value / 1000).toFixed(0)} هزار`;
  if (abs >= 1000) return `${(value / 1000).toFixed(1)} هزار`;
  return formatNumber(value, abs < 10 && abs % 1 !== 0 ? 1 : 0);
}

export function formatPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return `${value >= 0 ? '+' : ''}${value.toFixed(digits)}٪`;
}

export function formatRatioAsPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return `${(value * 100).toFixed(digits)}٪`;
}

/** Sensible precision for a value of unknown magnitude (prices, counts, rates). */
export function formatAuto(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  const abs = Math.abs(value);
  const digits = abs >= 1000 ? 0 : abs >= 10 ? 1 : abs >= 1 ? 2 : 3;
  return value.toLocaleString('en-US', { maximumFractionDigits: digits });
}

export function formatMetric(value: number | null | undefined, metric: string): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  if (metric === 'wape' || metric === 'r2' || metric === 'rmsle') {
    return metric === 'r2' ? value.toFixed(3) : `${(value * 100).toFixed(1)}٪`;
  }
  if (metric === 'mape' || metric === 'smape') return `${value.toFixed(1)}٪`;
  return value.toFixed(3);
}

export function toPersianDigits(input: string | number): string {
  return String(input).replace(/\d/g, (d) => FA_DIGITS[Number(d)]);
}

/**
 * Dates are formatted in the calendar the reader plans in. Iranian tourism runs
 * on Shamsi months - a peak in مرداد is a fact someone can act on, "اوت" is a
 * translation of one. The pipeline is Gregorian end to end; this is the edge.
 *
 * Prefer `useCalendar()` in components so a change of calendar re-renders them.
 * These take the calendar explicitly so they stay pure and testable.
 */
export function formatDate(
  iso: string | null | undefined,
  calendar: CalendarName = 'jalali',
): string {
  const date = parseIso(iso);
  if (!date) return iso ? iso : '—';
  if (calendar === 'jalali') {
    const jalali = toJalali(date);
    return `${jalali.day} ${JALALI_MONTHS[jalali.month - 1]}`;
  }
  return `${date.getDate()} ${GREGORIAN_MONTHS_FA[date.getMonth()]}`;
}

export function formatFullDate(
  iso: string | null | undefined,
  calendar: CalendarName = 'jalali',
): string {
  const date = parseIso(iso);
  if (!date) return iso ? iso : '—';
  if (calendar === 'jalali') {
    const jalali = toJalali(date);
    return `${jalali.day} ${JALALI_MONTHS[jalali.month - 1]} ${jalali.year}`;
  }
  return `${date.getDate()} ${GREGORIAN_MONTHS_FA[date.getMonth()]} ${date.getFullYear()}`;
}

export function formatDateTime(
  iso: string | null | undefined,
  calendar: CalendarName = 'jalali',
): string {
  const date = parseIso(iso);
  if (!date) return iso ? iso : '—';
  const time = `${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`;
  return `${formatFullDate(iso, calendar)} - ${time}`;
}

function parseIso(iso: string | null | undefined): Date | null {
  if (!iso) return null;
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? null : date;
}

export const CONFIDENCE_FA: Record<string, string> = {
  high: 'بالا',
  medium: 'متوسط',
  low: 'پایین',
  unknown: 'نامشخص',
};

export const SEVERITY_FA: Record<string, string> = {
  high: 'زیاد',
  medium: 'متوسط',
  low: 'کم',
};

export const LEVEL_FA: Record<string, string> = {
  listing: 'اقامتگاه',
  destination: 'مقصد',
  category: 'دسته‌بندی',
  market: 'کل بازار',
};

export const MODEL_KIND_FA: Record<string, string> = {
  baseline: 'مدل پایه',
  ml: 'یادگیری ماشین',
  deep: 'یادگیری عمیق',
  foundation: 'مدل بنیادین',
  meta: 'ترکیبی',
};

/** User-facing names stay Persian even when the API exposes technical IDs. */
export const METRIC_FA: Record<string, string> = {
  wape: 'خطای مطلق وزنی',
  mape: 'خطای درصدی مطلق',
  smape: 'خطای درصدی متقارن',
  rmsle: 'خطای لگاریتمی مربعات',
  r2: 'ضریب تعیین',
};

const DISPLAY_NAME_FA: Record<string, string> = {
  kish: 'کیش',
  tehran: 'تهران',
  isfahan: 'اصفهان',
  mashhad: 'مشهد',
  shiraz: 'شیراز',
  tabriz: 'تبریز',
  rasht: 'رشت',
  qeshm: 'قشم',
  chabahar: 'چابهار',
  bandarabbas: 'بندرعباس',
  ramser: 'رامسر',
  yazd: 'یزد',
  lightgbm: 'لایت جی‌بی‌ام',
  catboost: 'کت‌بوست',
  ensemble: 'مدل ترکیبی',
  moving_average: 'میانگین متحرک',
  historical_mean: 'میانگین تاریخی',
  seasonal_naive_7: 'مدل فصلی ۷ روزه',
  seasonal_naive_30: 'مدل فصلی ۳۰ روزه',
  chronos: 'کرونوس',
  nhits: 'ان‌هیتس',
};

export function displayNameFa(value: string | null | undefined): string {
  if (!value) return '—';
  return DISPLAY_NAME_FA[value.trim().toLowerCase().replace(/[\s-]/g, '_').replace('_', '')]
    ?? DISPLAY_NAME_FA[value.trim().toLowerCase()]
    ?? value;
}
