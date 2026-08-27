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
  if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (abs >= 10_000) return `${(value / 1000).toFixed(0)}K`;
  if (abs >= 1000) return `${(value / 1000).toFixed(1)}K`;
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

const FA_MONTHS = [
  'ژانویه', 'فوریه', 'مارس', 'آوریل', 'مه', 'ژوئن',
  'ژوئیه', 'اوت', 'سپتامبر', 'اکتبر', 'نوامبر', 'دسامبر',
];

/** Short readable date. The ML pipeline never sees this - display only. */
export function formatDate(iso: string | null | undefined): string {
  if (!iso) return '—';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return `${date.getDate()} ${FA_MONTHS[date.getMonth()]}`;
}

export function formatFullDate(iso: string | null | undefined): string {
  if (!iso) return '—';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return `${date.getDate()} ${FA_MONTHS[date.getMonth()]} ${date.getFullYear()}`;
}

export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return '—';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const time = `${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`;
  return `${formatFullDate(iso)} - ${time}`;
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
