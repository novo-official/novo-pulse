import { cn } from '@/lib/utils';
import type { ReactNode } from 'react';

const TONES = {
  neutral: 'bg-slate-100 text-slate-700 ring-slate-200 dark:bg-slate-800 dark:text-slate-200 dark:ring-slate-700',
  brand: 'bg-brand-50 text-brand-700 ring-brand-200',
  success: 'bg-emerald-50 text-emerald-700 ring-emerald-200',
  warning: 'bg-amber-50 text-amber-700 ring-amber-200',
  danger: 'bg-rose-50 text-rose-700 ring-rose-200',
  info: 'bg-sky-50 text-sky-700 ring-sky-200',
  violet: 'bg-violet-50 text-violet-700 ring-violet-200',
} as const;

export type BadgeTone = keyof typeof TONES;

export function Badge({
  children,
  tone = 'neutral',
  className,
  title,
}: {
  children: ReactNode;
  tone?: BadgeTone;
  className?: string;
  title?: string;
}) {
  return (
    <span
      title={title}
      className={cn(
        'inline-flex min-h-6 items-center gap-1 rounded-full px-2.5 py-0.5 text-[11px] font-semibold leading-5 ring-1 ring-inset',
        TONES[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function toneForChange(change: number | null | undefined): BadgeTone {
  if (change === null || change === undefined) return 'neutral';
  if (change > 3) return 'success';
  if (change < -3) return 'danger';
  return 'neutral';
}

export function toneForConfidence(label: string | null | undefined): BadgeTone {
  if (label === 'high') return 'success';
  if (label === 'medium') return 'warning';
  if (label === 'low') return 'danger';
  return 'neutral';
}

export function toneForSeverity(severity: string): BadgeTone {
  if (severity === 'high' || severity === 'critical') return 'danger';
  if (severity === 'medium') return 'warning';
  return 'neutral';
}
