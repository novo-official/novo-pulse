'use client';

/**
 * Loading / empty / error states.
 *
 * Every data surface in the app uses these, so a demo can never end up staring
 * at an infinite spinner: each state says what happened and offers a way out.
 */
import { AlertTriangle, DatabaseZap, Inbox, RefreshCw } from 'lucide-react';
import type { ReactNode } from 'react';

import { cn } from '@/lib/utils';

export function Skeleton({
  className,
  style,
}: {
  className?: string;
  style?: React.CSSProperties;
}) {
  return (
    <div
      style={style}
      className={cn(
        'relative overflow-hidden rounded-lg bg-slate-100',
        'after:absolute after:inset-0 after:-translate-x-full after:animate-shimmer',
        'after:bg-gradient-to-l after:from-transparent after:via-white/70 after:to-transparent',
        className,
      )}
    />
  );
}

export function CardSkeleton({ lines = 3, className }: { lines?: number; className?: string }) {
  return (
    <div className={cn('space-y-3 p-5', className)}>
      <Skeleton className="h-4 w-32" />
      {Array.from({ length: lines }).map((_, index) => (
        <Skeleton key={index} className="h-3 w-full" />
      ))}
    </div>
  );
}

export function ChartSkeleton({ height = 300 }: { height?: number }) {
  return (
    <div className="space-y-3 p-5">
      <Skeleton className="h-4 w-40" />
      <Skeleton className="w-full rounded-xl" style={{ height }} />
    </div>
  );
}

export function EmptyState({
  title = 'داده‌ای برای نمایش وجود ندارد',
  description,
  icon,
  action,
}: {
  title?: string;
  description?: ReactNode;
  icon?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 px-6 py-12 text-center">
      <span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-slate-100 text-slate-400">
        {icon ?? <Inbox className="h-6 w-6" />}
      </span>
      <div>
        <p className="text-sm font-semibold text-ink">{title}</p>
        {description ? <p className="mt-1.5 max-w-md text-sm leading-6 text-muted">{description}</p> : null}
      </div>
      {action}
    </div>
  );
}

export function NoModelState({ detail }: { detail?: string }) {
  return (
    <EmptyState
      icon={<DatabaseZap className="h-6 w-6" />}
      title="هنوز مدلی آموزش ندیده است"
      description={
        detail ??
        'برای دیدن پیش‌بینی‌ها ابتدا یک مدل آموزش دهید. با دستور make seed داده نمونه ساخته و مدل دمو آموزش داده می‌شود.'
      }
      action={
        <code className="rounded-lg bg-slate-900 px-3 py-1.5 text-xs text-slate-100" dir="ltr">
          python manage.py seed_demo
        </code>
      }
    />
  );
}

export function ErrorState({
  message,
  onRetry,
}: {
  message?: string;
  onRetry?: () => void;
}) {
  return (
    <EmptyState
      icon={<AlertTriangle className="h-6 w-6 text-amber-500" />}
      title="خطا در دریافت اطلاعات"
      description={message ?? 'ارتباط با سرور برقرار نشد.'}
      action={
        onRetry ? (
          <button
            type="button"
            onClick={onRetry}
            className="inline-flex items-center gap-1.5 rounded-lg border border-line bg-white px-3 py-1.5 text-sm font-medium text-ink transition hover:bg-slate-50"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            تلاش دوباره
          </button>
        ) : null
      }
    />
  );
}

/** One wrapper that resolves the four states for any query-backed surface. */
export function AsyncBoundary({
  isLoading,
  error,
  isEmpty,
  onRetry,
  skeleton,
  empty,
  children,
}: {
  isLoading: boolean;
  error?: unknown;
  isEmpty?: boolean;
  onRetry?: () => void;
  skeleton?: ReactNode;
  empty?: ReactNode;
  children: ReactNode;
}) {
  if (isLoading) return <>{skeleton ?? <CardSkeleton />}</>;
  if (error) {
    const message = error instanceof Error ? error.message : undefined;
    return <ErrorState message={message} onRetry={onRetry} />;
  }
  if (isEmpty) return <>{empty ?? <EmptyState />}</>;
  return <>{children}</>;
}
