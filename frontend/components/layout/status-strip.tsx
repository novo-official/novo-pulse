'use client';

import { useQuery } from '@tanstack/react-query';
import { CircleCheck, TriangleAlert, WifiOff } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { pol4 } from '@/lib/pol4/api';

/**
 * Whether the dashboard is showing a real forecast.
 *
 * There is no demo mode to report. Either the pipeline has produced artefacts -
 * in which case the cutoff and the grid size are shown - or it has not, and the
 * strip says so rather than letting an empty page look like a working one.
 */
export function StatusStrip() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['pol4-overview'],
    queryFn: pol4.overview,
    refetchInterval: 60_000,
  });

  if (isLoading) {
    return <span className="hidden h-6 w-44 animate-pulse rounded-full bg-slate-200 dark:bg-slate-700 sm:block" />;
  }

  if (isError) {
    return (
      <Badge tone="danger">
        <WifiOff className="h-3 w-3" />
        بک‌اند در دسترس نیست
      </Badge>
    );
  }

  if (!data?.available || !data.data) {
    return (
      <Badge tone="warning">
        <TriangleAlert className="h-3 w-3" />
        خروجی ساخته نشده
      </Badge>
    );
  }

  const { kpis, cutoff } = data.data;
  return (
    <Badge tone="success" title={`داده تا ${cutoff}`} className="hidden sm:inline-flex">
      <CircleCheck className="h-3 w-3" />
      <span dir="ltr">
        {kpis.cities}×{kpis.dates} · {cutoff}
      </span>
    </Badge>
  );
}
