'use client';

import { useQuery } from '@tanstack/react-query';
import { Cpu, WifiOff } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { api } from '@/lib/api/endpoints';
import { METRIC_FA } from '@/lib/utils';

/** Live backend status: demo mode, champion model, connectivity. */
export function StatusStrip() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['health'],
    queryFn: api.health,
    refetchInterval: 30_000,
  });

  if (isLoading) {
    return <span className="hidden h-6 w-40 animate-pulse rounded-full bg-slate-200 dark:bg-slate-700 sm:block" />;
  }

  if (isError || !data) {
    return (
      <Badge tone="danger">
        <WifiOff className="h-3 w-3" />
        بک‌اند در دسترس نیست
      </Badge>
    );
  }

  return (
    <div className="flex items-center gap-2">
      {data.demo_mode ? (
        <Badge tone="violet" title="داده‌های نمایشی مصنوعی فعال است">
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-violet-400 opacity-60" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-violet-500" />
          </span>
          حالت دمو
        </Badge>
      ) : null}
      <Badge
        tone={data.has_trained_model ? 'success' : 'warning'}
        title={data.latest_run ?? undefined}
        className="hidden sm:inline-flex"
      >
        <Cpu className="h-3 w-3" />
        {data.has_trained_model ? `مدل آماده · ${METRIC_FA[data.primary_metric] ?? 'معیار ارزیابی'}` : 'مدل آموزش‌ندیده'}
      </Badge>
    </div>
  );
}
