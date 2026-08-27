'use client';

import { useQuery } from '@tanstack/react-query';
import { CircleDot, Cpu, WifiOff } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { api } from '@/lib/api/endpoints';

/** Live backend status: demo mode, champion model, connectivity. */
export function StatusStrip() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['health'],
    queryFn: api.health,
    refetchInterval: 30_000,
  });

  if (isLoading) {
    return <span className="hidden h-6 w-40 animate-pulse rounded-full bg-slate-200 sm:block" />;
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
          <CircleDot className="h-3 w-3" />
          حالت دمو
        </Badge>
      ) : null}
      <Badge
        tone={data.has_trained_model ? 'success' : 'warning'}
        title={data.latest_run ?? undefined}
        className="hidden sm:inline-flex"
      >
        <Cpu className="h-3 w-3" />
        {data.has_trained_model ? `مدل آماده · ${data.primary_metric.toUpperCase()}` : 'مدل آموزش‌ندیده'}
      </Badge>
    </div>
  );
}
