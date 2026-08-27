'use client';

import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  Gauge,
  Minus,
  Sparkles,
  Target,
  TrendingUp,
} from 'lucide-react';
import type { ReactNode } from 'react';

import { Badge, toneForChange, toneForConfidence, type BadgeTone } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import { InfoHint } from '@/components/ui/tooltip';
import type { DashboardSummary } from '@/lib/types/api';
import {
  CONFIDENCE_FA,
  cn,
  formatCompact,
  formatMetric,
  formatNumber,
  formatPercent,
  formatRatioAsPercent,
} from '@/lib/utils';

function Kpi({
  label,
  value,
  hint,
  icon,
  tone = 'neutral',
  badge,
  footnote,
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  icon: ReactNode;
  tone?: BadgeTone;
  badge?: ReactNode;
  footnote?: ReactNode;
}) {
  return (
    <Card className="animate-fade-up p-5">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2 text-sm font-medium text-muted">
          <span
            className={cn(
              'flex h-8 w-8 items-center justify-center rounded-lg',
              tone === 'success' && 'bg-emerald-50 text-emerald-600',
              tone === 'danger' && 'bg-rose-50 text-rose-600',
              tone === 'warning' && 'bg-amber-50 text-amber-600',
              tone === 'violet' && 'bg-violet-50 text-violet-600',
              (tone === 'neutral' || tone === 'brand' || tone === 'info') &&
                'bg-brand-50 text-brand-600',
            )}
          >
            {icon}
          </span>
          <span className="flex items-center gap-1">
            {label}
            {hint ? <InfoHint>{hint}</InfoHint> : null}
          </span>
        </div>
        {badge}
      </div>
      <p className="kpi-value mt-4 nums">{value}</p>
      {footnote ? <p className="mt-2 text-xs leading-5 text-muted">{footnote}</p> : null}
    </Card>
  );
}

function ChangeIcon({ change }: { change: number | null }) {
  if (change === null) return <Minus className="h-3 w-3" />;
  if (change > 0) return <ArrowUpRight className="h-3 w-3" />;
  if (change < 0) return <ArrowDownRight className="h-3 w-3" />;
  return <Minus className="h-3 w-3" />;
}

export function KpiCards({ summary }: { summary: DashboardSummary }) {
  const metric = summary.model.primary_metric ?? 'wape';
  const growth = summary.insights.fastest_growing ?? summary.top_growth;

  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-6">
      <Kpi
        label={`تقاضای ${summary.horizon} روز آینده`}
        icon={<TrendingUp className="h-4 w-4" />}
        value={formatNumber(summary.forecast_total)}
        footnote={`مجموع پیش‌بینی در سطح ${summary.level === 'destination' ? 'مقصد' : 'انتخاب‌شده'}`}
        hint="مجموع مقادیر پیش‌بینی‌شده هدف در کل افق انتخاب‌شده."
      />

      <Kpi
        label="تغییر نسبت به دوره قبل"
        icon={<Activity className="h-4 w-4" />}
        tone={toneForChange(summary.change_pct)}
        value={formatPercent(summary.change_pct)}
        badge={
          <Badge tone={toneForChange(summary.change_pct)}>
            <ChangeIcon change={summary.change_pct} />
            {formatCompact(summary.previous_period_total)}
          </Badge>
        }
        footnote={`در مقایسه با ${summary.horizon} روز گذشته`}
        hint="پیش‌بینی افق آینده در مقابل مجموع واقعی همان تعداد روز گذشته."
      />

      <Kpi
        label="مقصد با بیشترین رشد"
        icon={<Sparkles className="h-4 w-4" />}
        tone="violet"
        value={growth ? growth.label : '—'}
        badge={
          growth ? (
            <Badge tone={toneForChange(growth.change_pct)}>{formatPercent(growth.change_pct)}</Badge>
          ) : null
        }
        footnote={
          growth ? `تقاضای پیش‌بینی‌شده: ${formatNumber(growth.forecast_total)}` : 'داده کافی نیست'
        }
      />

      <Kpi
        label="بازه اطمینان ۸۰٪"
        icon={<Gauge className="h-4 w-4" />}
        value={
          <span className="text-xl lg:text-2xl">
            {formatCompact(summary.forecast_lower)} – {formatCompact(summary.forecast_upper)}
          </span>
        }
        footnote={`پهنای نسبی: ${formatRatioAsPercent(summary.relative_interval_width)} از سطح پیش‌بینی`}
        hint="کران‌های P10 و P90. پهنای نسبی، پهنای بازه تقسیم بر مقدار پیش‌بینی است."
      />

      <Kpi
        label={`دقت مدل (${metric.toUpperCase()})`}
        icon={<Target className="h-4 w-4" />}
        tone={summary.model.improvement_pct && summary.model.improvement_pct > 0 ? 'success' : 'neutral'}
        value={formatMetric(summary.model.primary_value, metric)}
        badge={
          summary.model.improvement_pct !== null ? (
            <Badge tone="success" title={`نسبت به ${summary.model.baseline_model}`}>
              <ArrowUpRight className="h-3 w-3" />
              {summary.model.improvement_pct.toFixed(1)}٪ بهتر از پایه
            </Badge>
          ) : null
        }
        footnote={`مدل منتخب: ${summary.model.champion ?? '—'}`}
        hint="خطای مدل روی اعتبارسنجی متحرک زمانی، در مقایسه با بهترین مدل پایه."
      />

      <Kpi
        label="ناهنجاری‌های مهم"
        icon={<Activity className="h-4 w-4" />}
        tone={summary.anomalies.important > 0 ? 'warning' : 'success'}
        value={formatNumber(summary.anomalies.important)}
        badge={
          <Badge tone={toneForConfidence(summary.confidence.label)}>
            اطمینان {CONFIDENCE_FA[summary.confidence.label]}
          </Badge>
        }
        footnote={`مجموع شناسایی‌شده: ${formatNumber(summary.anomalies.total)}`}
        hint={summary.confidence.formula}
      />
    </div>
  );
}
