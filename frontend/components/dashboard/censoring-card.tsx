'use client';

import { PackageOpen, SlidersHorizontal } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Card, CardBody, CardHeader } from '@/components/ui/card';
import { Td, TableWrap, Th } from '@/components/ui/table';
import { InfoHint } from '@/components/ui/tooltip';
import type { CensoringReport, TuningRun } from '@/lib/types/api';
import { formatNumber, formatRatioAsPercent } from '@/lib/utils';

/** Demand censoring: what the target hides when supply runs out. */
export function CensoringCard({ report }: { report: CensoringReport }) {
  if (!report.enabled) return null;
  const heavy = report.censored_share > 0.25;

  return (
    <Card className="signal-card bg-gradient-to-b from-amber-50/30 to-surface">
      <CardHeader
        icon={<PackageOpen className="h-4.5 w-4.5" />}
        title="سانسور تقاضا (ظرفیت تکمیل)"
        subtitle="وقتی اقامتگاه پر می‌شود، عدد ثبت‌شده ظرفیت است نه تقاضای واقعی بازار"
        action={
          <Badge tone={heavy ? 'warning' : 'neutral'}>
            {formatRatioAsPercent(report.censored_share)} از دوره‌ها
          </Badge>
        }
      />
      <CardBody className="space-y-4">
        <div className="grid gap-4 sm:grid-cols-3">
          <div className="data-tile">
            <p className="text-xs text-muted">دوره‌های سانسورشده</p>
            <p className="nums mt-1 text-xl font-semibold text-ink">
              {formatNumber(report.censored_periods)}
            </p>
          </div>
          <div className="data-tile">
            <p className="text-xs text-muted">اقامتگاه‌های متأثر</p>
            <p className="nums mt-1 text-xl font-semibold text-ink">
              {formatNumber(report.affected_entities)}
            </p>
          </div>
          <div className="data-tile border-amber-200 bg-amber-50/50">
            <p className="text-xs text-muted">برآورد تقاضای پنهان</p>
            <p className="nums mt-1 text-xl font-semibold text-amber-600">
              {formatRatioAsPercent(report.mean_uplift)}
            </p>
          </div>
        </div>
        <p className="rounded-xl border border-amber-200/70 bg-amber-50 px-3.5 py-3 text-xs leading-6 text-amber-900">
          پیش‌بینی این سیستم «تقاضای قابل‌فروش» را نشان می‌دهد، نه تقاضای نامحدود بازار. در
          دوره‌هایی که ظرفیت تکمیل می‌شود، تقاضای واقعی بالاتر از عدد ثبت‌شده بوده است.
        </p>
        <p className="debug-only text-[11px] leading-5 text-muted" dir="ltr">
          {report.method}
        </p>
      </CardBody>
    </Card>
  );
}

/** Hyper-parameter search outcome, including the runs that changed nothing. */
export function TuningCard({ runs }: { runs: TuningRun[] }) {
  if (!runs.length) return null;

  return (
    <Card className="signal-card">
      <CardHeader
        icon={<SlidersHorizontal className="h-4.5 w-4.5" />}
        title="جست‌وجوی ابرپارامترها"
        subtitle="با Optuna روی پنجره زمانی کنارگذاشته‌شده، با سقف زمانی مشخص"
        action={
          <InfoHint>
            اگر جست‌وجو نتواند از مقادیر پیش‌فرض بهتر عمل کند، پیش‌فرض‌ها حفظ می‌شوند. نتیجه هر
            جست‌وجو — حتی وقتی بهبودی حاصل نشده — اینجا گزارش می‌شود.
          </InfoHint>
        }
      />
      <CardBody>
        <TableWrap minWidth={560}>
          <thead>
            <tr>
              <Th align="right">مدل</Th>
              <Th>آزمایش‌ها</Th>
              <Th>پیش‌فرض</Th>
              <Th>بهترین</Th>
              <Th>نتیجه</Th>
              <Th>زمان</Th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <tr key={run.model}>
                <Td align="right" className="font-medium">
                  <span dir="ltr">{run.model}</span>
                </Td>
                <Td className="nums text-muted">{run.n_trials}</Td>
                <Td className="nums text-muted">
                  {run.default_score === null ? '—' : run.default_score.toFixed(4)}
                </Td>
                <Td className="nums font-semibold">
                  {run.best_score === null ? '—' : run.best_score.toFixed(4)}
                </Td>
                <Td>
                  <Badge tone={run.improved ? 'success' : 'neutral'} title={run.note}>
                    {run.improved ? 'تنظیم شد' : 'پیش‌فرض حفظ شد'}
                  </Badge>
                </Td>
                <Td className="nums text-muted">{run.seconds}s</Td>
              </tr>
            ))}
          </tbody>
        </TableWrap>
        {runs.some((run) => run.note) ? (
          <div className="mt-3 space-y-1">
            {runs
              .filter((run) => run.note)
              .map((run) => (
                <p key={run.model} className="text-xs leading-5 text-muted">
                  <span dir="ltr">{run.model}</span> — {run.note}
                </p>
              ))}
          </div>
        ) : null}
      </CardBody>
    </Card>
  );
}
