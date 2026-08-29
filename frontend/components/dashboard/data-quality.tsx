'use client';

import { CheckCircle2, ShieldAlert } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Card, CardBody, CardHeader } from '@/components/ui/card';
import { EmptyState } from '@/components/ui/states';
import type { ValidationReport } from '@/lib/types/api';
import { cn } from '@/lib/utils';

const SEVERITY_TONE = {
  critical: 'danger',
  high: 'danger',
  medium: 'warning',
  low: 'neutral',
  info: 'info',
} as const;

const SEVERITY_FA = {
  critical: 'بحرانی',
  high: 'زیاد',
  medium: 'متوسط',
  low: 'کم',
  info: 'اطلاع',
} as const;

const GRADE_FA: Record<string, string> = {
  excellent: 'عالی',
  good: 'خوب',
  fair: 'متوسط',
  poor: 'ضعیف',
  critical: 'بحرانی',
};

export function DataQualityCard({ report }: { report: ValidationReport }) {
  const score = report.health_score ?? 0;
  const tone = score >= 85 ? 'emerald' : score >= 65 ? 'amber' : 'rose';

  return (
    <Card className="signal-card">
      <CardHeader
        icon={<ShieldAlert className="h-4.5 w-4.5" />}
        title="کیفیت داده"
        subtitle="بررسی خودکار پیش از آموزش مدل"
        action={<Badge tone="neutral">{report.n_findings} یافته</Badge>}
      />
      <CardBody>
        <div className="mb-5 flex items-center gap-4 rounded-2xl border border-line/70 bg-surface p-4 shadow-sm">
          <div className="relative h-20 w-20 shrink-0">
            <svg viewBox="0 0 36 36" className="h-20 w-20 -rotate-90">
              <circle cx="18" cy="18" r="15.9" fill="none" stroke="#e2e8f0" strokeWidth="3" />
              <circle
                cx="18"
                cy="18"
                r="15.9"
                fill="none"
                strokeWidth="3"
                strokeLinecap="round"
                stroke={tone === 'emerald' ? '#059669' : tone === 'amber' ? '#d97706' : '#e11d48'}
                strokeDasharray={`${score} 100`}
              />
            </svg>
            <span className="nums absolute inset-0 flex items-center justify-center text-lg font-bold text-ink">
              {score}
            </span>
          </div>
          <div>
            <p className="text-sm font-semibold text-ink">
              امتیاز سلامت داده: {GRADE_FA[report.grade] ?? report.grade}
            </p>
            <p className="mt-1 text-xs leading-6 text-muted">
              هر یافته بر اساس شدت، از ۱۰۰ کسر می‌شود. یافته‌ها مانع آموزش نمی‌شوند اما باید دیده
              شوند.
            </p>
          </div>
        </div>

        {report.findings.length === 0 ? (
          <EmptyState
            icon={<CheckCircle2 className="h-6 w-6 text-emerald-500" />}
            title="مشکلی شناسایی نشد"
            description="دیتاست برای آموزش آماده است."
          />
        ) : (
          <ul className="space-y-2.5">
            {report.findings.map((finding, index) => (
              <li
                key={`${finding.code}-${index}`}
                className={cn(
                  'insight-row px-3.5 py-3',
                  finding.severity === 'critical' || finding.severity === 'high'
                    ? 'border-rose-200 bg-rose-50/50'
                    : 'border-line/70',
                )}
              >
                <p className="flex flex-wrap items-center gap-2 text-sm font-medium text-ink">
                  {finding.title}
                  <Badge tone={SEVERITY_TONE[finding.severity]}>
                    {SEVERITY_FA[finding.severity]}
                  </Badge>
                </p>
                <p className="mt-1 text-xs leading-6 text-muted">{finding.detail}</p>
                {finding.recommendation ? (
                  <p className="mt-1 text-xs leading-6 text-brand-700">{finding.recommendation}</p>
                ) : null}
              </li>
            ))}
          </ul>
        )}

        {report.adapter_notes?.length ? (
          <div className="mt-4 space-y-1">
            <p className="text-xs font-semibold text-muted">یادداشت‌های پردازش:</p>
            {report.adapter_notes.map((note) => (
              <p key={note} className="text-xs leading-6 text-muted" dir="auto">
                • {note}
              </p>
            ))}
          </div>
        ) : null}
      </CardBody>
    </Card>
  );
}
