'use client';

import { useMutation, useQuery } from '@tanstack/react-query';
import { FlaskConical, Play, RotateCcw, TriangleAlert } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

import { ScenarioChart } from '@/components/charts/scenario-chart';
import { Badge, toneForChange } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardBody, CardHeader } from '@/components/ui/card';
import { Select, TextInput } from '@/components/ui/select';
import { AsyncBoundary, CardSkeleton, EmptyState, NoModelState } from '@/components/ui/states';
import { InfoHint } from '@/components/ui/tooltip';
import { api } from '@/lib/api/endpoints';
import type { Level, ScenarioAdjustable, ScenarioResult } from '@/lib/types/api';
import { LEVEL_FA, cn, formatAuto, formatNumber, formatPercent } from '@/lib/utils';

type Adjustments = Record<string, number>;

function defaultValue(item: ScenarioAdjustable): number {
  return item.mode === 'absolute' ? (item.current_mean ?? 0) : 0;
}

export default function ScenariosPage() {
  const optionsQuery = useQuery({ queryKey: ['scenario-options'], queryFn: api.scenarioOptions });
  const options = optionsQuery.data?.data;

  const [level, setLevel] = useState<Level>('destination');
  const [entityId, setEntityId] = useState<string>('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [adjustments, setAdjustments] = useState<Adjustments>({});
  const [result, setResult] = useState<ScenarioResult | null>(null);

  const membersQuery = useQuery({
    queryKey: ['scenario-members', level],
    queryFn: () => api.timeseries({ level, horizon: options?.horizon ?? 30 }),
    enabled: Boolean(options),
  });
  const members = membersQuery.data?.data?.members ?? [];

  useEffect(() => {
    if (!options) return;
    setAdjustments(
      Object.fromEntries(options.adjustable.map((item) => [item.column, defaultValue(item)])),
    );
    setStartDate(options.window.start);
    setEndDate(options.window.end);
  }, [options]);

  const simulate = useMutation({
    mutationFn: () =>
      api.simulate({
        level,
        entity_id: entityId || null,
        start_date: startDate || null,
        end_date: endDate || null,
        horizon: options?.horizon,
        adjustments: (options?.adjustable ?? [])
          .map((item) => {
            const value = adjustments[item.column];
            if (value === undefined) return null;
            if (item.mode === 'absolute') {
              if (value === (item.current_mean ?? 0)) return null;
              return { column: item.column, value, mode: 'absolute' };
            }
            if (value === 0) return null;
            return { column: item.column, change_pct: value, mode: 'relative' };
          })
          .filter((item): item is NonNullable<typeof item> => item !== null),
      }),
    onSuccess: (response) => setResult(response.data),
  });

  const reset = () => {
    if (!options) return;
    setAdjustments(
      Object.fromEntries(options.adjustable.map((item) => [item.column, defaultValue(item)])),
    );
    setResult(null);
  };

  const changed = useMemo(
    () =>
      (options?.adjustable ?? []).some(
        (item) => adjustments[item.column] !== undefined && adjustments[item.column] !== defaultValue(item),
      ),
    [adjustments, options],
  );

  if (optionsQuery.isSuccess && !optionsQuery.data?.available) {
    return (
      <Card>
        <NoModelState detail={optionsQuery.data?.detailFa} />
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="section-title">شبیه‌سازی سناریو</h2>
        <p className="mt-1 text-sm text-muted">
          تغییر متغیرهای قابل‌برنامه‌ریزی آینده و مشاهده واکنش همان مدل آموزش‌دیده. مدل دوباره آموزش
          نمی‌بیند؛ تنها ورودی‌های آینده تغییر می‌کنند.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[380px_1fr]">
        {/* ----------------------------------------------------- controls */}
        <Card className="h-fit">
          <CardHeader
            icon={<FlaskConical className="h-4.5 w-4.5" />}
            title="تنظیمات سناریو"
            subtitle={options ? `مدل: ${options.model}` : undefined}
            action={
              <InfoHint>
                تنها متغیرهایی نمایش داده می‌شوند که مدل واقعاً از آن‌ها استفاده می‌کند. تغییر یک
                متغیر بی‌اثر، پیش‌بینی را تغییر نمی‌دهد و بنابراین اینجا فهرست نمی‌شود.
              </InfoHint>
            }
          />
          <AsyncBoundary
            isLoading={optionsQuery.isLoading}
            error={optionsQuery.error}
            onRetry={() => optionsQuery.refetch()}
            skeleton={<CardSkeleton lines={7} />}
          >
            <CardBody className="space-y-4">
              <div className="grid gap-3 sm:grid-cols-2">
                <Select
                  label="سطح"
                  value={level}
                  onChange={(event) => {
                    setLevel(event.target.value as Level);
                    setEntityId('');
                  }}
                >
                  {(options?.levels ?? ['destination']).map((item) => (
                    <option key={item} value={item}>
                      {LEVEL_FA[item] ?? item}
                    </option>
                  ))}
                </Select>
                <Select
                  label={LEVEL_FA[level] ?? 'انتخاب'}
                  value={entityId}
                  onChange={(event) => setEntityId(event.target.value)}
                >
                  <option value="">همه</option>
                  {members.map((member) => (
                    <option key={member.id} value={member.id}>
                      {member.label}
                    </option>
                  ))}
                </Select>
              </div>

              <div className="grid gap-3 sm:grid-cols-2">
                <TextInput
                  label="از تاریخ"
                  type="date"
                  value={startDate}
                  min={options?.window.start}
                  max={options?.window.end}
                  onChange={(event) => setStartDate(event.target.value)}
                />
                <TextInput
                  label="تا تاریخ"
                  type="date"
                  value={endDate}
                  min={options?.window.start}
                  max={options?.window.end}
                  onChange={(event) => setEndDate(event.target.value)}
                />
              </div>

              <div className="space-y-4 border-t border-line pt-4">
                {(options?.adjustable ?? []).length === 0 ? (
                  <EmptyState
                    title="متغیر قابل تنظیمی وجود ندارد"
                    description="این دیتاست متغیر آینده‌ای که مدل از آن استفاده کند ندارد."
                  />
                ) : null}

                {(options?.adjustable ?? []).map((item) => {
                  const value = adjustments[item.column] ?? defaultValue(item);
                  if (item.mode === 'absolute' || item.is_binary) {
                    const on = value >= 0.5;
                    return (
                      <div key={item.column} className="flex items-center justify-between gap-3">
                        <span className="text-sm text-ink">
                          {item.label_fa}
                          <span className="debug-only mr-1.5 text-[11px] text-muted" dir="ltr">
                            {item.column}
                          </span>
                        </span>
                        <button
                          type="button"
                          role="switch"
                          aria-checked={on}
                          onClick={() =>
                            setAdjustments((prev) => ({ ...prev, [item.column]: on ? 0 : 1 }))
                          }
                          className={cn(
                            'relative h-6 w-11 shrink-0 rounded-full transition',
                            on ? 'bg-brand-600' : 'bg-slate-300',
                          )}
                        >
                          <span
                            className={cn(
                              'absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition',
                              on ? 'right-0.5' : 'right-[22px]',
                            )}
                          />
                        </button>
                      </div>
                    );
                  }
                  return (
                    <div key={item.column}>
                      <div className="flex items-baseline justify-between">
                        <span className="text-sm text-ink">
                          {item.label_fa}
                          <span className="debug-only mr-1.5 text-[11px] text-muted" dir="ltr">
                            {item.column}
                          </span>
                        </span>
                        <span
                          className={cn(
                            'nums text-sm font-semibold',
                            value > 0 && 'text-emerald-600',
                            value < 0 && 'text-rose-600',
                            value === 0 && 'text-muted',
                          )}
                        >
                          {value > 0 ? '+' : ''}
                          {value}٪
                        </span>
                      </div>
                      <input
                        type="range"
                        min={-40}
                        max={40}
                        step={5}
                        value={value}
                        onChange={(event) =>
                          setAdjustments((prev) => ({
                            ...prev,
                            [item.column]: Number(event.target.value),
                          }))
                        }
                        className="mt-2 h-1.5 w-full cursor-pointer appearance-none rounded-full bg-slate-200 accent-brand-600"
                      />
                    </div>
                  );
                })}
              </div>

              <div className="flex gap-2 border-t border-line pt-4">
                <Button
                  onClick={() => simulate.mutate()}
                  disabled={simulate.isPending || !changed}
                  className="flex-1"
                >
                  <Play className="h-4 w-4" />
                  {simulate.isPending ? 'در حال محاسبه...' : 'اجرای سناریو'}
                </Button>
                <Button variant="secondary" onClick={reset} disabled={simulate.isPending}>
                  <RotateCcw className="h-4 w-4" />
                </Button>
              </div>
              {!changed ? (
                <p className="text-xs text-muted">
                  حداقل یک متغیر را تغییر دهید تا سناریو معنا پیدا کند.
                </p>
              ) : null}
              {simulate.isError ? (
                <p className="text-xs text-rose-600">
                  {(simulate.error as Error).message ?? 'اجرای سناریو با خطا مواجه شد.'}
                </p>
              ) : null}
            </CardBody>
          </AsyncBoundary>
        </Card>

        {/* ------------------------------------------------------ results */}
        <div className="space-y-6">
          <Card>
            <CardHeader
              title="پایه در برابر سناریو"
              subtitle={
                result
                  ? `${result.label} · ${result.window.start} تا ${result.window.end}`
                  : 'برای مشاهده نتیجه، سناریو را اجرا کنید'
              }
              action={
                result ? (
                  <Badge tone={toneForChange(result.impact_pct)}>
                    اثر: {formatPercent(result.impact_pct)}
                  </Badge>
                ) : null
              }
            />
            <CardBody>
              {result ? (
                <>
                  <div className="mb-5 grid gap-4 sm:grid-cols-3">
                    <div className="rounded-xl border border-line/70 p-4">
                      <p className="text-xs text-muted">تقاضای پایه</p>
                      <p className="nums mt-1.5 text-2xl font-semibold text-ink">
                        {formatNumber(result.baseline_total)}
                      </p>
                    </div>
                    <div className="rounded-xl border border-brand-200 bg-brand-50/50 p-4">
                      <p className="text-xs text-brand-700">تقاضای سناریو</p>
                      <p className="nums mt-1.5 text-2xl font-semibold text-brand-700">
                        {formatNumber(result.scenario_total)}
                      </p>
                    </div>
                    <div className="rounded-xl border border-line/70 p-4">
                      <p className="text-xs text-muted">اختلاف</p>
                      <p
                        className={cn(
                          'nums mt-1.5 text-2xl font-semibold',
                          result.delta_total >= 0 ? 'text-emerald-600' : 'text-rose-600',
                        )}
                      >
                        {result.delta_total >= 0 ? '+' : ''}
                        {formatNumber(result.delta_total)}
                      </p>
                    </div>
                  </div>
                  <ScenarioChart series={result.series} />
                  <p className="mt-4 flex items-start gap-2 rounded-xl bg-amber-50 px-3.5 py-2.5 text-xs leading-6 text-amber-800">
                    <TriangleAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                    {result.note_fa}
                  </p>
                </>
              ) : (
                <EmptyState
                  icon={<FlaskConical className="h-6 w-6" />}
                  title="هنوز سناریویی اجرا نشده"
                  description="متغیرها را در پنل کناری تغییر دهید و «اجرای سناریو» را بزنید."
                />
              )}
            </CardBody>
          </Card>

          {result && (result.applied.length > 0 || result.rejected.length > 0) ? (
            <Card>
              <CardHeader title="تغییرات اعمال‌شده" />
              <CardBody className="space-y-2">
                {result.applied.map((item) => (
                  <div
                    key={item.column}
                    className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-line/70 px-3.5 py-2.5 text-sm"
                  >
                    <span className="text-ink">{item.label_fa}</span>
                    <span className="nums text-xs text-muted" dir="ltr">
                      {formatAuto(item.mean_before)} → {formatAuto(item.mean_after)} ({item.change})
                    </span>
                  </div>
                ))}
                {result.rejected.map((item) => (
                  <div
                    key={item.column}
                    className="rounded-xl border border-amber-200 bg-amber-50 px-3.5 py-2.5 text-xs text-amber-800"
                  >
                    <span dir="ltr" className="font-mono">
                      {item.column}
                    </span>{' '}
                    — {item.reason}
                  </div>
                ))}
              </CardBody>
            </Card>
          ) : null}
        </div>
      </div>
    </div>
  );
}
