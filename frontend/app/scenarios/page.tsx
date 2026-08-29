'use client';

import { useMutation, useQuery } from '@tanstack/react-query';
import { Activity, CalendarDays, ChevronLeft, ChevronRight, FlaskConical, Info, LoaderCircle, Play, RotateCcw, TriangleAlert } from 'lucide-react';
import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { ScenarioChart } from '@/components/charts/scenario-chart';
import { GlassTooltip } from '@/components/charts/glass-tooltip';
import { Card } from '@/components/ui/card';
import { Select } from '@/components/ui/select';
import { NoModelState } from '@/components/ui/states';
import { api } from '@/lib/api/endpoints';
import type { Level, ScenarioAdjustable, ScenarioResult } from '@/lib/types/api';
import { LEVEL_FA, cn } from '@/lib/utils';

type Adjustments = Record<string, number>;
const subtitle = 'اثر تغییر متغیرهای قابل‌برنامه‌ریزی را پیش از اجرا بسنجید؛ مدل ثابت می‌ماند و تنها ورودی‌های آینده تغییر می‌کنند.';
const defaultValue = (item: ScenarioAdjustable) => item.mode === 'absolute' ? (item.current_mean ?? 0) : 0;
const faNumber = new Intl.NumberFormat('fa-IR');
const faDecimal = new Intl.NumberFormat('fa-IR', { maximumFractionDigits: 2 });
const faPercent = new Intl.NumberFormat('fa-IR', { minimumFractionDigits: 1, maximumFractionDigits: 1, signDisplay: 'always' });
const faDate = new Intl.DateTimeFormat('fa-IR-u-ca-gregory', { year: 'numeric', month: '2-digit', day: '2-digit' });
const faPersianDate = new Intl.DateTimeFormat('fa-IR-u-ca-persian', { year: 'numeric', month: 'long', day: 'numeric' });
const persianParts = new Intl.DateTimeFormat('en-US-u-ca-persian', { year: 'numeric', month: 'numeric', day: 'numeric' });
const persianMonth = new Intl.DateTimeFormat('fa-IR-u-ca-persian', { year: 'numeric', month: 'long' });
const number = (value: number) => faNumber.format(value);
const auto = (value: number) => faDecimal.format(value);
const percent = (value: number) => `${faPercent.format(value)}٪`;
const date = (value: string) => value ? faDate.format(new Date(`${value}T00:00:00`)) : '—';
const isoDate = (value: Date) => `${value.getFullYear()}-${String(value.getMonth() + 1).padStart(2, '0')}-${String(value.getDate()).padStart(2, '0')}`;
const addDays = (value: Date, days: number) => { const next = new Date(value); next.setDate(next.getDate() + days); return next; };
const getPersianParts = (value: Date) => Object.fromEntries(persianParts.formatToParts(value).filter((part) => part.type !== 'literal').map((part) => [part.type, Number(part.value)])) as Record<string, number>;
const samePersianMonth = (a: Date, b: Date) => { const first = getPersianParts(a), second = getPersianParts(b); return first.year === second.year && first.month === second.month; };
const monthStart = (value: Date) => { let cursor = value; while (getPersianParts(cursor).day !== 1) cursor = addDays(cursor, -1); return cursor; };
const nextPersianMonth = (value: Date, direction: 1 | -1) => { let cursor = addDays(monthStart(value), direction * 32); while (getPersianParts(cursor).day !== 1) cursor = addDays(cursor, direction); return cursor; };

function Field({ children, modified = false, className }: { children: React.ReactNode; modified?: boolean; className?: string }) {
  return <div className={cn('scenario-field', modified && 'scenario-field--modified', className)}>{children}</div>;
}

export default function ScenariosPage() {
  const optionsQuery = useQuery({ queryKey: ['scenario-options'], queryFn: api.scenarioOptions });
  const options = optionsQuery.data?.data;
  const [level, setLevel] = useState<Level>('destination'); const [entityId, setEntityId] = useState('');
  const [startDate, setStartDate] = useState(''); const [endDate, setEndDate] = useState('');
  const [adjustments, setAdjustments] = useState<Adjustments>({}); const [result, setResult] = useState<ScenarioResult | null>(null);
  const levelId = useId(), destinationId = useId(), startId = useId(), endId = useId();
  const membersQuery = useQuery({ queryKey: ['scenario-members', level], queryFn: () => api.timeseries({ level, horizon: options?.horizon ?? 30 }), enabled: Boolean(options) });
  const members = membersQuery.data?.data?.members ?? [];
  useEffect(() => { if (!options) return; setAdjustments(Object.fromEntries(options.adjustable.map((item) => [item.column, defaultValue(item)]))); setStartDate(options.window.start); setEndDate(options.window.end); }, [options]);
  const simulate = useMutation({
    mutationFn: () => api.simulate({ level, entity_id: entityId || null, start_date: startDate || null, end_date: endDate || null, horizon: options?.horizon,
      adjustments: (options?.adjustable ?? []).map((item) => { const value = adjustments[item.column]; if (value === undefined) return null; if (item.mode === 'absolute') return value === (item.current_mean ?? 0) ? null : { column: item.column, value, mode: 'absolute' as const }; return value === 0 ? null : { column: item.column, change_pct: value, mode: 'relative' as const }; }).filter((item): item is NonNullable<typeof item> => item !== null), }),
    onSuccess: (response) => setResult(response.data),
  });
  const reset = () => { if (!options) return; setAdjustments(Object.fromEntries(options.adjustable.map((item) => [item.column, defaultValue(item)]))); setResult(null); };
  const changed = useMemo(() => (options?.adjustable ?? []).some((item) => adjustments[item.column] !== undefined && adjustments[item.column] !== defaultValue(item)), [adjustments, options]);
  if (optionsQuery.isSuccess && !optionsQuery.data?.available) return <Card><NoModelState detail={optionsQuery.data?.detailFa} /></Card>;

  return <div className="page-stack scenarios-page" dir="rtl">
    <header className="scenario-page-header"><span className="scenario-header-icon"><Activity aria-hidden="true" /></span><div><p className="scenario-eyebrow">آزمایش تصمیم</p><h1>شبیه‌سازی سناریو</h1><p>{subtitle}</p></div></header>
    <div className="scenario-layout">
      <aside className="scenario-settings-card">
        <div className="scenario-settings-head"><span className="scenario-settings-icon"><FlaskConical aria-hidden="true" /></span><div><h2>تنظیمات سناریو</h2>{options ? <p>مدل: <bdi dir="ltr" className="scenario-mono-chip">{options.model}</bdi></p> : null}</div><span className="scenario-info" tabIndex={0} aria-label="توضیح تکمیلی"><Info aria-hidden="true" /><span className="scenario-info__tip"><GlassTooltip>{subtitle}</GlassTooltip></span></span></div>
        {optionsQuery.isLoading ? <div className="scenario-loading" aria-busy="true" /> : null}{optionsQuery.isError ? <p role="alert" className="scenario-error">{(optionsQuery.error as Error).message}</p> : null}
        {options ? <div className="scenario-settings-body">
          <div className="scenario-two-cols"><Select id={levelId} label="سطح" containerClassName="scenario-dashboard-select" className="filter-bar__select" value={level} onChange={(event) => { setLevel(event.target.value as Level); setEntityId(''); }}>{(options.levels ?? ['destination']).map((item) => <option key={item} value={item}>{LEVEL_FA[item] ?? item}</option>)}</Select><Select id={destinationId} label="مقصد" containerClassName="scenario-dashboard-select" className="filter-bar__select" value={entityId} onChange={(event) => setEntityId(event.target.value)}><option value="">همه</option>{members.map((member) => <option key={member.id} value={member.id}>{member.label}</option>)}</Select></div>
          <div className="scenario-two-cols"><DateControl id={startId} label="از تاریخ" value={startDate} min={options.window.start} max={options.window.end} onChange={setStartDate} /><DateControl id={endId} label="تا تاریخ" value={endDate} min={options.window.start} max={options.window.end} onChange={setEndDate} /></div>
          <div className="scenario-controls">{options.adjustable.map((item) => <Adjustment key={item.column} item={item} value={adjustments[item.column] ?? defaultValue(item)} onChange={(value) => setAdjustments((prev) => ({ ...prev, [item.column]: value }))} />)}</div>
          <div className="scenario-actions"><button type="button" className="scenario-reset" onClick={reset} disabled={simulate.isPending} aria-label="بازنشانی سناریو" title="بازنشانی سناریو"><RotateCcw aria-hidden="true" /></button><button type="button" className="scenario-run" onClick={() => simulate.mutate()} disabled={simulate.isPending || !changed} aria-busy={simulate.isPending}>{simulate.isPending ? <LoaderCircle className="scenario-spinner" aria-hidden="true" /> : <Play aria-hidden="true" />}{simulate.isPending ? 'در حال محاسبه...' : 'اجرای سناریو'}</button></div>
          {!changed ? <p className="scenario-helper">حداقل یک متغیر را تغییر دهید تا سناریو معنا پیدا کند.</p> : null}{simulate.isError ? <p role="alert" className="scenario-error">{(simulate.error as Error).message ?? 'اجرای سناریو با خطا مواجه شد.'}</p> : null}
        </div> : null}
      </aside>
      <main className="scenario-results-stack"><section className={cn('scenario-results-card', result && 'scenario-results-card--filled')}><header className="scenario-results-head"><div><h2>پایه در برابر سناریو</h2><p>{result ? `${result.label} · ${date(result.window.start)} تا ${date(result.window.end)}` : 'برای مشاهده نتیجه، سناریو را اجرا کنید'}</p></div>{result ? <span className={cn('scenario-effect', result.impact_pct && result.impact_pct > 0 ? 'scenario-effect--up' : result.impact_pct && result.impact_pct < 0 ? 'scenario-effect--down' : 'scenario-effect--neutral')}>اثر: <bdi dir="ltr">{percent(result.impact_pct ?? 0)}</bdi></span> : null}</header>{result ? <div className="scenario-result-content"><div className="scenario-stats"><Stat label="تقاضای پایه" value={number(result.baseline_total)} /><Stat label="تقاضای سناریو" value={number(result.scenario_total)} scenario /><Stat label="اختلاف" value={`${result.delta_total >= 0 ? '▲ ' : '▼ '}${number(Math.abs(result.delta_total))}`} delta={result.delta_total} /></div><ScenarioChart series={result.series} /><p className="scenario-caution"><TriangleAlert aria-hidden="true" />{result.note_fa}</p></div> : <div className="scenario-empty"><div className="scenario-empty-tile"><FlaskConical aria-hidden="true" /></div><h3>هنوز سناریویی اجرا نشده</h3><p>متغیرها را در پنل کناری تغییر دهید و «اجرای سناریو» را بزنید.</p></div>}</section>{result && (result.applied.length > 0 || result.rejected.length > 0) ? <section className="scenario-applied-card"><h2>تغییرات اعمال‌شده</h2><div>{result.applied.map((item, index) => <div key={item.column} className="scenario-applied-row" style={{ '--row-delay': `${index * 50}ms` } as React.CSSProperties}><span>{item.label_fa}</span><bdi dir="ltr"><strong>{auto(item.mean_after)}</strong><i>←</i><em>{auto(item.mean_before)}</em><small className={item.mean_after >= item.mean_before ? 'scenario-delta--up' : 'scenario-delta--down'}>{item.change}</small></bdi></div>)}{result.rejected.map((item) => <div key={item.column} className="scenario-rejected"><bdi dir="ltr">{item.column}</bdi> — {item.reason}</div>)}</div></section> : null}</main>
    </div>
  </div>;
}

function Adjustment({ item, value, onChange }: { item: ScenarioAdjustable; value: number; onChange: (value: number) => void }) {
  const id = useId(), modified = value !== defaultValue(item), binary = item.mode === 'absolute' || item.is_binary;
  if (binary) { const on = value >= .5; return <Field modified={modified} className="scenario-toggle-field"><div className="scenario-control-label"><label htmlFor={id}>{item.label_fa}</label><bdi dir="ltr" className="scenario-mono-chip">{item.column}</bdi></div><button id={id} type="button" role="switch" aria-checked={on} onClick={() => onChange(on ? 0 : 1)} className={cn('scenario-switch', on && 'scenario-switch--on')}><span /></button></Field>; }
  return <SliderControl id={id} item={item} value={value} modified={modified} onChange={onChange} />;
}
function DateControl({ id, label, value, min, max, onChange }: { id: string; label: string; value: string; min: string; max: string; onChange: (value: string) => void }) {
  const [open, setOpen] = useState(false); const selected = value ? new Date(`${value}T00:00:00`) : new Date(); const [view, setView] = useState(selected);
  useEffect(() => { if (!open) setView(selected); }, [value, open]);
  const first = monthStart(view); const gridStart = addDays(first, -((first.getDay() + 1) % 7));
  const days = Array.from({ length: 42 }, (_, index) => addDays(gridStart, index));
  const choose = (day: Date) => { const next = isoDate(day); if (next >= min && next <= max) onChange(next); setOpen(false); };
  return <Field className="scenario-date-field"><label htmlFor={id}>{label}</label><button id={id} type="button" aria-haspopup="dialog" aria-expanded={open} className="scenario-date-display" onClick={() => setOpen((current) => !current)}><span>{value ? faPersianDate.format(selected) : '—'}</span><CalendarDays aria-hidden="true" /></button>{open ? <div className="scenario-jalali-picker" role="dialog" aria-label={label} onKeyDown={(event) => { if (event.key === 'Escape') setOpen(false); }}><div className="scenario-jalali-picker__head"><button type="button" aria-label="ماه بعد" onClick={() => setView((current) => nextPersianMonth(current, 1))}><ChevronRight aria-hidden="true" /></button><strong>{persianMonth.format(first)}</strong><button type="button" aria-label="ماه قبل" onClick={() => setView((current) => nextPersianMonth(current, -1))}><ChevronLeft aria-hidden="true" /></button></div><div className="scenario-jalali-picker__week" aria-hidden="true"><span>ش</span><span>ی</span><span>د</span><span>س</span><span>چ</span><span>پ</span><span>ج</span></div><div className="scenario-jalali-picker__days">{days.map((day) => { const iso = isoDate(day), parts = getPersianParts(day), inMonth = samePersianMonth(day, first), disabled = iso < min || iso > max; return <button key={iso} type="button" disabled={disabled} aria-label={faPersianDate.format(day)} aria-pressed={iso === value} onClick={() => choose(day)} className={cn(!inMonth && 'scenario-jalali-picker__outside', iso === value && 'scenario-jalali-picker__selected')}>{faNumber.format(parts.day)}</button>; })}</div></div> : null}</Field>;
}
function SliderControl({ id, item, value, modified, onChange }: { id: string; item: ScenarioAdjustable; value: number; modified: boolean; onChange: (value: number) => void }) {
  const inputRef = useRef<HTMLInputElement>(null); const frame = useRef<number | null>(null); const latest = useRef(value);
  const paint = (next: number) => { const pos = (next + 40) / 80; const zero = .5; const start = Math.min(zero, pos) * 100; const width = Math.abs(pos - zero) * 100; const track = inputRef.current?.parentElement; if (!track) return; inputRef.current?.setAttribute('aria-valuetext', percent(next)); track.style.setProperty('--fill-start', `${start}%`); track.style.setProperty('--fill-width', `${width}%`); track.dataset.tone = next > 0 ? 'up' : next < 0 ? 'down' : 'neutral'; const badge = track.parentElement?.querySelector<HTMLElement>('.scenario-value-badge'); if (badge) { badge.textContent = percent(next); badge.className = `scenario-value-badge scenario-value-badge--${next > 0 ? 'up' : next < 0 ? 'down' : 'neutral'}`; } };
  useEffect(() => { if (inputRef.current && Number(inputRef.current.value) !== value) inputRef.current.value = String(value); latest.current = value; paint(value); }, [value]);
  useEffect(() => () => { if (frame.current) cancelAnimationFrame(frame.current); }, []);
  const onInput = (event: React.FormEvent<HTMLInputElement>) => { latest.current = Number(event.currentTarget.value); if (frame.current) return; frame.current = requestAnimationFrame(() => { frame.current = null; paint(latest.current); }); };
  const commit = () => onChange(latest.current);
  const tone = value > 0 ? 'up' : value < 0 ? 'down' : 'neutral';
  return <Field modified={modified} className="scenario-slider-field"><div className="scenario-slider-head"><div className="scenario-control-label"><label htmlFor={id}>{item.label_fa}</label><bdi dir="ltr" className="scenario-mono-chip">{item.column}</bdi></div><bdi dir="ltr" className={cn('scenario-value-badge', `scenario-value-badge--${tone}`)}>{percent(value)}</bdi></div><div className="scenario-range-track" data-tone={tone} style={{ '--fill-start': `${Math.min(.5, (value + 40) / 80) * 100}%`, '--fill-width': `${Math.abs((value + 40) / 80 - .5) * 100}%` } as React.CSSProperties}><i className="scenario-range-fill" /><input ref={inputRef} id={id} type="range" min={-40} max={40} step={5} defaultValue={value} aria-valuetext={percent(value)} onInput={onInput} onPointerUp={commit} onBlur={commit} onKeyUp={commit} className="scenario-range" /></div></Field>;
}
function Stat({ label, value, scenario = false, delta }: { label: string; value: string; scenario?: boolean; delta?: number }) { return <div className={cn('scenario-stat', scenario && 'scenario-stat--scenario', delta !== undefined && (delta >= 0 ? 'scenario-stat--up' : 'scenario-stat--down'))}><p>{label}</p><bdi dir="ltr">{value}</bdi></div>; }
