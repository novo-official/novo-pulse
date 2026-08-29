'use client';

import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import { Map } from 'lucide-react';
import { toneForChange, toneForConfidence } from '@/components/ui/badge';
import { GlassTooltip } from '@/components/charts/glass-tooltip';
import { Card, CardBody, CardHeader } from '@/components/ui/card';
import { EmptyState } from '@/components/ui/states';
import type { OverviewRow } from '@/lib/types/api';
import { cn, CONFIDENCE_FA } from '@/lib/utils';

const STATUS_TONE: Record<string, 'success' | 'danger' | 'warning' | 'neutral'> = { growing: 'success', declining: 'danger', spike_risk: 'warning', stable: 'neutral', unknown: 'neutral' };
const faNumber = new Intl.NumberFormat('fa-IR', { maximumFractionDigits: 0 });
const faPercent = new Intl.NumberFormat('fa-IR', { minimumFractionDigits: 1, maximumFractionDigits: 1, signDisplay: 'always' });
const showNumber = (value: number | null | undefined) => value === null || value === undefined || !Number.isFinite(value) ? '—' : faNumber.format(value);
const showPercent = (value: number | null | undefined) => value === null || value === undefined || !Number.isFinite(value) ? '—' : `${faPercent.format(value)}٪`;

function DestinationAvatar({ label }: { label: string }) { return <span aria-hidden="true" className="destination-table__avatar">{label.trim().slice(0, 1)}</span>; }
function DeltaPill({ value }: { value: number | null | undefined }) { const tone = toneForChange(value); return <bdi dir="ltr" className={cn('destination-table__delta', `destination-table__delta--${tone}`)}>{showPercent(value)}</bdi>; }
function LevelPill({ confidence }: { confidence: string | null | undefined }) { const tone = toneForConfidence(confidence); return <span className={cn('destination-table__level', `destination-table__level--${tone}`, confidence && `destination-table__level--${confidence}`)}><i />{CONFIDENCE_FA[confidence ?? ''] ?? confidence}</span>; }
function StatusPill({ status, label }: { status: string; label: string }) { return <span className={cn('destination-table__status', `destination-table__status--${STATUS_TONE[status] ?? 'neutral'}`, status === 'stable' && 'destination-table__status--stable')}><i />{label}</span>; }

type TooltipEvents = { onMouseEnter: (event: React.MouseEvent<HTMLElement>) => void; onMouseLeave: () => void; onFocus: (event: React.FocusEvent<HTMLElement>) => void; onBlur: () => void; onKeyDown: (event: React.KeyboardEvent<HTMLElement>) => void };
function RangeCell({ low, high, tooltip }: { low: number | null | undefined; high: number | null | undefined; tooltip: TooltipEvents }) {
  return <span className="destination-table__tooltip-target" tabIndex={0} {...tooltip}><bdi dir="ltr" className="destination-table__range">{showNumber(low)} – {showNumber(high)}</bdi></span>;
}
function NumericCell({ current, forecast, tooltip }: { current: number | null | undefined; forecast: number | null | undefined; tooltip: TooltipEvents }) {
  return <span className="destination-table__tooltip-target" tabIndex={0} {...tooltip}><bdi dir="ltr" className="destination-table__number">{showNumber(current)}</bdi></span>;
}

export function OverviewTable({ rows, onSelect, title = 'نمای کلی — مقصد' }: { rows: OverviewRow[]; onSelect?: (id: string) => void; title?: string }) {
  const sentinelRef = useRef<HTMLDivElement>(null);
  const tooltipFrame = useRef<number | null>(null);
  const [headerStuck, setHeaderStuck] = useState(false);
  const [tooltip, setTooltip] = useState<{ content: ReactNode; x: number; y: number } | null>(null);
  useEffect(() => { const sentinel = sentinelRef.current; if (!sentinel || !('IntersectionObserver' in window)) return; const observer = new IntersectionObserver(([entry]) => setHeaderStuck(!entry.isIntersecting), { threshold: 0 }); observer.observe(sentinel); return () => observer.disconnect(); }, []);
  const hideTooltip = useCallback(() => { if (tooltipFrame.current) cancelAnimationFrame(tooltipFrame.current); setTooltip(null); }, []);
  const showTooltip = useCallback((event: React.MouseEvent<HTMLElement> | React.FocusEvent<HTMLElement>, content: ReactNode) => {
    const point = 'clientX' in event && event.type === 'mouseenter' ? { x: event.clientX, y: event.clientY } : (() => { const rect = event.currentTarget.getBoundingClientRect(); return { x: rect.left + rect.width / 2, y: rect.top }; })();
    if (tooltipFrame.current) cancelAnimationFrame(tooltipFrame.current);
    tooltipFrame.current = requestAnimationFrame(() => setTooltip({ content, x: Math.max(12, Math.min(point.x + 12, window.innerWidth - 252)), y: Math.max(12, point.y - 12) }));
  }, []);
  useEffect(() => { const dismiss = () => hideTooltip(); window.addEventListener('resize', dismiss); window.addEventListener('scroll', dismiss, true); return () => { window.removeEventListener('resize', dismiss); window.removeEventListener('scroll', dismiss, true); }; }, [hideTooltip]);
  const selectKey = (event: React.KeyboardEvent<HTMLElement>, id: string) => { if (onSelect && (event.key === 'Enter' || event.key === ' ')) { event.preventDefault(); onSelect(id); } };
  const rangeTooltip = (low: number | null | undefined, high: number | null | undefined) => <><p className="destination-table__tooltip-title">بازه اطمینان ۸۰٪</p><dl><div><dt><bdi dir="ltr">{showNumber(low)}</bdi></dt><dd>–</dd></div><div><dt><bdi dir="ltr">{showNumber(high)}</bdi></dt></div></dl></>;
  const pairTooltip = (current: number | null | undefined, forecast: number | null | undefined) => <dl><div><dt>تقاضای فعلی</dt><dd className="destination-table__tip-strong">{showNumber(current)}</dd></div><div><dt>پیش‌بینی</dt><dd>{showNumber(forecast)}</dd></div></dl>;
  const tooltipEvents = (content: ReactNode): TooltipEvents => ({ onMouseEnter: (event) => showTooltip(event, content), onMouseLeave: hideTooltip, onFocus: (event) => showTooltip(event, content), onBlur: hideTooltip, onKeyDown: (event) => { if (event.key === 'Escape') { event.currentTarget.blur(); hideTooltip(); } } });
  return <Card className="destination-table-card overflow-hidden"><CardHeader icon={<Map className="h-[19px] w-[19px]" />} title={title} subtitle="تقاضای فعلی، پیش‌بینی، تغییر و وضعیت هر مقصد" /><CardBody className="destination-table-card__body">{rows.length === 0 ? <EmptyState title="مقصدی برای نمایش نیست" /> : <>
    <div ref={sentinelRef} aria-hidden="true" className="destination-table__sentinel" />
    <div className="destination-table__scroll"><table className="destination-table" dir="rtl"><thead className={cn('destination-table__header', headerStuck && 'destination-table__header--stuck')}><tr><th scope="col">مقصد</th><th scope="col">تقاضای فعلی</th><th scope="col">پیش‌بینی</th><th scope="col">تغییر</th><th scope="col">بازه اطمینان</th><th scope="col">اطمینان</th><th scope="col">وضعیت</th></tr></thead><tbody>{rows.map((row, index) => <tr key={row.entity_id} style={{ animationDelay: `${index * 35}ms` }} onClick={() => onSelect?.(row.entity_id)} tabIndex={onSelect ? 0 : undefined} onKeyDown={(event) => selectKey(event, row.entity_id)} className={onSelect ? 'destination-table__selectable' : undefined}><td><span className="destination-table__city"><DestinationAvatar label={row.label} /><span>{row.label}</span></span></td><td><NumericCell current={row.current_demand} forecast={row.forecast_total} tooltip={tooltipEvents(pairTooltip(row.current_demand, row.forecast_total))} /></td><td><bdi dir="ltr" className="destination-table__number destination-table__number--muted">{showNumber(row.forecast_total)}</bdi></td><td><DeltaPill value={row.change_pct} /></td><td><RangeCell low={row.lower_total} high={row.upper_total} tooltip={tooltipEvents(rangeTooltip(row.lower_total, row.upper_total))} /></td><td><LevelPill confidence={row.confidence} /></td><td><StatusPill status={row.status} label={row.status_fa} /></td></tr>)}</tbody></table></div>
    <div className="destination-table__mobile">{rows.map((row) => <article key={row.entity_id} className="destination-table__mobile-card" onClick={() => onSelect?.(row.entity_id)} tabIndex={onSelect ? 0 : undefined} onKeyDown={(event) => selectKey(event, row.entity_id)}><header><span className="destination-table__city"><DestinationAvatar label={row.label} /><span>{row.label}</span></span><DeltaPill value={row.change_pct} /></header><dl><div><dt>تقاضای فعلی</dt><dd><bdi dir="ltr" className="destination-table__number">{showNumber(row.current_demand)}</bdi></dd></div><div><dt>پیش‌بینی</dt><dd><bdi dir="ltr" className="destination-table__number destination-table__number--muted">{showNumber(row.forecast_total)}</bdi></dd></div><div><dt>تغییر</dt><dd><DeltaPill value={row.change_pct} /></dd></div><div><dt>بازه اطمینان</dt><dd><RangeCell low={row.lower_total} high={row.upper_total} tooltip={tooltipEvents(rangeTooltip(row.lower_total, row.upper_total))} /></dd></div><div><dt>اطمینان</dt><dd><LevelPill confidence={row.confidence} /></dd></div><div><dt>وضعیت</dt><dd><StatusPill status={row.status} label={row.status_fa} /></dd></div></dl></article>)}</div>
    {tooltip ? <div aria-hidden="true" className="destination-table__floating-tooltip" style={{ transform: `translate3d(${tooltip.x}px, ${tooltip.y}px, 0) translateY(-100%)` }}><GlassTooltip>{tooltip.content}</GlassTooltip></div> : null}
  </>}</CardBody></Card>;
}
