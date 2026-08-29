'use client';

import { useCallback, useEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react';
import { Grid3x3, Info } from 'lucide-react';
import { GlassTooltip } from '@/components/charts/glass-tooltip';
import { Card, CardBody, CardHeader } from '@/components/ui/card';
import { EmptyState } from '@/components/ui/states';
import type { HeatmapResponse } from '@/lib/types/api';

const day = new Intl.DateTimeFormat('fa-IR-u-ca-gregory', { day: 'numeric' });
const fullDate = new Intl.DateTimeFormat('fa-IR-u-ca-gregory', { weekday: 'long', day: 'numeric', month: 'long' });
const number = new Intl.NumberFormat('fa-IR', { maximumFractionDigits: 1 });
type Tip = { content: ReactNode; x: number; y: number };

function heatStyle(intensity: number | null): CSSProperties {
  if (intensity === null || !Number.isFinite(intensity) || intensity === 0) return { '--heat-color': 'var(--heat-neutral)' } as CSSProperties;
  const step = Math.min(3, Math.max(0, Math.round((Math.min(.6, Math.abs(intensity)) / .6) * 3)));
  return { '--heat-color': `var(--heat-${intensity > 0 ? 'up' : 'down'}-${step})` } as CSSProperties;
}
function signal(intensity: number | null) { return (intensity ?? 0) >= 0 ? 'بیشتر از معمول' : 'کمتر از معمول'; }

export function HeatLegend({ show, hide }: { show: (event: React.MouseEvent<HTMLElement> | React.FocusEvent<HTMLElement>, content: ReactNode) => void; hide: () => void }) {
  const eventProps = (text: string) => ({ onMouseEnter: (event: React.MouseEvent<HTMLElement>) => show(event, <p className="heatmap-tooltip__single">{text}</p>), onMouseLeave: hide, onFocus: (event: React.FocusEvent<HTMLElement>) => show(event, <p className="heatmap-tooltip__single">{text}</p>), onBlur: hide });
  return <div className="heatmap-legend"><span {...eventProps('کمتر از معمول')} tabIndex={0}>کمتر از معمول</span><div className="heatmap-legend__bar" aria-hidden="true"><i {...eventProps('کمتر از معمول')} /><i {...eventProps('بیشتر از معمول')} /></div><span {...eventProps('بیشتر از معمول')} tabIndex={0}>بیشتر از معمول</span></div>;
}

export function HeatGrid({ data, show, hide }: { data: HeatmapResponse; show: (event: React.MouseEvent<HTMLElement> | React.FocusEvent<HTMLElement>, content: ReactNode) => void; hide: () => void }) {
  const { dates, rows } = data;
  const step = Math.max(1, Math.ceil(dates.length / 12));
  const [active, setActive] = useState(0);
  const [hovered, setHovered] = useState<{ row: number; col: number } | null>(null);
  const cells = rows.flatMap((row, rowIndex) => row.values.map((value, colIndex) => ({ row, rowIndex, colIndex, value, intensity: row.intensity[colIndex] ?? null })));
  const move = (index: number, key: string) => {
    const cell = cells[index]; if (!cell) return;
    let next = index;
    if (key === 'ArrowLeft') next = Math.min(cells.length - 1, index + 1);
    if (key === 'ArrowRight') next = Math.max(0, index - 1);
    if (key === 'ArrowDown') next = Math.min(cells.length - 1, index + dates.length);
    if (key === 'ArrowUp') next = Math.max(0, index - dates.length);
    if (key === 'Home') next = cell.rowIndex * dates.length;
    if (key === 'End') next = cell.rowIndex * dates.length + dates.length - 1;
    if (next !== index) { setActive(next); requestAnimationFrame(() => document.querySelector<HTMLElement>(`[data-heat-index="${next}"]`)?.focus()); }
  };
  return <div className="heatmap-panel"><div className="heatmap-scroll"><div className="heatmap-grid" role="grid" style={{ gridTemplateColumns: `110px repeat(${dates.length}, minmax(24px, 1fr))` }}>
    <span className="heatmap-grid__corner" />
    {dates.map((date, index) => <span key={date} data-heat-tick={index} className={hovered?.col === index ? 'heatmap-tick heatmap-tick--active' : 'heatmap-tick'} title={date}>{index % step === 0 ? day.format(new Date(date)) : ''}</span>)}
    {rows.map((row, rowIndex) => <div key={row.entity_id} role="row" className="heatmap-row"><span className={hovered?.row === rowIndex ? 'heatmap-city heatmap-city--active' : 'heatmap-city'} title={row.label}>{row.label}</span>{row.values.map((value, colIndex) => {
      const index = rowIndex * dates.length + colIndex; const intensity = row.intensity[colIndex] ?? null; const label = signal(intensity); const tip = <><p className="heatmap-tooltip__title">{row.label}</p><p className="heatmap-tooltip__date">{fullDate.format(new Date(dates[colIndex]))}</p><div className="heatmap-tooltip__row"><i style={heatStyle(intensity)} /><span>{label}</span></div>{value !== null ? <strong>{number.format(value)}</strong> : null}</>;
      const activeAxis = !hovered || hovered.row === rowIndex || hovered.col === colIndex;
      return <span key={`${row.entity_id}-${dates[colIndex]}`} data-heat-index={index} data-row={rowIndex} data-col={colIndex} role="gridcell" tabIndex={active === index ? 0 : -1} aria-label={`${row.label}، ${fullDate.format(new Date(dates[colIndex]))}، ${label}`} className={activeAxis ? 'heatmap-cell' : 'heatmap-cell heatmap-cell--dim'} style={{ ...heatStyle(intensity), animationDelay: `${rowIndex * 28 + colIndex * 6}ms` }} onMouseEnter={(event) => { setHovered({ row: rowIndex, col: colIndex }); show(event, tip); }} onMouseLeave={() => { setHovered(null); hide(); }} onFocus={(event) => { setHovered({ row: rowIndex, col: colIndex }); show(event, tip); }} onBlur={() => { setHovered(null); hide(); }} onKeyDown={(event) => { if (event.key === 'Escape') { event.currentTarget.blur(); hide(); return; } if (['ArrowLeft', 'ArrowRight', 'ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) { event.preventDefault(); move(index, event.key); } }} />;
    })}</div>)}
  </div></div></div>;
}

export function HeatmapCard({ data }: { data: HeatmapResponse }) {
  const frame = useRef<number | null>(null); const [tip, setTip] = useState<Tip | null>(null);
  const hide = useCallback(() => { if (frame.current) cancelAnimationFrame(frame.current); setTip(null); }, []);
  const show = useCallback((event: React.MouseEvent<HTMLElement> | React.FocusEvent<HTMLElement>, content: ReactNode) => { const point = 'clientX' in event && event.type === 'mouseenter' ? { x: event.clientX, y: event.clientY } : (() => { const rect = event.currentTarget.getBoundingClientRect(); return { x: rect.left + rect.width / 2, y: rect.top }; })(); if (frame.current) cancelAnimationFrame(frame.current); frame.current = requestAnimationFrame(() => setTip({ content, x: Math.max(12, Math.min(point.x + 12, window.innerWidth - 252)), y: Math.max(12, point.y - 12) })); }, []);
  useEffect(() => { const dismiss = () => hide(); window.addEventListener('resize', dismiss); window.addEventListener('scroll', dismiss, true); return () => { window.removeEventListener('resize', dismiss); window.removeEventListener('scroll', dismiss, true); }; }, [hide]);
  const subtitle = 'مقصد × تاریخ — برای شناسایی سریع دوره‌های اوج';
  return <Card className="heatmap-card overflow-hidden"><CardHeader icon={<Grid3x3 className="h-[19px] w-[19px]" />} title="نقشه حرارتی تقاضا" subtitle={subtitle} action={<button type="button" aria-label={subtitle} className="heatmap-info" onMouseEnter={(event) => show(event, <p className="heatmap-tooltip__single">{subtitle}</p>)} onMouseLeave={hide} onFocus={(event) => show(event, <p className="heatmap-tooltip__single">{subtitle}</p>)} onBlur={hide}><Info className="h-4 w-4" /></button>} /><CardBody className="heatmap-card__body">{data.rows.length === 0 ? <EmptyState title="داده‌ای برای نقشه حرارتی نیست" /> : <><HeatGrid data={data} show={show} hide={hide} /><HeatLegend show={show} hide={hide} /></>}{tip ? <div aria-hidden="true" className="heatmap-floating-tooltip" style={{ transform: `translate3d(${tip.x}px, ${tip.y}px, 0) translateY(-100%)` }}><GlassTooltip>{tip.content}</GlassTooltip></div> : null}</CardBody></Card>;
}

export function DemandHeatmap({ data }: { data: HeatmapResponse }) { return <HeatmapCard data={data} />; }
