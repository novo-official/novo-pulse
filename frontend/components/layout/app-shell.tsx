'use client';

import {
  Activity,
  BarChart3,
  ChevronLeft,
  ChevronRight,
  Database,
  FlaskConical,
  LayoutDashboard,
  LineChart,
  Menu,
  Presentation,
  Sparkles,
  X,
} from 'lucide-react';
import Link from 'next/link';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import { Suspense, useEffect, useState } from 'react';

import { StatusStrip } from '@/components/layout/status-strip';
import { ThemeToggle } from '@/components/theme-toggle';
import { useCalendar } from '@/hooks/useCalendar';
import { storedCalendar, useForecastFilters } from '@/hooks/useForecastFilters';
import { cn } from '@/lib/utils';

const NAV = [
  { href: '/dashboard', label: 'داشبورد', icon: LayoutDashboard, story: 'چه اتفاقی می‌افتد؟' },
  { href: '/forecasts', label: 'پیش‌بینی‌ها', icon: LineChart, story: 'کجا اتفاق می‌افتد؟' },
  { href: '/scenarios', label: 'شبیه‌سازی سناریو', icon: FlaskConical, story: 'اگر شرایط تغییر کند؟' },
  { href: '/backtesting', label: 'اعتبارسنجی', icon: Activity, story: 'مدل چقدر دقیق بوده؟' },
  { href: '/models', label: 'مدل‌ها', icon: BarChart3, story: 'کدام مدل برنده است؟' },
  { href: '/data-lab', label: 'آزمایشگاه داده', icon: Database, story: 'دیتاست جدید' },
];

/** Restore the reader's calendar choice once the browser is available. */
function CalendarSync() {
  const setCalendar = useForecastFilters((state) => state.setCalendar);
  useEffect(() => {
    const saved = storedCalendar();
    if (saved !== 'jalali') setCalendar(saved);
  }, [setCalendar]);
  return null;
}

function PresentationSync() {
  const params = useSearchParams();
  const setPresentation = useForecastFilters((state) => state.setPresentation);
  useEffect(() => {
    setPresentation(params.get('presentation') === 'true');
  }, [params, setPresentation]);
  return null;
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [compact, setCompact] = useState(true);
  const presentation = useForecastFilters((state) => state.presentation);
  const setPresentation = useForecastFilters((state) => state.setPresentation);
  const calendar = useCalendar();

  useEffect(() => setOpen(false), [pathname]);

  const togglePresentation = () => {
    const next = !presentation;
    setPresentation(next);
    const url = new URL(window.location.href);
    if (next) url.searchParams.set('presentation', 'true');
    else url.searchParams.delete('presentation');
    router.replace(`${url.pathname}${url.search}`);
  };

  return (
    <div className={cn('flex min-h-screen bg-transparent', presentation && 'presentation')}>
      <Suspense fallback={null}>
        <PresentationSync />
      </Suspense>

      {/* ------------------------------------------------------- sidebar */}
      <aside
        className={cn(
          'app-sidebar fixed inset-y-0 right-0 z-40 w-[276px] shrink-0 text-ink transition-all duration-300 lg:h-screen lg:translate-x-0',
          compact ? 'lg:w-[88px]' : 'lg:w-[248px]',
          open ? 'translate-x-0' : 'translate-x-full',
        )}
      >
        <div className="flex h-full flex-col">
          <div className={cn('flex items-center justify-between px-5 pb-4 pt-5', compact && 'lg:justify-center lg:px-3')}>
            <Link href="/dashboard" className="flex items-center gap-3">
              <span className="relative flex h-11 w-11 items-center justify-center overflow-hidden rounded-[14px] bg-gradient-to-br from-indigo-400 via-brand-500 to-cyan-400 text-white shadow-lg shadow-indigo-950/40">
                <Activity className="relative z-10 h-5 w-5" />
                <span className="absolute inset-x-0 top-1/2 h-px bg-white/30" />
              </span>
              <span className={cn(compact && 'lg:hidden')}>
                <span className="block text-base font-bold leading-tight text-ink">نووُ پالس</span>
                <span className="mt-1 block text-[10px] font-medium tracking-wide text-slate-500">
                  هوش بازار
                </span>
              </span>
            </Link>
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="rounded-lg p-1.5 text-muted hover:bg-black/5 hover:text-ink lg:hidden"
              aria-label="بستن منو"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          <button
            type="button"
            onClick={() => setCompact((value) => !value)}
            className="absolute -left-3 top-[125px] z-10 hidden h-7 w-7 items-center justify-center rounded-full border border-line bg-surface text-muted shadow-md transition hover:border-brand-300 hover:text-brand-600 lg:flex"
            aria-label={compact ? 'باز کردن سایدبار' : 'جمع کردن سایدبار'}
          >
            {compact ? <ChevronLeft className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
          </button>

          <div className={cn('mx-4 mb-4 rounded-xl border border-brand-100 bg-brand-50/55 px-3 py-2.5', compact && 'lg:hidden')}>
            <p className="flex items-center gap-2 text-xs text-slate-600"><Sparkles className="h-3.5 w-3.5 text-cyan-500" />نبض بازار را در دست بگیر</p>
          </div>
          <p className={cn('px-5 pb-2 text-[10px] font-bold tracking-wider text-slate-400', compact && 'lg:hidden')}>فضای تحلیل</p>
          <nav className={cn('flex-1 space-y-1 overflow-y-auto px-3 pb-4', compact && 'lg:px-2')}>
            {NAV.map((item) => {
              const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
              const Icon = item.icon;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  title={compact ? item.label : undefined}
                  className={cn(
                    'group relative flex items-start gap-3 rounded-xl px-3 py-2.5 transition duration-200',
                    compact && 'lg:justify-center lg:px-2 lg:py-3',
                    active
                      ? 'bg-brand-50 text-ink shadow-sm ring-1 ring-brand-100'
                      : 'text-muted hover:bg-black/5 hover:text-ink',
                  )}
                >
                  {active ? <span className="absolute inset-y-3 right-1 w-0.5 rounded-full bg-cyan-400" /> : null}
                  <Icon className={cn('mt-0.5 h-4.5 w-4.5 shrink-0', active && 'text-cyan-400')} />
                  <span className={cn('min-w-0', compact && 'lg:hidden')}>
                    <span className="block text-sm font-medium">{item.label}</span>
                    <span className="debug-only block truncate text-[11px] text-slate-500 group-hover:text-slate-400">
                      {item.story}
                    </span>
                  </span>
                </Link>
              );
            })}
          </nav>

          <div className={cn('border-t border-white/10 px-3 py-3', compact && 'lg:px-2')}>
            <ThemeToggle className="mb-2 hidden w-full justify-center lg:flex" compact={compact} />
            <button
              type="button"
              onClick={togglePresentation}
              title={compact ? (presentation ? 'خروج از حالت ارائه' : 'حالت ارائه') : undefined}
              className={cn(
                'flex w-full items-center gap-2.5 rounded-xl px-3 py-2.5 text-sm font-medium transition',
                compact && 'lg:justify-center lg:px-2',
                presentation
                  ? 'bg-violet-50 text-violet-700'
                  : 'text-muted hover:bg-black/5 hover:text-ink',
              )}
            >
              <Presentation className="h-4 w-4" />
              <span className={cn(compact && 'lg:hidden')}>{presentation ? 'خروج از حالت ارائه' : 'حالت ارائه'}</span>
            </button>

            {/* Shamsi is the calendar the audience plans in; Gregorian stays a
                click away for anyone reading the raw dates alongside. */}
            <div
              className={cn('calendar-switch mt-3', compact && 'lg:hidden')}
              role="group"
              aria-label="تقویم نمایش تاریخ‌ها"
            >
              <span
                aria-hidden="true"
                className={cn(
                  'calendar-switch__indicator right-1',
                  calendar.calendar === 'jalali' ? 'translate-x-0' : '-translate-x-full',
                )}
              />
              {(
                [
                  ['jalali', 'شمسی'],
                  ['gregorian', 'میلادی'],
                ] as const
              ).map(([value, label]) => (
                <button
                  key={value}
                  type="button"
                  onClick={() => calendar.setCalendar(value)}
                  aria-pressed={calendar.calendar === value}
                  className={cn(
                    'calendar-switch__option',
                    calendar.calendar === value
                      ? 'text-white'
                      : 'text-slate-500 hover:text-brand-700',
                  )}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </aside>

      {open ? (
        <button
          type="button"
          aria-label="بستن منو"
          onClick={() => setOpen(false)}
          className="fixed inset-0 z-30 bg-slate-900/30 backdrop-blur-sm lg:hidden"
        />
      ) : null}

      {/* ---------------------------------------------------------- main */}
      <div className={cn('flex min-w-0 flex-1 flex-col transition-[margin] duration-200', compact ? 'lg:mr-[88px]' : 'lg:mr-[248px]')}>
        <header className="sticky top-0 z-20 border-b border-line/80 bg-canvas/80 backdrop-blur-xl">
          <div className="mx-auto flex max-w-[1600px] items-center gap-3 px-4 py-3 lg:px-8">
            <button
              type="button"
              onClick={() => setOpen(true)}
              className="rounded-lg p-2 text-muted hover:bg-slate-100 lg:hidden"
              aria-label="باز کردن منو"
            >
              <Menu className="h-5 w-5" />
            </button>
            <div className="min-w-0 flex-1">
              <h1 className="truncate text-sm font-semibold leading-tight text-ink sm:text-base">مرکز فرمان هوشمند بازار</h1>
              <p className="mt-0.5 hidden truncate text-xs text-muted sm:block">تصمیم دقیق‌تر با سیگنال‌های زنده و پیش‌بینی داده‌محور</p>
            </div>
            <Suspense fallback={null}>
              <StatusStrip />
            </Suspense>
            <ThemeToggle compact className="lg:hidden" />
          </div>
        </header>

        <main className="flex-1 px-4 py-5 sm:px-6 lg:px-8 lg:py-7">
          <div className="mx-auto w-full max-w-[1500px]">{children}</div>
        </main>

        <footer className="debug-only border-t border-line px-4 py-4 text-xs text-muted lg:px-8">
          نوو پالس — پیش‌بینی تقاضا با مدل‌های محلی و متن‌باز. هیچ سرویس ابری یا API پولی استفاده
          نمی‌شود.
        </footer>
      </div>
    </div>
  );
}
