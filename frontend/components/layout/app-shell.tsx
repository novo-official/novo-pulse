'use client';

import {
  Activity,
  BarChart3,
  Database,
  FlaskConical,
  LayoutDashboard,
  LineChart,
  Menu,
  Presentation,
  X,
} from 'lucide-react';
import Link from 'next/link';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import { Suspense, useEffect, useState } from 'react';

import { StatusStrip } from '@/components/layout/status-strip';
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
    <div className={cn('flex min-h-screen', presentation && 'presentation')}>
      <CalendarSync />
      <Suspense fallback={null}>
        <PresentationSync />
      </Suspense>

      {/* ------------------------------------------------------- sidebar */}
      <aside
        className={cn(
          'fixed inset-y-0 right-0 z-40 w-72 shrink-0 border-l border-line bg-surface transition-transform lg:sticky lg:top-0 lg:h-screen lg:translate-x-0',
          open ? 'translate-x-0' : 'translate-x-full',
        )}
      >
        <div className="flex h-full flex-col">
          <div className="flex items-center justify-between px-5 py-5">
            <Link href="/dashboard" className="flex items-center gap-3">
              <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-brand-500 to-accent-violet text-white shadow-card">
                <Activity className="h-5 w-5" />
              </span>
              <span>
                <span className="block text-base font-bold leading-tight text-ink">نوو پالس</span>
                <span className="block text-[11px] leading-tight text-muted">
                  AI Demand Intelligence
                </span>
              </span>
            </Link>
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="rounded-lg p-1.5 text-muted hover:bg-slate-100 lg:hidden"
              aria-label="بستن منو"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          <nav className="flex-1 space-y-1 overflow-y-auto px-3 pb-4">
            {NAV.map((item) => {
              const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
              const Icon = item.icon;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    'group flex items-start gap-3 rounded-xl px-3 py-2.5 transition',
                    active
                      ? 'bg-brand-50 text-brand-700'
                      : 'text-muted hover:bg-slate-50 hover:text-ink',
                  )}
                >
                  <Icon className={cn('mt-0.5 h-4.5 w-4.5 shrink-0', active && 'text-brand-600')} />
                  <span className="min-w-0">
                    <span className="block text-sm font-medium">{item.label}</span>
                    <span className="debug-only block truncate text-[11px] text-muted/80">
                      {item.story}
                    </span>
                  </span>
                </Link>
              );
            })}
          </nav>

          <div className="border-t border-line px-3 py-3">
            <button
              type="button"
              onClick={togglePresentation}
              className={cn(
                'flex w-full items-center gap-2.5 rounded-xl px-3 py-2.5 text-sm font-medium transition',
                presentation
                  ? 'bg-accent-violet/10 text-accent-violet'
                  : 'text-muted hover:bg-slate-50 hover:text-ink',
              )}
            >
              <Presentation className="h-4 w-4" />
              {presentation ? 'خروج از حالت ارائه' : 'حالت ارائه'}
            </button>

            {/* Shamsi is the calendar the audience plans in; Gregorian stays a
                click away for anyone reading the raw dates alongside. */}
            <div
              className="mt-2 flex rounded-xl border border-line bg-surface p-1"
              role="group"
              aria-label="تقویم نمایش تاریخ‌ها"
            >
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
                    'flex-1 rounded-lg px-2 py-1.5 text-xs font-semibold transition',
                    calendar.calendar === value
                      ? 'bg-brand-600 text-white shadow-sm'
                      : 'text-muted hover:bg-slate-50 hover:text-ink',
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
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 border-b border-line bg-canvas/85 backdrop-blur">
          <div className="flex items-center gap-3 px-4 py-3 lg:px-8">
            <button
              type="button"
              onClick={() => setOpen(true)}
              className="rounded-lg p-2 text-muted hover:bg-slate-100 lg:hidden"
              aria-label="باز کردن منو"
            >
              <Menu className="h-5 w-5" />
            </button>
            <div className="min-w-0 flex-1">
              <h1 className="truncate text-lg font-bold leading-tight text-ink lg:text-xl">
                پل چهارم — هوش پیش‌بینی تقاضا
              </h1>
              <p className="truncate text-xs text-muted lg:text-sm">
                پیش‌بینی، تحلیل و شبیه‌سازی تقاضای بازار اقامت
              </p>
            </div>
            <Suspense fallback={null}>
              <StatusStrip />
            </Suspense>
          </div>
        </header>

        <main className="flex-1 px-4 py-6 lg:px-8 lg:py-8">
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
