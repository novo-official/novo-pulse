'use client';

import { BarChart3, LineChart, Map, ShieldCheck } from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

import { StatusStrip } from '@/components/layout/status-strip';
import { ThemeToggle } from '@/components/theme-toggle';
import { cn } from '@/lib/utils';

/**
 * Four routes, a top bar, no sidebar.
 *
 * The nav is the argument: what will happen, where, how stable it is, and how
 * we know. Anything that did not serve one of those questions was removed.
 */
const NAV = [
  { href: '/overview', label: 'نمای کلی', icon: LineChart },
  { href: '/city', label: 'تحلیل شهر', icon: Map },
  { href: '/stability', label: 'پایداری پیش‌بینی', icon: ShieldCheck },
  { href: '/reports', label: 'گزارش‌ها', icon: BarChart3 },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="pol4-shell">
      <header className="pol4-topbar">
        <Link href="/overview" className="pol4-brand">
          <span className="pol4-brand__mark" aria-hidden="true" />
          <span>
            <strong>نوو پالس</strong>
            <span>هوش تقاضای پل ۴</span>
          </span>
        </Link>

        <nav className="pol4-nav" aria-label="بخش‌های اصلی">
          {NAV.map((item) => {
            const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                aria-current={active ? 'page' : undefined}
                className={cn('pol4-nav__item', active && 'is-active')}
              >
                <Icon className="h-4 w-4" aria-hidden="true" />
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="pol4-topbar__end">
          <StatusStrip />
          <ThemeToggle />
        </div>
      </header>

      <main className="pol4-content">{children}</main>
    </div>
  );
}
