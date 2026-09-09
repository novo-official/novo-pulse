'use client';

import { Moon, Sun } from 'lucide-react';
import { useTheme } from '@/components/theme-provider';
import { cn } from '@/lib/utils';

export function ThemeToggle({ compact = false, className }: { compact?: boolean; className?: string }) {
  const { theme, toggleTheme } = useTheme();
  const isDark = theme === 'dark';
  return (
    <button
      type="button"
      onClick={toggleTheme}
      aria-label={isDark ? 'فعال کردن حالت روشن' : 'فعال کردن حالت تاریک'}
      aria-pressed={isDark}
      title={isDark ? 'حالت روشن' : 'حالت تاریک'}
      className={cn('theme-toggle', className)}
    >
      <span className="theme-toggle__icon" aria-hidden="true">{isDark ? <Moon /> : <Sun />}</span>
      {!compact ? <span>{isDark ? 'تاریک' : 'روشن'}</span> : null}
    </button>
  );
}
