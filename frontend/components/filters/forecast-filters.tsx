'use client';

import { Filter } from 'lucide-react';

import { Select } from '@/components/ui/select';
import { useForecastFilters } from '@/hooks/useForecastFilters';
import type { Level, Member } from '@/lib/types/api';
import { LEVEL_FA, cn } from '@/lib/utils';

export function HorizonSelector({
  options = [7, 14, 30, 60, 90],
  className,
}: {
  options?: number[];
  className?: string;
}) {
  const horizon = useForecastFilters((state) => state.horizon);
  const setHorizon = useForecastFilters((state) => state.setHorizon);

  return (
    <div
      className={cn('inline-flex rounded-xl border border-line bg-slate-50/80 p-1 shadow-inner', className)}
      role="group"
      aria-label="افق پیش‌بینی"
    >
      {options.map((option) => (
        <button
          key={option}
          type="button"
          onClick={() => setHorizon(option)}
          className={cn(
            'nums rounded-lg px-3 py-1.5 text-xs font-semibold transition duration-200',
            horizon === option
              ? 'bg-brand-600 text-white shadow-sm'
              : 'text-muted hover:bg-slate-50 hover:text-ink',
          )}
        >
          {option}D
        </button>
      ))}
    </div>
  );
}

export function LevelEntityFilter({
  levels,
  members,
  className,
}: {
  levels: Level[];
  members: Member[];
  className?: string;
}) {
  const { level, entityId, setLevel, setEntityId } = useForecastFilters();

  return (
    <div className={cn('grid gap-3 sm:grid-cols-2', className)}>
      <Select
        label="سطح تحلیل"
        value={level}
        onChange={(event) => setLevel(event.target.value as Level)}
      >
        {levels.map((option) => (
          <option key={option} value={option}>
            {LEVEL_FA[option] ?? option}
          </option>
        ))}
      </Select>
      <Select
        label={LEVEL_FA[level] ?? 'انتخاب'}
        value={entityId ?? ''}
        onChange={(event) => setEntityId(event.target.value || null)}
        disabled={members.length === 0}
      >
        <option value="">همه ({members.length})</option>
        {members.map((member) => (
          <option key={member.id} value={member.id}>
            {member.label}
          </option>
        ))}
      </Select>
    </div>
  );
}

export function FilterBar({
  levels,
  members,
  horizons,
}: {
  levels: Level[];
  members: Member[];
  horizons: number[];
}) {
  return (
    <section aria-label="فیلترهای پیش‌بینی" className="relative flex flex-col gap-4 overflow-visible rounded-2xl border border-line/90 bg-surface/95 p-4 shadow-card lg:flex-row lg:items-end lg:justify-between lg:p-5">
      <span className="pointer-events-none absolute inset-y-0 right-0 w-1 bg-gradient-to-b from-brand-500 to-cyan-400" />
      <div className="flex items-center gap-2 lg:hidden">
        <Filter className="h-4 w-4 text-muted" />
        <span className="text-sm font-medium text-ink">فیلترها</span>
      </div>
      <LevelEntityFilter levels={levels} members={members} className="lg:w-[460px]" />
      <div className="flex flex-col gap-1.5">
        <span className="text-xs font-medium text-muted">افق پیش‌بینی</span>
        <HorizonSelector options={horizons} />
      </div>
    </section>
  );
}
