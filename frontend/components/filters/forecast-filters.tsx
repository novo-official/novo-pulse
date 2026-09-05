'use client';

import { Filter } from 'lucide-react';
import { useEffect } from 'react';

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

  // A run trained to 30 days cannot answer a 90-day question, so a horizon the
  // current run does not cover snaps down to the longest one it does.
  useEffect(() => {
    if (options.length > 0 && !options.includes(horizon)) {
      setHorizon(Math.max(...options));
    }
  }, [options, horizon, setHorizon]);

  return (
    <div
      className={cn('inline-flex rounded-xl border border-line bg-surface p-1', className)}
      role="group"
      aria-label="افق پیش‌بینی"
    >
      {options.map((option) => (
        <button
          key={option}
          type="button"
          onClick={() => setHorizon(option)}
          className={cn(
            'nums rounded-lg px-3 py-1.5 text-xs font-semibold transition',
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
    <div className="flex flex-col gap-4 rounded-2xl border border-line bg-surface p-4 shadow-card lg:flex-row lg:items-end lg:justify-between">
      <div className="flex items-center gap-2 lg:hidden">
        <Filter className="h-4 w-4 text-muted" />
        <span className="text-sm font-medium text-ink">فیلترها</span>
      </div>
      <LevelEntityFilter levels={levels} members={members} className="lg:w-[420px]" />
      <div className="flex flex-col gap-1.5">
        <span className="text-xs font-medium text-muted">افق پیش‌بینی</span>
        <HorizonSelector options={horizons} />
      </div>
    </div>
  );
}
