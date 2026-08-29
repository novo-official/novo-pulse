'use client';

import { Filter, Layers3, MapPin, Sparkles } from 'lucide-react';

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
      className={cn('forecast-segment inline-flex rounded-xl', className)}
      role="group"
      aria-label="افق پیش‌بینی"
    >
      {options.map((option) => (
        <button
          key={option}
          type="button"
          onClick={() => setHorizon(option)}
          aria-pressed={horizon === option}
          className={cn(
            'forecast-segment__option nums rounded-[9px] px-3 py-1.5 text-xs font-semibold transition duration-200',
            horizon === option
              ? 'forecast-segment__option--active bg-brand-600 text-white'
              : 'text-muted hover:text-ink',
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
        icon={<Layers3 className="h-3.5 w-3.5" aria-hidden="true" />}
        containerClassName="filter-bar__field"
        className="filter-bar__select"
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
        icon={<MapPin className="h-3.5 w-3.5" aria-hidden="true" />}
        containerClassName="filter-bar__field"
        className="filter-bar__select"
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
    <section aria-label="فیلترهای پیش‌بینی" className="filter-bar relative flex flex-col gap-5 overflow-visible rounded-[20px] border border-slate-900/[0.07] bg-surface p-5 shadow-[0_1px_2px_rgb(15_23_42_/_0.02),0_14px_30px_-24px_rgb(15_23_42_/_0.24)] lg:flex-row lg:items-end lg:justify-between lg:px-6 lg:py-5">
      <div className="filter-bar__analysis min-w-0">
        <div className="filter-bar__context">
          <Filter className="h-3.5 w-3.5" aria-hidden="true" />
          <span>فیلترهای تحلیل</span>

        </div>
        <LevelEntityFilter levels={levels} members={members} className="mt-2.5 lg:w-[500px]" />
      </div>
      <div className="filter-bar__horizon flex flex-col gap-2">
        <span className="flex items-center gap-1.5 px-0.5 text-xs font-semibold text-muted"><Sparkles className="h-3.5 w-3.5 text-brand-500/80" aria-hidden="true" />افق پیش‌بینی</span>
        <HorizonSelector options={horizons} />
      </div>
    </section>
  );
}
