'use client';

import { useMemo, useState } from 'react';

import type { CityRow } from '@/lib/pol4/types';

import { num } from './theme';

/**
 * Searchable city selector.
 *
 * Judges never see a numeric code: the list is searched and displayed by name
 * and province. The code travels underneath as the identifier, exactly as the
 * competition defines it.
 */
export function CityPicker({
  cities,
  value,
  onChange,
  label = 'شهر',
}: {
  cities: CityRow[];
  value: number | null;
  onChange: (cityCode: number) => void;
  label?: string;
}) {
  const [query, setQuery] = useState('');

  const matches = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const pool = needle
      ? cities.filter(
          (city) =>
            city.city.toLowerCase().includes(needle) ||
            city.province.toLowerCase().includes(needle) ||
            String(city.city_code).includes(needle),
        )
      : cities;
    return pool.slice(0, 60);
  }, [cities, query]);

  const selected = cities.find((city) => city.city_code === value) ?? null;

  return (
    <div className="pol4-picker">
      <label className="pol4-picker__label" htmlFor="pol4-city-search">
        {label}
      </label>
      <input
        id="pol4-city-search"
        className="pol4-picker__input"
        type="search"
        placeholder="جست‌وجوی شهر یا استان…"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        autoComplete="off"
      />
      <ul className="pol4-picker__list" role="listbox" aria-label={label}>
        {matches.map((city) => (
          <li key={city.city_code}>
            <button
              type="button"
              role="option"
              aria-selected={city.city_code === value}
              className={city.city_code === value ? 'is-selected' : undefined}
              onClick={() => onChange(city.city_code)}
            >
              <span className="pol4-picker__name" dir="ltr">
                {city.city}
              </span>
              <span className="pol4-picker__province" dir="ltr">
                {city.province}
              </span>
              <span className="pol4-picker__value" dir="ltr">
                {num(city.predicted_demand)}
              </span>
            </button>
          </li>
        ))}
        {matches.length === 0 ? <li className="pol4-picker__empty">شهری یافت نشد</li> : null}
      </ul>
      {selected ? (
        <p className="pol4-picker__selected">
          انتخاب‌شده: <strong dir="ltr">{selected.city}</strong>
        </p>
      ) : null}
    </div>
  );
}
