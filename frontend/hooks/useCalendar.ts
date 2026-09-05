'use client';

/**
 * Date formatters bound to the reader's chosen calendar.
 *
 * Components take their formatters from here rather than importing the raw
 * ones, so switching between Shamsi and Gregorian re-renders every date on the
 * page instead of leaving stale text behind.
 */
import { useMemo } from 'react';

import { useForecastFilters } from '@/hooks/useForecastFilters';
import type { CalendarName } from '@/lib/calendar';
import { formatDate, formatDateTime, formatFullDate } from '@/lib/utils';

export interface CalendarFormatters {
  calendar: CalendarName;
  setCalendar: (calendar: CalendarName) => void;
  /** Day + month, for axes and dense labels. */
  date: (iso: string | null | undefined) => string;
  /** Day + month + year. */
  fullDate: (iso: string | null | undefined) => string;
  dateTime: (iso: string | null | undefined) => string;
}

export function useCalendar(): CalendarFormatters {
  const calendar = useForecastFilters((state) => state.calendar);
  const setCalendar = useForecastFilters((state) => state.setCalendar);

  return useMemo(
    () => ({
      calendar,
      setCalendar,
      date: (iso) => formatDate(iso, calendar),
      fullDate: (iso) => formatFullDate(iso, calendar),
      dateTime: (iso) => formatDateTime(iso, calendar),
    }),
    [calendar, setCalendar],
  );
}
