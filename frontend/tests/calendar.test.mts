/**
 * Jalali conversion tests.
 *
 * Every date the dashboard shows an Iranian planner goes through `toJalali`, so
 * an off-by-one here is wrong on every screen at once. The fixtures below were
 * generated with Python's `jdatetime` and cover leap years, month boundaries
 * and the Nowruz rollover, which is where arithmetic conversions break.
 *
 * Run with: node --experimental-strip-types --test tests/
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { JALALI_MONTHS, toJalali } from '../lib/calendar.ts';
import { formatDate, formatFullDate } from '../lib/utils.ts';

const CASES: [string, number, number, number][] = [
  // Nowruz: the Jalali year turns on the March equinox, not 1 January.
  ['2024-03-19', 1402, 12, 29],
  ['2024-03-20', 1403, 1, 1],
  ['2024-03-21', 1403, 1, 2],
  ['2025-03-20', 1403, 12, 30],  // 1403 is a Jalali leap year: Esfand has 30 days
  ['2025-03-21', 1404, 1, 1],
  ['2023-03-20', 1401, 12, 29],
  ['2023-03-21', 1402, 1, 1],
  // Month-length boundary: the first six months have 31 days, the next five 30.
  ['2024-06-20', 1403, 3, 31],
  ['2024-06-21', 1403, 4, 1],
  ['2024-09-21', 1403, 6, 31],
  ['2024-09-22', 1403, 7, 1],
  ['2024-12-20', 1403, 9, 30],
  ['2024-12-21', 1403, 10, 1],
  // Gregorian leap day.
  ['2024-02-29', 1402, 12, 10],
  // Ordinary dates spread across the range the product will see.
  ['2024-08-02', 1403, 5, 12],
  ['2026-01-01', 1404, 10, 11],
  ['2020-01-01', 1398, 10, 11],
  ['2030-07-15', 1409, 4, 24],
];

test('converts Gregorian dates to Jalali', () => {
  for (const [iso, year, month, day] of CASES) {
    const [gy, gm, gd] = iso.split('-').map(Number);
    const got = toJalali(new Date(gy, gm - 1, gd));
    assert.deepEqual(
      { year: got.year, month: got.month, day: got.day },
      { year, month, day },
      `${iso} should be ${year}/${month}/${day}, got ${got.year}/${got.month}/${got.day}`,
    );
  }
});

test('every day of a Jalali year maps to a distinct, in-range date', () => {
  // 1403 is a leap year: 6x31 + 5x30 + 30 = 366 days.
  const seen = new Set<string>();
  const start = new Date(2024, 2, 20); // 1403-01-01
  for (let i = 0; i < 366; i += 1) {
    const date = new Date(start.getTime());
    date.setDate(start.getDate() + i);
    const jalali = toJalali(date);
    assert.equal(jalali.year, 1403, `day ${i} fell outside 1403`);
    assert.ok(jalali.month >= 1 && jalali.month <= 12, `bad month ${jalali.month}`);
    const limit = jalali.month <= 6 ? 31 : 30;
    assert.ok(
      jalali.day >= 1 && jalali.day <= limit,
      `${jalali.month}/${jalali.day} exceeds ${limit} days`,
    );
    seen.add(`${jalali.month}/${jalali.day}`);
  }
  assert.equal(seen.size, 366, 'a leap Jalali year has 366 distinct days');
});

test('formats dates in the requested calendar', () => {
  assert.equal(formatDate('2024-08-02', 'jalali'), '12 مرداد');
  assert.equal(formatDate('2024-08-02', 'gregorian'), '2 اوت');
  assert.equal(formatFullDate('2024-08-02', 'jalali'), '12 مرداد 1403');
  assert.equal(formatFullDate('2024-08-02', 'gregorian'), '2 اوت 2024');
});

test('defaults to Jalali, because that is what the audience plans in', () => {
  assert.equal(formatDate('2024-08-02'), '12 مرداد');
});

test('does not invent a date it cannot parse', () => {
  assert.equal(formatDate(null), '—');
  assert.equal(formatDate(''), '—');
  assert.equal(formatDate('not a date'), 'not a date');
});

test('month names are the Shamsi ones', () => {
  assert.equal(JALALI_MONTHS.length, 12);
  assert.equal(JALALI_MONTHS[0], 'فروردین');
  assert.equal(JALALI_MONTHS[11], 'اسفند');
});
