/**
 * Jalali (Shamsi) calendar conversion for display.
 *
 * The audience plans in Shamsi months: a peak in مرداد means something to an
 * Iranian revenue manager in a way that "August" does not. The pipeline works
 * exclusively in Gregorian - the calendar is a presentation concern and never
 * reaches a model - so the conversion lives here, at the edge.
 *
 * Algorithm: the standard arithmetic Jalali conversion (jalaali-js), which is
 * exact for the 1178-1633 Jalali range. It is reproduced rather than pulled in
 * as a dependency so the dashboard keeps working with no network access.
 */

const BREAKS = [
  -61, 9, 38, 199, 426, 686, 756, 818, 1111, 1181, 1210, 1635, 2060, 2097,
  2192, 2262, 2324, 2394, 2456, 3178,
];

/** Days from the Gregorian epoch to a Gregorian date. */
function gregorianToDays(gy: number, gm: number, gd: number): number {
  let days =
    div((gy + div(gm - 8, 6) + 100100) * 1461, 4) +
    div(153 * mod(gm + 9, 12) + 2, 5) +
    gd -
    34840408;
  days = days - div(div(gy + 100100 + div(gm - 8, 6), 100) * 3, 4) + 752;
  return days;
}

function div(a: number, b: number): number {
  return ~~(a / b);
}

function mod(a: number, b: number): number {
  return a - ~~(a / b) * b;
}

/** Leap-year bookkeeping for a Jalali year: which Gregorian year it starts in. */
function jalaliCalendar(jy: number): { gy: number; march: number; leap: number } {
  const bl = BREAKS.length;
  const gy = jy + 621;
  let leapJ = -14;
  let jp = BREAKS[0];

  let jump = 0;
  for (let i = 1; i < bl; i += 1) {
    const jm = BREAKS[i];
    jump = jm - jp;
    if (jy < jm) break;
    leapJ = leapJ + div(jump, 33) * 8 + div(mod(jump, 33), 4);
    jp = jm;
  }
  let n = jy - jp;

  leapJ = leapJ + div(n, 33) * 8 + div(mod(n, 33) + 3, 4);
  if (mod(jump, 33) === 4 && jump - n === 4) leapJ += 1;

  const leapG = div(gy, 4) - div((div(gy, 100) + 1) * 3, 4) - 150;
  const march = 20 + leapJ - leapG;

  if (jump - n < 6) n = n - jump + div(jump + 4, 33) * 33;
  let leap = mod(mod(n + 1, 33) - 1, 4);
  if (leap === -1) leap = 4;

  return { gy, march, leap };
}

export interface JalaliDate {
  year: number;
  month: number; // 1-12
  day: number;
}

/** Convert a Gregorian date to Jalali. */
export function toJalali(date: Date): JalaliDate {
  const gy = date.getFullYear();
  const gm = date.getMonth() + 1;
  const gd = date.getDate();

  const days = gregorianToDays(gy, gm, gd);
  let jy = gy - 621;
  const r = jalaliCalendar(jy);
  const jdays1f = gregorianToDays(gy, 3, r.march);
  let k = days - jdays1f;

  if (k >= 0) {
    if (k <= 185) {
      return { year: jy, month: 1 + div(k, 31), day: mod(k, 31) + 1 };
    }
    k -= 186;
  } else {
    // `r` describes the year we started from, so the leap check has to happen
    // against it - not against the year we are about to fall back into.
    jy -= 1;
    k += 179;
    if (r.leap === 1) k += 1;
  }
  return { year: jy, month: 7 + div(k, 30), day: mod(k, 30) + 1 };
}

export const JALALI_MONTHS = [
  'فروردین', 'اردیبهشت', 'خرداد', 'تیر', 'مرداد', 'شهریور',
  'مهر', 'آبان', 'آذر', 'دی', 'بهمن', 'اسفند',
];

export const GREGORIAN_MONTHS_FA = [
  'ژانویه', 'فوریه', 'مارس', 'آوریل', 'مه', 'ژوئن',
  'ژوئیه', 'اوت', 'سپتامبر', 'اکتبر', 'نوامبر', 'دسامبر',
];

export type CalendarName = 'jalali' | 'gregorian';
