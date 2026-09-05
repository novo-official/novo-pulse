/**
 * The competition-day flow, driven entirely through the browser.
 *
 * The hackathon dataset arrives as several files, from an Iranian marketplace:
 * a raw booking log with Jalali dates and Persian headers, plus separate
 * accommodation and calendar tables. This walks that exact shape through the
 * Data Lab - upload, join, map, validate, train - and checks that the product
 * says what it did rather than converting and counting silently.
 */
import { chromium } from 'playwright';
import { mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const SP = process.env.SP ?? new URL('.', import.meta.url).pathname;
const results = [];
const errors = [];
const check = (n, ok, d = '') => {
  results.push({ n, ok });
  console.log(`${ok ? '  OK ' : '  !! '}${n}${d ? '  [' + d + ']' : ''}`);
};

// ------------------------------------------------------------ the fixtures
const dir = join(tmpdir(), `novo-competition-${Date.now()}`);
mkdirSync(dir, { recursive: true });

// Gregorian -> Jalali, so the fixture is written the way a real export is.
function toJalali(date) {
  const gy = date.getUTCFullYear();
  const gm = date.getUTCMonth() + 1;
  const gd = date.getUTCDate();
  const g_d_m = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
  let jy = gy <= 1600 ? 0 : 979;
  const gy2 = gy <= 1600 ? gy - 621 : gy - 1600;
  const gm2 = gm > 2 ? gy + 1 : gy;
  let days =
    365 * gy2 +
    Math.floor((gm2 + 3) / 4) -
    Math.floor((gm2 + 99) / 100) +
    Math.floor((gm2 + 399) / 400) -
    80 +
    gd +
    g_d_m[gm - 1];
  jy += 33 * Math.floor(days / 12053);
  days %= 12053;
  jy += 4 * Math.floor(days / 1461);
  days %= 1461;
  if (days > 365) {
    jy += Math.floor((days - 1) / 365);
    days = (days - 1) % 365;
  }
  const jm = days < 186 ? 1 + Math.floor(days / 31) : 7 + Math.floor((days - 186) / 30);
  const jd = 1 + (days < 186 ? days % 31 : (days - 186) % 30);
  return `${String(jy).padStart(4, '0')}/${String(jm).padStart(2, '0')}/${String(jd).padStart(2, '0')}`;
}

const listings = ['ACC-001', 'ACC-002', 'ACC-003', 'ACC-004', 'ACC-005', 'ACC-006'];
const cities = ['تهران', 'مشهد', 'اصفهان'];
const kinds = ['ویلا', 'هتل', 'آپارتمان'];

// One row per booking - no demand column anywhere in the file.
const bookings = ['شناسه_رزرو,تاریخ_رزرو,کد_اقامتگاه,شهر,نوع_اقامتگاه,مبلغ'];
const holidays = ['تاریخ_رزرو,تعطیل_رسمی'];
let id = 0;
const start = Date.UTC(2023, 3, 1);
const DAYS = 640;
for (let day = 0; day < DAYS; day += 1) {
  const date = new Date(start + day * 86400000);
  const jalali = toJalali(date);
  const weekday = date.getUTCDay();
  const holiday = weekday === 5 ? 1 : 0;
  holidays.push(`${jalali},${holiday}`);
  const seasonal = 1 + 0.5 * Math.sin((2 * Math.PI * day) / 365.25);
  listings.forEach((listing, index) => {
    const rate = (1.2 + index * 0.4) * seasonal * (holiday ? 2.0 : 1.0);
    const count = Math.max(0, Math.round(rate + ((day * 7 + index * 13) % 5) / 4 - 0.5));
    for (let n = 0; n < count; n += 1) {
      id += 1;
      const price = 1500000 + ((day * 31 + n * 17 + index * 101) % 60) * 100000;
      bookings.push(
        `${id},${jalali},${listing},${cities[index % 3]},${kinds[index % 3]},${price}`,
      );
    }
  });
}

const accommodations = ['کد_اقامتگاه,ظرفیت,امتیاز'];
listings.forEach((listing, index) => accommodations.push(`${listing},${2 + index * 2},${(3.5 + index * 0.2).toFixed(1)}`));

const bookingsPath = join(dir, 'bookings_fa.csv');
const accommodationsPath = join(dir, 'accommodations.csv');
const holidaysPath = join(dir, 'holidays.csv');
writeFileSync(bookingsPath, bookings.join('\n'), 'utf8');
writeFileSync(accommodationsPath, accommodations.join('\n'), 'utf8');
writeFileSync(holidaysPath, holidays.join('\n'), 'utf8');
console.log(`fixtures: ${bookings.length - 1} bookings, ${listings.length} listings, ${DAYS} calendar days`);

// ------------------------------------------------------------------ drive
const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' });
const page = await browser.newPage({ viewport: { width: 1500, height: 1200 } });
page.on('pageerror', (e) => errors.push(`PAGEERROR: ${e.message}`));

console.log('\n=== DATA LAB: a Jalali booking log with Persian headers');
await page.goto('http://127.0.0.1:3000/data-lab', { waitUntil: 'networkidle' });
await page.waitForTimeout(1500);

await page.locator('input[type=file]').first().setInputFiles(bookingsPath);
await page.waitForTimeout(12000);

const body1 = await page.locator('body').innerText();
check('Jalali calendar reported to the user', body1.includes('تقویم شمسی'));
check('raw booking log reported to the user', body1.includes('جدول خام رزرو'));

const selects = page.locator('select');
check('timestamp auto-detected', (await selects.nth(0).inputValue()) === 'تاریخ_رزرو',
      await selects.nth(0).inputValue());
check('entity auto-detected', (await selects.nth(2).inputValue()) === 'کد_اقامتگاه',
      await selects.nth(2).inputValue());

// The target selector must be empty AND disabled: demand is the row count, and
// "مبلغ" is exactly the plausible-looking column that must not be summed.
check('target left empty for a booking log', (await selects.nth(1).inputValue()) === '',
      (await selects.nth(1).inputValue()) || '(empty)');
check('target selector disabled while counting', await selects.nth(1).isDisabled());

const aggregation = page.getByLabel('تجمیع سطرهای تکراری');
check('aggregation set to count', (await aggregation.inputValue()) === 'count',
      await aggregation.inputValue());

const calendar = page.getByLabel('تقویم ستون تاریخ');
check('calendar set to jalali', (await calendar.inputValue()) === 'jalali',
      await calendar.inputValue());

// ------------------------------------------------------------ side tables
console.log('\n=== JOINING THE OTHER TWO FILES');
const addSide = page.getByRole('button', { name: /افزودن فایل جانبی/ });
check('join control present', (await addSide.count()) > 0);

const sideInputs = page.locator('input[type=file]');
await sideInputs.last().setInputFiles(accommodationsPath);
await page.waitForTimeout(9000);
await page.locator('input[type=file]').last().setInputFiles(holidaysPath);
await page.waitForTimeout(9000);

const body2 = await page.locator('body').innerText();
check('both side files listed', body2.includes('۲ فایل جانبی') || body2.includes('2 فایل جانبی'),
      (body2.match(/[۰-۹0-9]+ فایل جانبی/) || ['(not shown)'])[0]);

// ---------------------------------------------------------------- validate
console.log('\n=== VALIDATE');
await page.getByRole('button', { name: /اعتبارسنجی داده/ }).click();
await page.waitForTimeout(15000);
const body3 = await page.locator('body').innerText();
check('data quality report rendered', body3.includes('امتیاز سلامت داده'));
check('the calendar conversion is raised as a finding', body3.includes('شمسی') || body3.includes('Jalali'));

// ------------------------------------------------------------------- train
console.log('\n=== TRAIN');
await page.getByRole('button', { name: /شروع آموزش/ }).click();
console.log('     training started, polling...');
let status = '';
for (let i = 0; i < 120; i += 1) {
  await page.waitForTimeout(4000);
  const body = await page.locator('body').innerText();
  if (body.includes('ناموفق')) { status = 'failed'; break; }   // the negative first
  if (/(^|\s)موفق(\s|$)/.test(body)) { status = 'succeeded'; break; }
}
check('training completed via the UI', status === 'succeeded', status || 'timed out');

const trained = await page.locator('body').innerText();
const championLine = (trained.match(/مدل منتخب:.*/) || ['(not shown)'])[0];
check('champion reported', championLine.includes('منتخب'), championLine.slice(0, 120));
await page.screenshot({ path: `${SP}/ui_competition_trained.png`, fullPage: true });

// --------------------------------------------------------------- dashboard
console.log('\n=== DASHBOARD on the competition-shaped data');
await page.goto('http://127.0.0.1:3000/dashboard', { waitUntil: 'networkidle' });
await page.waitForTimeout(5000);
const dash = await page.locator('body').innerText();
check('Persian destinations reached the dashboard', /تهران|مشهد|اصفهان/.test(dash),
      (dash.match(/تهران|مشهد|اصفهان/g) || []).slice(0, 3).join(','));
check('no infinite skeletons', (await page.locator('.animate-shimmer').count()) === 0);
check('chart renders', (await page.locator('svg.recharts-surface').count()) > 0);
await page.screenshot({ path: `${SP}/ui_competition_dashboard.png`, fullPage: true });

console.log('\nERRORS:', errors.length ? errors.slice(0, 6) : 'none');
const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length} checks, ${failed.length} failed`);
await browser.close();
process.exit(failed.length ? 1 : 0);
