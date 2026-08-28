import { chromium } from 'playwright';
const SP = process.env.SP ?? new URL('.', import.meta.url).pathname;
const results = [];
const errors = [];
const check = (n, ok, d='') => { results.push({n, ok}); console.log(`${ok?'  OK ':'  !! '}${n}${d?'  ['+d+']':''}`); };

const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' });
const page = await browser.newPage({ viewport: { width: 1500, height: 1100 } });
page.on('pageerror', e => errors.push(`PAGEERROR: ${e.message}`));

console.log('=== DATA LAB: upload -> map -> validate -> train, entirely in the browser');
await page.goto('http://127.0.0.1:3000/data-lab', { waitUntil: 'networkidle' });
await page.waitForTimeout(1500);

// 1. upload
await page.locator('input[type=file]').setInputFiles(`${SP}/browser_upload.csv`);
await page.waitForTimeout(9000);
const cols = await page.locator('table tbody tr').count();
check('uploaded file profiled', cols >= 8, `${cols} columns detected`);

// 2. the auto-suggested mapping should have found the new names
const selects = page.locator('select');
const timestamp = await selects.nth(0).inputValue();
const target = await selects.nth(1).inputValue();
const entity = await selects.nth(2).inputValue();
check('timestamp auto-detected', timestamp === 'booking_day', timestamp);
check('target auto-detected', target === 'room_nights', target || '(none)');
check('entity auto-detected', entity === 'unit_code', entity || '(none)');

// 3. validate
await page.getByRole('button', { name: /اعتبارسنجی داده/ }).click();
await page.waitForTimeout(10000);
const health = await page.locator('text=امتیاز سلامت داده').count();
check('data quality report rendered', health > 0);

// 4. train from the UI
await page.getByRole('button', { name: /شروع آموزش/ }).click();
console.log('     training started, polling...');
let status = '';
for (let i = 0; i < 90; i++) {
  await page.waitForTimeout(4000);
  const body = await page.locator('body').innerText();
  if (body.includes('ناموفق')) { status = 'failed'; break; }   // check the negative FIRST
  if (/(^|\s)موفق(\s|$)/.test(body)) { status = 'succeeded'; break; }
}
check('training completed via the UI', status === 'succeeded', status || 'timed out');
const finalText = await page.locator('body').innerText();
const championLine = (finalText.match(/مدل منتخب:.*/) || ['(not shown)'])[0];
check('champion reported in the UI', championLine.includes('منتخب'), championLine.slice(0, 110));
await page.screenshot({ path: `${SP}/ui_upload_trained.png`, fullPage: true });

// 5. the dashboard must now reflect the NEW dataset
console.log('\n=== DASHBOARD after training on the uploaded dataset');
await page.goto('http://127.0.0.1:3000/dashboard', { waitUntil: 'networkidle' });
await page.waitForTimeout(4000);
const dashText = await page.locator('body').innerText();
check('dashboard shows the new zones', /zone_\d/.test(dashText), (dashText.match(/zone_\d/g)||[]).slice(0,4).join(','));
check('no infinite skeletons', (await page.locator('.animate-shimmer').count()) === 0);
check('chart still renders', (await page.locator('svg.recharts-surface').count()) > 0);
await page.screenshot({ path: `${SP}/ui_dashboard_newdata.png`, fullPage: true });

console.log('\nERRORS:', errors.length ? errors.slice(0,6) : 'none');
const failed = results.filter(r => !r.ok);
console.log(`\n${results.length} checks, ${failed.length} failed`);
await browser.close();
process.exit(failed.length ? 1 : 0);
