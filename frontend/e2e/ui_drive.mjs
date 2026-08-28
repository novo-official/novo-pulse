import { chromium } from 'playwright';

const SP = process.env.SP ?? new URL('.', import.meta.url).pathname;
const BASE = 'http://127.0.0.1:3000';
const results = [];
const errors = [];

function check(name, ok, detail = '') {
  results.push({ name, ok, detail });
  console.log(`${ok ? '  OK ' : '  !! '}${name}${detail ? '  [' + detail + ']' : ''}`);
}

const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' });
const page = await browser.newPage({ viewport: { width: 1500, height: 1000 } });
page.on('pageerror', (e) => errors.push(`PAGEERROR ${page.url()}: ${e.message}`));
page.on('console', (m) => { if (m.type() === 'error' && !m.text().includes('favicon')) errors.push(`CONSOLE ${page.url()}: ${m.text()}`); });

const go = async (path) => {
  await page.goto(`${BASE}${path}`, { waitUntil: 'networkidle', timeout: 60000 });
  await page.waitForTimeout(1800);
};

// ---------------------------------------------------------------- dashboard
console.log('\n=== DASHBOARD');
await go('/dashboard');
check('KPI cards rendered', (await page.locator('.kpi-value').count()) >= 6,
      `${await page.locator('.kpi-value').count()} cards`);
const kpiText = await page.locator('.kpi-value').first().innerText();
check('KPI shows a real number', /[0-9]/.test(kpiText), kpiText);
check('main chart drawn', (await page.locator('svg.recharts-surface').count()) > 0,
      `${await page.locator('svg.recharts-surface').count()} charts`);
check('no infinite skeletons', (await page.locator('.animate-shimmer').count()) === 0,
      `${await page.locator('.animate-shimmer').count()} skeletons`);
check('overview table populated', (await page.locator('table tbody tr').count()) > 0,
      `${await page.locator('table tbody tr').count()} rows`);
check('heatmap cells drawn', (await page.locator('[title*="·"]').count()) > 0);
const narrative = await page.locator('text=تحلیل هوشمند').count();
check('narrative card present', narrative > 0);

// -- horizon buttons actually change the numbers
const kpiBefore = await page.locator('.kpi-value').first().innerText();
await page.getByRole('button', { name: '7D' }).click();
await page.waitForTimeout(2500);
const kpiAfter = await page.locator('.kpi-value').first().innerText();
check('horizon 7D changes the forecast total', kpiBefore !== kpiAfter, `${kpiBefore} -> ${kpiAfter}`);
await page.getByRole('button', { name: '30D' }).click();
await page.waitForTimeout(2000);

// -- entity selector filters
const selects = page.locator('select');
const entitySelect = selects.nth(1);
const optionCount = await entitySelect.locator('option').count();
check('entity dropdown populated', optionCount > 1, `${optionCount} options`);
const totalBefore = await page.locator('.kpi-value').first().innerText();
await entitySelect.selectOption({ index: 1 });
await page.waitForTimeout(2500);
const chartTitle = await page.locator('h3').filter({ hasText: 'روند تقاضا' }).innerText();
check('selecting an entity updates the chart title', chartTitle.includes('—'), chartTitle);

// -- level switch
await selects.nth(0).selectOption('listing');
await page.waitForTimeout(2500);
const memberCount = await selects.nth(1).locator('option').count();
check('level switch repopulates entities', memberCount > 50, `${memberCount} listings`);
await selects.nth(0).selectOption('destination');
await page.waitForTimeout(2000);

await page.screenshot({ path: `${SP}/ui_dashboard.png`, fullPage: true });

// ---------------------------------------------------------------- forecasts
console.log('\n=== FORECASTS');
await go('/forecasts');
check('forecast chart drawn', (await page.locator('svg.recharts-surface').count()) > 0);
check('detail table populated', (await page.locator('table tbody tr').count()) > 0,
      `${await page.locator('table tbody tr').count()} rows`);
// clicking a table row should select that entity
const firstRow = page.locator('table tbody tr').first();
const rowLabel = (await firstRow.locator('td').first().innerText()).trim();
await firstRow.click();
await page.waitForTimeout(2500);
const heading = await page.locator('h3').first().innerText();
check('clicking a row selects that destination', heading.includes(rowLabel), `${rowLabel} -> ${heading}`);

// ---------------------------------------------------------------- scenarios
console.log('\n=== SCENARIOS');
await go('/scenarios');
const sliders = page.locator('input[type=range]');
const sliderCount = await sliders.count();
check('scenario levers present', sliderCount > 0, `${sliderCount} sliders`);
const runButton = page.getByRole('button', { name: /اجرای سناریو/ });
check('run button disabled before any change', await runButton.isDisabled());
// move the first slider (price)
await sliders.first().evaluate((el) => {
  const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
  setter.call(el, '-20');
  el.dispatchEvent(new Event('input', { bubbles: true }));
  el.dispatchEvent(new Event('change', { bubbles: true }));
});
await page.waitForTimeout(600);
check('run button enabled after a change', !(await runButton.isDisabled()));
await runButton.click();
await page.waitForTimeout(9000);
const impact = await page.locator('text=/اثر:/').count();
check('scenario result rendered', impact > 0);
const scenarioNums = await page.locator('.nums').allInnerTexts();
check('baseline and scenario totals shown', scenarioNums.filter((t) => /[0-9],?[0-9]/.test(t)).length >= 3);
check('scenario chart drawn', (await page.locator('svg.recharts-surface').count()) > 0);
await page.screenshot({ path: `${SP}/ui_scenario.png`, fullPage: true });

// -------------------------------------------------------------- backtesting
console.log('\n=== BACKTESTING');
await go('/backtesting');
check('actual vs predicted chart', (await page.locator('svg.recharts-surface').count()) > 0,
      `${await page.locator('svg.recharts-surface').count()} charts`);
check('coverage table populated', (await page.locator('table tbody tr').count()) > 0,
      `${await page.locator('table tbody tr').count()} rows`);
check('fold cards shown', (await page.locator('text=/valid /').count()) > 0);

// ------------------------------------------------------------------ models
console.log('\n=== MODELS');
await go('/models');
const lbRows = await page.locator('table tbody tr').count();
check('leaderboard populated', lbRows >= 5, `${lbRows} rows`);
check('champion highlighted', (await page.locator('tr.bg-brand-50\\/60').count()) > 0);
check('model registry listed', (await page.locator('text=فعال').count()) > 0);
check('censoring card shown', (await page.locator('text=سانسور تقاضا').count()) > 0);
check('technical transparency shown', (await page.locator('text=شفافیت فنی').count()) > 0);

// ---------------------------------------------------------------- data lab
console.log('\n=== DATA LAB');
await go('/data-lab');
check('upload step shown', (await page.locator('text=بارگذاری دیتاست').count()) > 0);
check('existing dataset listed', (await page.locator('text=synthetic_pol_e_chaharom').count()) > 0);
// select the existing dataset -> should profile it and show the mapping form
await page.locator('button').filter({ hasText: 'synthetic_pol_e_chaharom' }).first().click();
await page.waitForTimeout(6000);
const colRows = await page.locator('table tbody tr').count();
check('dataset profiled: columns listed', colRows > 10, `${colRows} columns`);
const mappingSelects = await page.locator('select').count();
check('mapping form appeared', mappingSelects >= 6, `${mappingSelects} selects`);
const targetValue = await page.locator('select').nth(1).inputValue();
check('target auto-suggested', targetValue.length > 0, targetValue);
await page.screenshot({ path: `${SP}/ui_datalab.png`, fullPage: true });

// ------------------------------------------------------- presentation mode
console.log('\n=== PRESENTATION MODE');
await go('/dashboard?presentation=true');
check('presentation class applied', (await page.locator('.presentation').count()) > 0);
const hiddenDebug = await page.locator('.debug-only').first().isVisible().catch(() => false);
check('debug chrome hidden', hiddenDebug === false);
await page.screenshot({ path: `${SP}/ui_presentation.png`, fullPage: true });

// ---------------------------------------------------------------- 404 page
console.log('\n=== ERROR STATES');
await go('/no-such-page');
check('404 page renders', (await page.locator('text=صفحه پیدا نشد').count()) > 0);

console.log('\n=== CONSOLE / PAGE ERRORS');
if (errors.length === 0) console.log('  none');
else errors.slice(0, 12).forEach((e) => console.log('  !!', e));

const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length} UI checks, ${failed.length} failed, ${errors.length} runtime errors`);
await browser.close();
process.exit(failed.length || errors.length ? 1 : 0);
