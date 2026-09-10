import { chromium } from 'playwright';
import { existsSync } from 'node:fs';
import assert from 'node:assert/strict';

const base = process.env.BASE_URL ?? 'http://127.0.0.1:3000';
const executablePath = process.env.CHROME_BIN ?? ['/opt/google/chrome/chrome', '/usr/bin/chromium', '/opt/pw-browsers/chromium'].find(existsSync);
const browser = await chromium.launch({ ...(executablePath ? { executablePath } : {}), args: ['--no-sandbox'] });
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
const errors = [];
page.on('pageerror', error => errors.push(error.message));
try {
  await page.goto(`${base}/jury`, { waitUntil: 'networkidle' });
  await page.getByRole('heading', { name: /See demand forming/ }).waitFor();
  assert.match(await page.locator('.jury-metrics').innerText(), /14.79%/);
  assert.match(await page.locator('.jury-callout').innerText(), /46.3/);
  assert.equal(await page.locator('svg.recharts-surface').count() >= 2, true);
  const allFirst = await page.locator('.jury-action h3').innerText();
  await page.getByLabel('Province', { exact: true }).selectOption('tehran');
  const cityButton = page.locator('.jury-workspace tbody button').last();
  const cityName = await cityButton.innerText();
  await cityButton.click();
  assert.equal(await page.locator('.jury-action h3').innerText(), cityName);
  assert.equal(await page.locator('.jury-workspace tbody tr').count() > 0, true);
  await page.getByLabel('Current analyst hours / week').fill('10');
  await page.getByLabel('Pilot analyst hours / week').fill('5');
  await page.getByLabel('Loaded cost / hour').fill('20');
  await page.getByLabel('Monthly service + operating cost').fill('100');
  assert.match(await page.locator('output').innerText(), /333/);
  await page.getByLabel('Monthly service + operating cost').fill('500');
  assert.match(await page.locator('output').innerText(), /-67/);
  const [download] = await Promise.all([page.waitForEvent('download'), page.getByRole('link', { name: /Export all/ }).click()]);
  assert.equal(download.suggestedFilename(), 'novo-pulse-decision-queue.csv');
  const stream = await download.createReadStream();
  let csv = ''; for await (const chunk of stream) csv += chunk;
  assert.equal(csv.trim().split('\n').length, 322);
  assert.match(csv, /required_before_spend/);
  await page.screenshot({ path: '/tmp/novo-jury-desktop.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
  await page.screenshot({ path: '/tmp/novo-jury-mobile.png', fullPage: true });
  await page.goto(base, { waitUntil: 'networkidle' });
  await page.locator('.model-switcher').waitFor();
  assert.match(await page.locator('.jury-callout').innerText(), /۷ استان/);
  const before = await page.locator('.jury-callout').innerText();
  await page.getByRole('radio').first().click();
  await page.waitForFunction(previous => document.querySelector('.jury-callout')?.textContent !== previous, before);
  assert.equal(await page.locator('html').getAttribute('dir'), 'rtl');
  await page.getByRole('link', { name: 'شواهد و برنامه کسب‌وکار' }).click();
  await page.getByRole('heading', { name: /See demand forming/ }).waitFor();
  // API empty state must replace the numbers, never reuse a successful packet.
  await page.route('**/api/v1/pol4/jury/', route => route.fulfill({json:{available:false,data:null,detail:'Evidence is stale. Regenerate it.'}}));
  await page.reload({waitUntil:'networkidle'});
  await page.getByRole('heading', {name:'Evidence unavailable'}).waitFor();
  assert.match(await page.locator('body').innerText(), /stale/);
  // The offline deck must work without either server or any external requests.
  await page.goto(new URL('../../artifacts/pol4/pitch.html', import.meta.url).href);
  assert.equal(await page.locator('.slide.active').getAttribute('data-slide'), '0');
  await page.keyboard.press('ArrowRight');
  assert.equal(await page.locator('.slide.active').getAttribute('data-slide'), '1');
  await page.keyboard.press('n');
  assert.equal(await page.locator('.slide.active .notes').isVisible(), true);
  await page.keyboard.press('End');
  assert.equal(await page.locator('.slide.active').getAttribute('data-slide'), '7');
  await page.getByText('What has been proven commercially?', {exact: true}).click();
  assert.equal(await page.getByText(/No customer traction, booking lift/).isVisible(), true);
  await page.setViewportSize({width:1440,height:1000});
  await page.locator('details').evaluateAll(items => items.forEach(item => { item.open = true; }));
  await page.pdf({path:new URL('../../artifacts/pol4/pitch.pdf', import.meta.url).pathname,
                  format:'A4',landscape:true,printBackground:true,margin:{top:'10mm',bottom:'10mm',left:'10mm',right:'10mm'}});
  assert.deepEqual(errors, []);
  console.log(`PASS: charts, filtering (${allFirst}), review selection, CSV integrity, economics, mobile layout, RTL, model switch, stale state; no browser errors.`);
} finally {
  await browser.close();
}
