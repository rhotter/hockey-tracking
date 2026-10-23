// Verify the published evidence and actual video overlays, without authentication.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const { chromium } = require(process.env.SPORT_SCANS_PLAYWRIGHT_PATH || 'playwright');
const base = process.env.SPORT_SCANS_SITE_URL || 'http://127.0.0.1:8873';
const artifact = process.env.SPORT_SCANS_SITE_REPORT || 'build/sam3-browser.json';

(async () => {
  const result = { url: base, checked_at: new Date().toISOString(), checks: [], errors: [] };
  const browser = await chromium.launch({headless: true, executablePath: process.env.SPORT_SCANS_CHROME, args: ['--no-sandbox']});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
    page.setDefaultTimeout(60000);
    page.on('pageerror', e => result.errors.push(e.message));
    page.on('response', r => { if (r.url().startsWith(base) && r.status() >= 400) result.errors.push(`${r.status()} ${r.url()}`); });
    assert.equal((await page.goto(base + '/sam3.html')).status(), 200);
    await page.locator('details').evaluateAll(nodes => nodes.forEach(n => n.open = true));
    const pictures = page.locator('main img');
    for (const pic of await pictures.all()) {
      await pic.scrollIntoViewIfNeeded();
      await pic.evaluate(img => img.decode());
    }
    assert.equal(await pictures.count(), 10); // Four pose sheets and all three wrong-join pairs.
    assert.equal(await page.locator('#sam3-errors rect').count(), 6);
    const data = await page.evaluate(async () => (await fetch('data/reliability/sam3/results.json')).json());
    const native = data.tracking.native_totals;
    assert((await page.locator('main').innerText()).includes(`${native.correct}/${native.same_total}`));
    result.checks.push('Public research page, results JSON, four pose sheets and all three identity-error pairs load');
    await page.setViewportSize({width: 390, height: 844});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
    await page.screenshot({path: artifact.replace('.json', '-mobile.png'), fullPage: true});
    await page.setViewportSize({width: 1440, height: 1000});
    const methods = ['sam3-native'];
    if (data.tracking.sam31_regression) methods.push('sam31-native');
    for (const row of data.tracking.native_regression) {
      for (const method of methods) {
        await page.goto(`${base}/tracking-replay.html?clip=${row.clip}&method=${method}&t=10`);
        await page.waitForFunction(({clip, method}) => document.body.dataset.trackingClip === clip && document.body.dataset.trackingMethod === method && document.body.dataset.trackingFrame === '150' && document.getElementById('tracking-video').readyState >= 2, {clip: row.clip, method});
        assert(await page.locator('#tracking-about-sam3').isVisible());
        assert.equal(await page.locator('#tracking-about-new').isVisible(), false);
        const count = await page.evaluate(async ({clip, method}) => (await (await fetch(`media/tracking-fourhour/${clip}-${method}.json`)).json()).frames[150].length, {clip: row.clip, method});
        assert((await page.locator('#tracking-status').innerText()).includes(`${count} tracked observations`));
        assert(await page.locator('#tracking-overlay').evaluate(c => c.getContext('2d').getImageData(0, 0, c.width, c.height).data.some(v => v !== 0)));
        await page.locator('#tracking-next').click();
        await page.waitForFunction(() => document.body.dataset.trackingFrame === '151');
        result.checks.push(`${row.clip}/${method}: video, frame count, visible overlay and frame stepping agree`);
      }
    }
    await page.screenshot({path: artifact.replace('.json', '-replay.png'), fullPage: true});
    assert.equal(result.errors.length, 0);
    result.success = true;
  } catch (error) {
    result.success = false;
    result.failure = error.stack;
  } finally {
    await browser.close();
    fs.writeFileSync(artifact, JSON.stringify(result, null, 2) + '\n');
    console.log(JSON.stringify(result, null, 2));
    if (!result.success) process.exitCode = 1;
  }
})();
