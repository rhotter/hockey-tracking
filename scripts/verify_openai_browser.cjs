'use strict';
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const {chromium}=require(process.env.SPORT_SCANS_PLAYWRIGHT_PATH||'playwright');
const root=path.resolve(__dirname,'..'),base=process.env.SPORT_SCANS_SITE_URL||'http://127.0.0.1:8884';
const output=process.env.SPORT_SCANS_SITE_REPORT||path.join(root,'build/openai-20261024/browser.json');
const sha=x=>crypto.createHash('sha256').update(x).digest('hex');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.SPORT_SCANS_CHROME||undefined,args:['--no-sandbox']});
 const result={url:base,checks:[],errors:[]};
 try{
  const page=await browser.newPage({viewport:{width:1280,height:900}});
  page.on('pageerror',e=>result.errors.push(e.message));
  page.on('response',r=>{if(r.status()>=400&&new URL(r.url()).origin===new URL(base).origin)result.errors.push(`${r.status()} ${new URL(r.url()).pathname}`);});
  await page.goto(base+'/vlm.html#openai-results',{waitUntil:'networkidle'});
  await page.waitForFunction(()=>!document.getElementById('vlm-conclusion').textContent.includes('Loading'));
  assert.equal(await page.locator('#openai-video-table tbody tr').count(),3);
  assert.equal(await page.locator('#openai-video-table tbody tr').allTextContents().then(x=>x.map(s=>s.replace(/\s+/g,' ').trim()).join('|')),
   'GPT-6 Astra96 / 9669 / 690 / 6090|GPT-6.1 Sol96 / 9669 / 690 / 6090|Existing pipeline94 / 9665 / 690 / 6092');
  await page.locator('#openai-reconnect summary').click();
  assert.equal(await page.locator('#openai-reconnect-table tbody tr').count(),4);
  const rows=await page.locator('#openai-reconnect-table tbody tr').evaluateAll(rows=>rows.map(r=>Array.from(r.cells).map(c=>c.textContent.trim())));
  assert.deepEqual(rows,[['GPT-6 Astra','129','51 / 64','1','34','26'],['GPT-6.1 Sol','130','50 / 65','2','32','25'],['Qwen3-VL 8B','130','51 / 65','7','42','15'],['Qwen3-VL 235B','130','51 / 65','7','46','9']]);
  await page.locator('#openai-reproducibility summary').click();
  const text=await page.locator('#openai-results').innerText();
  for(const phrase of ['24,576-token','no independent human verification','Game statistics remain unverified','$16.49','one unresolved request reservation'])assert(text.includes(phrase),phrase);
  result.checks.push('Both result tables, incomplete requests, reference and stats limitations');
  const hrefs=await page.locator('#openai-results a[download]').evaluateAll(a=>a.map(x=>x.getAttribute('href')));
  for(const href of hrefs){const r=await page.request.get(new URL(href,base).href);assert(r.ok());assert.equal(sha(await r.body()),sha(fs.readFileSync(path.join(root,'web',href))));}
  result.checks.push(`${hrefs.length} public result downloads match local bytes`);
  fs.mkdirSync(path.dirname(output),{recursive:true});
  await page.locator('#openai-results').screenshot({path:output.replace(/\.json$/,'-desktop.png')});
  await page.setViewportSize({width:390,height:844});
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Mobile page overflow');
  await page.locator('#openai-results').screenshot({path:output.replace(/\.json$/,'-mobile.png')});
  result.checks.push('Expanded research tables fit a 390px mobile page with contained table scrolling');
  assert.deepEqual(result.errors,[]);
 }finally{fs.mkdirSync(path.dirname(output),{recursive:true});fs.writeFileSync(output,JSON.stringify(result,null,2)+'\n');await browser.close();}
 console.log(JSON.stringify(result));
})().catch(e=>{console.error(e);process.exitCode=1;});
