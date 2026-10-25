'use strict';
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const {chromium}=require(process.env.SPORT_SCANS_PLAYWRIGHT_PATH||'playwright');
const root=path.resolve(__dirname,'..'),base=process.env.SPORT_SCANS_SITE_URL||'http://127.0.0.1:8884';
const output=process.env.SPORT_SCANS_SITE_REPORT||path.join(root,'build/roboflow-20261025/browser.json');
const resultData=JSON.parse(fs.readFileSync(path.join(root,'web/data/reliability/roboflow-20261025/results.json')));
const sha=x=>crypto.createHash('sha256').update(x).digest('hex');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.SPORT_SCANS_CHROME||undefined,args:['--no-sandbox']});
 const result={url:base,checks:[],errors:[]};
 try{
  const page=await browser.newPage({viewport:{width:1280,height:900}});
  page.on('pageerror',e=>result.errors.push(e.message));
  page.on('response',r=>{if(r.status()>=400&&new URL(r.url()).origin===new URL(base).origin)result.errors.push(`${r.status()} ${new URL(r.url()).pathname}`);});
  await page.goto(base+'/roboflow.html',{waitUntil:'networkidle'});
  assert.equal(await page.locator('#roboflow-components tbody tr').count(),7);
  const rows=await page.locator('#roboflow-tracking tbody tr').evaluateAll(rows=>rows.map(r=>Array.from(r.cells).map(c=>c.textContent.trim())));
  const expected=[resultData.existing_pipeline,...resultData.tracking].map(r=>{const c=r.counts;return [`${c.correct} / ${c.same_total}`,`${c.wrong} / ${c.different_total}`,String(c.unknown_same),`${c.matched_observations} / ${c.reference_observations}`];});
  assert.deepEqual(rows.map(r=>r.slice(1)),expected);
  assert.equal(await page.locator('#roboflow-teams tbody tr').count(),4);
  const content=await page.locator('main').innerText();
  for(const phrase of ['no independent human verification','evaluation footage contributes unlabeled fitting samples','not independent validation','miles and speed remain disabled'])assert(content.includes(phrase),phrase);
  for(const href of await page.locator('a[download]').evaluateAll(a=>a.map(x=>x.getAttribute('href')))){
   const r=await page.request.get(new URL(href,base).href);assert(r.ok());assert.equal(sha(await r.body()),sha(fs.readFileSync(path.join(root,'web',href))));
  }
  result.checks.push('Published tables match JSON; evidence downloads match local bytes');
  await page.getByText('Compare team predictions with the frozen reference',{exact:true}).click();
  await page.locator('img').first().scrollIntoViewIfNeeded();
  await page.waitForFunction(()=>Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0));
  fs.mkdirSync(path.dirname(output),{recursive:true});
  await page.screenshot({path:output.replace(/\.json$/,'-report.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Mobile overflow');
  result.checks.push('Team reference images load; mobile report fits 390px');
  const href=await page.locator('#roboflow-replay').getAttribute('href');
  await page.setViewportSize({width:1280,height:900});
  await page.goto(new URL(href,base).href,{waitUntil:'networkidle'});
  await page.waitForFunction(()=>document.body.dataset.trackingMethod?.startsWith('roboflow-'));
  const video=page.locator('#tracking-video');
  await video.evaluate(async v=>{if(v.readyState<1)await new Promise(r=>v.addEventListener('loadedmetadata',r,{once:true}));v.currentTime=10;});
  await page.waitForFunction(()=>Number(document.body.dataset.trackingFrame)>=149);
  assert((await video.evaluate(v=>v.duration))>=299.9);
  assert(await page.locator('#tracking-error').isHidden());
  assert(await page.locator('#tracking-overlay').evaluate(c=>{const a=c.getContext('2d').getImageData(0,0,c.width,c.height).data;return a.some((v,i)=>i%4===3&&v>0);}),'No visible overlay');
  const before=Number(await page.locator('body').getAttribute('data-tracking-frame'));
  await page.locator('#tracking-next').click();
  await page.waitForFunction(f=>Number(document.body.dataset.trackingFrame)===f+1,before);
  await page.locator('#tracking-stage').screenshot({path:output.replace(/\.json$/,'-replay.png')});
  await page.locator('#tracking-boxes').uncheck();
  assert(await page.locator('#tracking-overlay').evaluate(c=>!c.getContext('2d').getImageData(0,0,c.width,c.height).data.some((v,i)=>i%4===3&&v>0)));
  await page.locator('#tracking-boxes').check();
  await page.locator('#tracking-method').selectOption('vlm-embedding-2b-joint');
  await page.waitForFunction(()=>document.body.dataset.trackingMethod==='vlm-embedding-2b-joint');
  assert(await page.locator('#tracking-error').isHidden());
  result.checks.push('Five-minute video, rendered boxes, frame stepping, box toggle and existing-baseline switch work');
  assert.deepEqual(result.errors,[]);
 }finally{fs.mkdirSync(path.dirname(output),{recursive:true});fs.writeFileSync(output,JSON.stringify(result,null,2)+'\n');await browser.close();}
 console.log(JSON.stringify(result));
})().catch(e=>{console.error(e);process.exitCode=1;});
