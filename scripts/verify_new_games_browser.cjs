'use strict';
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const {chromium}=require(process.env.SPORT_SCANS_PLAYWRIGHT_PATH||'playwright');
const root=path.resolve(__dirname,'..'),base=process.env.SPORT_SCANS_SITE_URL||'http://127.0.0.1:8885';
const output=process.env.SPORT_SCANS_SITE_REPORT||path.join(root,'build/new-games-20261025/browser.json');
const resultData=JSON.parse(fs.readFileSync(path.join(root,'web/data/reliability/new-games-20261025/results.json')));
const sha=x=>crypto.createHash('sha256').update(x).digest('hex');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.SPORT_SCANS_CHROME||undefined,args:['--no-sandbox']});
 const result={url:base,checks:[],errors:[]};
 try{
  const page=await browser.newPage({viewport:{width:1280,height:900}});
  page.on('pageerror',e=>result.errors.push(e.message));
  page.on('response',r=>{if(r.status()>=400&&new URL(r.url()).origin===new URL(base).origin)result.errors.push(`${r.status()} ${new URL(r.url()).pathname}`);});
  await page.goto(base+'/new-games.html',{waitUntil:'networkidle'});
  const rows=await page.locator('#new-games-totals tbody tr').evaluateAll(rows=>rows.map(r=>Array.from(r.cells).map(c=>c.textContent.trim())));
  const expected=resultData.methods.map(r=>{const c=r.counts;return [`${c.correct} / ${c.same_total}`,`${c.wrong} / ${c.different_total}`,String(c.unknown_same),`${c.matched_observations} / ${c.reference_observations}`];});
  assert.deepEqual(rows.map(r=>r.slice(1)),expected);
  assert.equal(await page.locator('.new-game-replay').count(),4);
  const text=await page.locator('main').innerText();
  for(const phrase of ['no independent human verification','not full-game tracking accuracy','No fine-tuning'])assert(text.includes(phrase),phrase);
  for(const href of await page.locator('a[download]').evaluateAll(a=>a.map(x=>x.getAttribute('href')))){
   const r=await page.request.get(new URL(href,base).href);assert(r.ok());assert.equal(sha(await r.body()),sha(fs.readFileSync(path.join(root,'web',href))));
  }
  result.checks.push('Published counts and downloadable evidence agree with local JSON');
  for(const summary of await page.getByText('Review the reference photos',{exact:true}).all())await summary.click();
  assert.equal(await page.locator('figure img').count(),24);
  for(const im of await page.locator('figure img').all()){
   await im.scrollIntoViewIfNeeded();await im.evaluate(async i=>{if(!i.complete)await new Promise((resolve,reject)=>{i.onload=resolve;i.onerror=reject;});});
   assert(await im.evaluate(i=>i.naturalWidth===1280&&i.naturalHeight===720));
  }
  await page.setViewportSize({width:390,height:844});
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Mobile overflow');
  result.checks.push('All 24 reference photos load; report fits 390px mobile width');
  await page.setViewportSize({width:1280,height:900});
  for(const summary of await page.getByText('Review the reference photos',{exact:true}).all())await summary.click();
  fs.mkdirSync(path.dirname(output),{recursive:true});
  await page.screenshot({path:output.replace(/\.json$/,'-report.png'),fullPage:true});
  for(const clip of resultData.clips){
   await page.goto(base+`/tracking-replay.html?clip=${clip.clip}&method=existing-sam2-osnet-qwen2b`,{waitUntil:'networkidle'});
   await page.waitForFunction(c=>document.body.dataset.trackingClip===c,clip.clip);
   const video=page.locator('#tracking-video');
   await video.evaluate(async v=>{if(v.readyState<1)await new Promise(r=>v.addEventListener('loadedmetadata',r,{once:true}));v.currentTime=10;});
   await page.waitForFunction(()=>Number(document.body.dataset.trackingFrame)>=149);
   const duration=await video.evaluate(v=>v.duration);assert(duration>=89.9&&duration<=90.1);
   assert((await page.locator('#tracking-source').getAttribute('href')).includes(clip.source_id));
   for(const method of resultData.methods){
    await page.locator('#tracking-method').selectOption(method.method);
    await page.waitForFunction(m=>document.body.dataset.trackingMethod===m,method.method);
    assert(await page.locator('#tracking-error').isHidden());
    const data=JSON.parse(fs.readFileSync(path.join(root,'web/media/new-games-20261025',`${clip.clip}-${method.method}.json`)));
    const frame=Number(await page.locator('body').getAttribute('data-tracking-frame'));
    assert((await page.locator('#tracking-status').innerText()).includes(`${data.frames[frame].length} tracked observations`));
   }
   await page.locator('#tracking-method').selectOption('existing-sam2-osnet-qwen2b');
   await page.waitForFunction(()=>document.body.dataset.trackingMethod==='existing-sam2-osnet-qwen2b');
   assert(await page.locator('#tracking-overlay').evaluate(c=>c.getContext('2d').getImageData(0,0,c.width,c.height).data.some((v,i)=>i%4===3&&v>0)),'No visible overlay');
   const before=Number(await page.locator('body').getAttribute('data-tracking-frame'));
   await page.locator('#tracking-next').click();await page.waitForFunction(f=>Number(document.body.dataset.trackingFrame)===f+1,before);
   await page.locator('#tracking-boxes').uncheck();
   assert(await page.locator('#tracking-overlay').evaluate(c=>!c.getContext('2d').getImageData(0,0,c.width,c.height).data.some((v,i)=>i%4===3&&v>0)));
   await page.locator('#tracking-boxes').check();
   await video.evaluate(async v=>{v.muted=true;await v.play();});
   await page.waitForFunction(f=>Number(document.body.dataset.trackingFrame)>f+10,before);
   await video.evaluate(v=>v.pause());
   await page.locator('#tracking-stage').screenshot({path:output.replace(/\.json$/,`-${clip.clip}.png`)});
   result.checks.push(`${clip.clip}: 90 seconds, all four methods, correct original source, boxes, stepping and actual playback`);
  }
  assert.deepEqual(result.errors,[]);
 }finally{fs.mkdirSync(path.dirname(output),{recursive:true});fs.writeFileSync(output,JSON.stringify(result,null,2)+'\n');await browser.close();}
 console.log(JSON.stringify(result));
})().catch(e=>{console.error(e);process.exitCode=1;});
