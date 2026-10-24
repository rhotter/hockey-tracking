'use strict';
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const {chromium}=require(process.env.SPORT_SCANS_PLAYWRIGHT_PATH||'playwright');
const root=path.resolve(__dirname,'..');
const base=process.env.SPORT_SCANS_SITE_URL||'http://127.0.0.1:8873';
const catalog=JSON.parse(fs.readFileSync(path.join(root,'web/data/reliability/sam3-20261024/replay-catalog.json'),'utf8'));
const output=process.env.SPORT_SCANS_SITE_REPORT||path.join(root,'build/sam3-20261024/browser.json');
const sha=buffer=>crypto.createHash('sha256').update(buffer).digest('hex');
function local(rel){for(const dir of ['web','public']){const p=path.join(root,dir,rel);if(fs.existsSync(p))return fs.readFileSync(p);}throw Error('Missing local asset '+rel);}
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.SPORT_SCANS_CHROME||undefined,args:['--no-sandbox']});
 const result={url:base,verified_at:new Date().toISOString(),checks:[],errors:[]};
 try{
  const page=await browser.newPage({viewport:{width:1280,height:900}});
  page.on('pageerror',e=>result.errors.push(e.message));
  page.on('response',r=>{if(r.status()>=400&&new URL(r.url()).origin===new URL(base).origin)result.errors.push(`${r.status()} ${new URL(r.url()).pathname}`);});
  await page.goto(base+'/sam3-improvements.html',{waitUntil:'networkidle'});
  assert.equal(await page.locator('h1').innerText(),'SAM3 follow-up');
  const summary=JSON.parse(local('data/reliability/sam3-20261024/results.json'));
  assert((await page.locator('main').innerText()).includes((summary.detection[1].metrics.f1*100).toFixed(1)+'%'));
  for(const detail of await page.locator('details').all())await detail.evaluate(d=>d.open=true);
  for(const img of await page.locator('main img').all()){
   await img.scrollIntoViewIfNeeded();await img.evaluate(i=>i.decode());
   assert(await img.evaluate(i=>i.naturalWidth>0));
  }
  for(const href of await page.locator('a[href*="rink-filter-review-"]').evaluateAll(a=>a.map(x=>x.getAttribute('href')))){
   const response=await page.request.get(new URL(href,base).href);assert(response.ok());
   assert.equal(sha(await response.body()),sha(local(href)));
  }
  for(const href of await page.locator('a[href$=".json"]').evaluateAll(a=>a.map(x=>x.getAttribute('href')))){
   const response=await page.request.get(new URL(href,base).href);assert(response.ok());assert(await response.json());
  }
  await page.setViewportSize({width:390,height:844});
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Research page overflow');
  await page.screenshot({path:output.replace(/\.json$/, '-research-mobile.png'),fullPage:true});
  result.checks.push('Research tables, all review images, evidence links and expanded mobile layout');
  await page.setViewportSize({width:1280,height:900});
  for(const clip of catalog){
   const first=clip.methods[0].id;
   await page.goto(`${base}/tracking-replay.html?clip=${clip.clip}&method=${first}`,{waitUntil:'domcontentloaded'});
   await page.waitForFunction(({id,method})=>document.body.dataset.trackingClip===id&&document.body.dataset.trackingMethod===method&&document.getElementById('tracking-video').readyState>=2,{id:clip.clip,method:first});
   assert.equal(Math.round(await page.locator('#tracking-video').evaluate(v=>v.duration*15)),clip.frames);
   for(const method of clip.methods){
    await page.locator('#tracking-method').selectOption(method.id);
    await page.waitForFunction(m=>document.body.dataset.trackingMethod===m,method.id);
    const folder=clip.clip==='pittsburgh-sam3-20min'?'media/sam3-20261024':'media/tracking-fourhour';
    const rel=`${folder}/${clip.clip}-${method.id}.json`,bytes=local(rel),data=JSON.parse(bytes);
    const response=await page.request.get(base+'/'+rel);assert(response.ok());assert.equal(sha(await response.body()),sha(bytes));
    const frames=clip.frames===18225?[0,4500,4612,4724,9000,9112,9224,11025,13500,13612,13724,18224]:[0,150,Math.floor(clip.frames/2),clip.frames-1];
    for(const frame of frames){
     await page.locator('#tracking-video').evaluate((v,f)=>v.currentTime=(f+.1)/15,frame);
     await page.waitForFunction(f=>Number(document.body.dataset.trackingFrame)===f&&!document.getElementById('tracking-video').seeking&&document.getElementById('tracking-video').readyState>=2,frame);
     const status=await page.locator('#tracking-status').innerText();assert(status.includes(`${data.frames[frame].length} tracked observations`));
     const link=new URL(await page.locator('#tracking-source').getAttribute('href'));
     assert.equal(link.searchParams.get('t'),Math.floor(data.start_s+frame/15)+'s');
     const drawn=await page.locator('#tracking-overlay').evaluate(c=>c.getContext('2d').getImageData(0,0,c.width,c.height).data.some(v=>v!==0));
     assert.equal(drawn,data.frames[frame].length>0);
    }
    if(method.id.startsWith('sam3')){
     assert(await page.locator('#tracking-about-method').isVisible());
     assert((await page.locator('#tracking-method-note').textContent()).length>50);
    }
    await page.locator('#tracking-previous').click();
    await page.waitForFunction(f=>Number(document.body.dataset.trackingFrame)===f&&!document.getElementById('tracking-video').seeking,clip.frames-2);
    await page.locator('#tracking-next').click();
    await page.waitForFunction(f=>Number(document.body.dataset.trackingFrame)===f&&!document.getElementById('tracking-video').seeking,clip.frames-1);
    result.checks.push({clip:clip.clip,method:method.id,frames,track_hash:sha(bytes),source_time_and_observation_counts:true});
   }
  }
  await page.goto(base+'/tracking-replay.html?clip=pittsburgh-sam3-20min&method=sam3-long&t=735',{waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>document.body.dataset.trackingClip==='pittsburgh-sam3-20min'&&document.body.dataset.trackingFrame==='11025'&&!document.getElementById('tracking-video').seeking);
  await page.locator('#tracking-boxes').uncheck();
  assert.equal(await page.locator('#tracking-overlay').evaluate(c=>c.getContext('2d').getImageData(0,0,c.width,c.height).data.some(v=>v!==0)),false);
  await page.locator('#tracking-boxes').check();
  await page.locator('#tracking-video').evaluate(v=>v.play());
  await page.waitForFunction(()=>Number(document.body.dataset.trackingFrame)>11030);
  await page.locator('#tracking-video').evaluate(v=>v.pause());
  await page.screenshot({path:output.replace(/\.json$/, '-replay-desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await page.locator('#tracking-about-method').evaluate(d=>d.open=true);
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Replay overflow');
  await page.screenshot({path:output.replace(/\.json$/, '-replay-mobile.png'),fullPage:true});
  result.checks.push('20m 15s replay: direct audit link, playback synchronization, boxes toggle, desktop and mobile');
  assert.deepEqual(result.errors,[]);result.success=true;
 }catch(error){result.success=false;result.failure=error.stack;}
 finally{await browser.close();fs.mkdirSync(path.dirname(output),{recursive:true});fs.writeFileSync(output,JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));if(!result.success)process.exitCode=1;}
})().catch(error=>{console.error(error.stack);process.exitCode=1;});
