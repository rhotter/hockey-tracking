'use strict';
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {chromium}=require(process.env.SPORT_SCANS_PLAYWRIGHT_PATH||'playwright');
const root=path.resolve(__dirname,'..'),base=process.env.SPORT_SCANS_SITE_URL||'http://127.0.0.1:8885';
const output=process.env.SPORT_SCANS_SITE_REPORT||path.join(root,'build/overhead-20261025/browser.json');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.SPORT_SCANS_CHROME||undefined,args:['--no-sandbox']});
 const result={url:base,checks:[],errors:[]};
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  page.on('pageerror',e=>result.errors.push(e.message));
  page.on('response',r=>{if(r.status()>=400&&new URL(r.url()).origin===new URL(base).origin)result.errors.push(`${r.status()} ${new URL(r.url()).pathname}`);});
  await page.goto(base+'/overhead.html',{waitUntil:'networkidle',timeout:120000});await page.waitForFunction(()=>document.body.dataset.overheadReady==='true');
  const replay=JSON.parse(fs.readFileSync(path.join(root,'web/data/overhead-20261025/replay.json'))),report=JSON.parse(fs.readFileSync(path.join(root,'web/data/overhead-20261025/report.json')));
  assert.equal(await page.locator('#trajectory-player').inputValue(),replay.default_player);
  assert((await page.locator('#coverage-text').innerText()).includes((report.selected_mapped_frames/15).toFixed(1)+'s'));
  const video=page.locator('#overhead-video');await video.evaluate(async v=>{if(v.readyState<1)await new Promise(r=>v.addEventListener('loadedmetadata',r,{once:true}));});
  assert(Math.abs(await video.evaluate(v=>v.duration)-300)<.1);
  const mapped=replay.frames.find(r=>r.objects.some(o=>o.player_id===replay.default_player&&o.xy));
  async function seek(f){await video.evaluate((v,f)=>{v.currentTime=f/15+.00001;},f);await page.waitForFunction(f=>Number(document.body.dataset.overheadFrame)===f,f);}
  await seek(mapped.frame);assert((await page.locator('#selected-status').innerText()).includes('mapped now'));
  assert(await page.locator('#overhead-overlay').evaluate(c=>c.getContext('2d').getImageData(0,0,c.width,c.height).data.some((v,i)=>i%4===3&&v>0)));
  await page.locator('#frame-next').click();await page.waitForFunction(f=>Number(document.body.dataset.overheadFrame)===f+1,mapped.frame);
  await seek(480);assert((await page.locator('#camera-status').innerText()).includes('unavailable'));assert((await page.locator('#selected-status').innerText()).includes('not confidently tracked'));
  await seek(1575);await page.locator('#show-calibration').check();await page.locator('#show-others').uncheck();
  fs.mkdirSync(path.dirname(output),{recursive:true});await page.screenshot({path:output.replace(/\.json$/,'-replay.png'),fullPage:true});
  await page.locator('#show-calibration').uncheck();await page.locator('#trajectory-window').selectOption('15');await page.locator('#trajectory-window').selectOption('all');
  await page.getByText('Player evidence',{exact:true}).click();
  for(const img of await page.locator('#player-evidence img').all()){await img.scrollIntoViewIfNeeded();await img.evaluate(async i=>{if(!i.complete)await new Promise((r,j)=>{i.onload=r;i.onerror=j;});});assert(await img.evaluate(i=>i.naturalWidth>0));}
  await page.locator('#player-evidence button').first().click();
  await page.locator('#trajectory-image').evaluate(a=>a.onclick());assert((await page.locator('#trajectory-image').getAttribute('href')).startsWith('data:image/png;base64,'));
  const csv=await page.locator('#trajectory-csv').evaluate(async a=>(await fetch(a.href)).text());assert.equal(csv.trim().split('\n').length,4501);assert(csv.includes('not_tracked'));
  const response=await page.request.get(base+'/data/overhead-20261025/report.json');assert(response.ok());assert.deepEqual(await response.json(),report);
  result.checks.push('Complete 300-second video, synchronized boxes and map, frame stepping, cut withholding, path controls, evidence photos, 4,500-row CSV and PNG export');
  if(replay.players.length>1){await page.locator('#trajectory-player').selectOption(replay.players[1].id);assert((await page.locator('#identity-note').innerText()).includes('Automatic track fragment'));await page.locator('#trajectory-player').selectOption(replay.default_player);}
  await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Mobile overflow');
  await video.evaluate(async v=>{v.muted=true;await v.play();});const before=Number(await page.locator('body').getAttribute('data-overhead-frame'));await page.waitForFunction(f=>Number(document.body.dataset.overheadFrame)>f+10,before);await video.evaluate(v=>v.pause());
  await page.screenshot({path:output.replace(/\.json$/,'-mobile.png'),fullPage:true});result.checks.push('390px mobile layout, fragment selection, actual playback');
  assert(await page.locator('#overhead-error').isHidden());assert.deepEqual(result.errors,[]);
 }finally{fs.mkdirSync(path.dirname(output),{recursive:true});fs.writeFileSync(output,JSON.stringify(result,null,2)+'\n');await browser.close();}
 console.log(JSON.stringify(result));
})().catch(e=>{console.error(e);process.exitCode=1;});
