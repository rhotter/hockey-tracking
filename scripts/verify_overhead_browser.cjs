'use strict';
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {chromium}=require(process.env.SPORT_SCANS_PLAYWRIGHT_PATH||'playwright');
const ballMap=require('../web/ball-map.js');
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
  async function decoded(f){await page.waitForFunction(f=>{const v=document.getElementById('overhead-video');return !v.seeking&&v.readyState>=2&&Number(document.body.dataset.overheadFrame)===f;},f);}
  async function seek(f){await video.evaluate((v,f)=>{v.currentTime=f/15+.00001;},f);await decoded(f);}
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
  const ballFolder=path.join(root,'web/data/ball-sequence-20261026');
  const ballReport=JSON.parse(fs.readFileSync(path.join(ballFolder,'report.json'))),ballData=JSON.parse(fs.readFileSync(path.join(ballFolder,'positions.json')));
  const rawBallData=JSON.parse(fs.readFileSync(path.join(ballFolder,'replay.json')));
  const previousBall=JSON.parse(fs.readFileSync(path.join(root,'web/data/ball-rebuild-20261026/replay.json')));
  const ballResponse=await page.request.get(base+'/data/ball-sequence-20261026/report.json');assert(ballResponse.ok());assert.deepEqual(await ballResponse.json(),ballReport);
  const positionsResponse=await page.request.get(base+'/data/ball-sequence-20261026/positions.json');assert(positionsResponse.ok());assert.deepEqual(await positionsResponse.json(),ballData);
  const parseCSV=text=>{const rows=[];let row=[],cell='',quoted=false;for(let i=0;i<text.length;i++){const c=text[i];if(c==='"'){if(quoted&&text[i+1]==='"'){cell+='"';i++;}else quoted=!quoted;}else if(c===','&&!quoted){row.push(cell);cell='';}else if(c==='\n'&&!quoted){row.push(cell);rows.push(row);row=[];cell='';}else cell+=c;}return rows;};
  await page.locator('[data-ball-seek="2850"]').click();await decoded(2850);
  assert.equal(await page.locator('body').getAttribute('data-ball-state'),ballData.frames[2850].state);
  assert.equal(ballData.frames[2850].state,'detected');
  assert(!(await page.locator('#ball-status').innerText()).includes('error radius'));
  assert(await page.locator('#ball-zoom').evaluate(c=>new Set(c.getContext('2d').getImageData(0,0,c.width,c.height).data).size>50));
  await page.locator('#ball-mode').selectOption('baseline');await seek(910);
  assert.equal(await page.locator('body').getAttribute('data-ball-state'),previousBall.frames[910].state);
  const oldCSV=parseCSV(await page.locator('#ball-csv').evaluate(async a=>(await fetch(a.href)).text()));
  assert.equal(oldCSV.length,4501);assert.equal(oldCSV.filter(r=>r[3]==='detected').length,284);assert.equal(oldCSV.filter(r=>r[3]==='inferred').length,20);
  await page.locator('#ball-mode').selectOption('off');assert.equal(await page.locator('body').getAttribute('data-ball-state'),'off');assert.equal(await page.locator('body').getAttribute('data-ball-map-method'),'off');
  await page.locator('#ball-mode').selectOption('context');
  await page.locator('[data-ball-seek="910"]').click();await decoded(910);
  assert.equal(await page.locator('body').getAttribute('data-ball-state'),'context');
  assert((await page.locator('#ball-status').innerText()).includes('Context estimate'));
  assert((await page.locator('#ball-evidence').innerText()).length>10);
  assert(await page.locator('#overhead-map').evaluate(c=>{const a=c.getContext('2d').getImageData(0,0,c.width,c.height).data;for(let y=0;y<c.height;y++)for(let x=0;x<c.width;x++){if(x>=45&&x<955&&y>=40&&y<495)continue;const i=(y*c.width+x)*4;if(a[i]===21&&a[i+1]===106&&a[i+2]===156)return false;}return true;}),'Ball marker escaped rink bounds');
  await page.screenshot({path:output.replace(/\.json$/,'-ball-sequence.png'),fullPage:true});
  const alternative=rawBallData.frames.find(r=>r.state==='context'&&r.hypotheses.length>1);
  assert(alternative);await seek(alternative.frame);assert(!(await page.locator('#ball-status').innerText()).includes('alternatives'));
  assert(await page.locator('#overhead-overlay').evaluate((c,pixel)=>{
   const a=c.getContext('2d').getImageData(0,0,c.width,c.height).data;let count=0;
   for(let y=0;y<c.height;y++)for(let x=0;x<c.width;x++){const i=(y*c.width+x)*4;
    if(a[i]===21&&a[i+1]===106&&a[i+2]===156&&a[i+3]>128){count++;if(Math.hypot(x-pixel[0]*1920,y-pixel[1]*1080)>10)return false;}}
   return count>10;
  },alternative.pixel),'Expected one small marker at the selected location, with no alternate marker or uncertainty ring');
  await page.screenshot({path:output.replace(/\.json$/,'-single-ball.png'),fullPage:true});
  const tracked=ballData.frames.find(r=>r.state==='tracked');assert(tracked);await seek(tracked.frame);
  assert.equal(await page.locator('body').getAttribute('data-ball-state'),'tracked');assert((await page.locator('#ball-status').innerText()).includes('video model'));
  async function checkMap(f,source){
   await seek(f);const estimate=source.frames[f],mapped=ballMap.position(estimate,replay.frames[f],replay.relative_bounds);
   assert.equal(await page.locator('body').getAttribute('data-ball-map-method'),mapped.method);
   if(mapped.xy){
    assert(await page.locator('#overhead-map').evaluate((c,{xy,state})=>{
     const color=state==='context'?[21,106,156]:state==='inferred'?[166,85,0]:[23,99,62];
     const ctx=c.getContext('2d'),px=Math.round(45+910*xy[0]),py=Math.round(40+455*xy[1]);
     const a=ctx.getImageData(px-4,py-4,9,9).data;let count=0;
     for(let i=0;i<a.length;i+=4)if(color.every((v,j)=>a[i+j]===v))count++;
     return count>15;
    },{xy:mapped.xy,state:estimate.state}),'Ball marker must actually appear at the projected map location');
   }
   return mapped;
  }
  for(const [f,support] of [[3368,'held'],[4446,'airborne']]){
   const mapped=await checkMap(f,ballData);assert.equal(mapped.method,'floor_approximation');
   assert.equal(ballData.frames[f].xy,null);
   assert((await page.locator('#ball-status').innerText()).includes(support));
   assert((await page.locator('#ball-map-status').innerText()).includes('approximate'));
   assert(await page.locator('#ball-zoom').evaluate(c=>new Set(c.getContext('2d').getImageData(0,0,c.width,c.height).data).size>50));
  }
  await checkMap(2614,ballData);assert((await page.locator('#ball-map-status').innerText()).includes('camera alignment unavailable'));
  await checkMap(1132,ballData);assert((await page.locator('#ball-map-status').innerText()).includes('outside the rink'));
  await checkMap(480,ballData);assert.equal(await page.locator('body').getAttribute('data-ball-state'),'unknown');
  const ballCSV=parseCSV(await page.locator('#ball-csv').evaluate(async a=>(await fetch(a.href)).text()));
  assert.equal(ballCSV.length,4501);
  assert(!ballCSV[0].some(h=>/alternatives|radius|covariance|weight/.test(h)));
  for(let f=0;f<4500;f++){const row=ballCSV[f+1],point=ballData.frames[f];assert.equal(row.length,16);assert.equal(row[0],String(f));assert.equal(row[3],point.state);assert.equal(row[6],point.pixel?String(point.pixel[0]*1920):'');assert.equal(row[7],point.pixel?String(point.pixel[1]*1080):'');assert(!['hypotheses','model_radius_px','covariance','pixel_covariance','known_weight'].some(k=>k in point));}
  assert((await page.locator('#ball-summary').innerText()).includes((ballReport.estimated_coverage*100).toFixed(1)+'%'));
  await page.getByText('Video-model checks and sequence summaries',{exact:true}).click();
  assert.equal(await page.locator('#ball-comparisons tbody tr').count(),ballReport.point_comparisons.length);
  assert.equal(await page.locator('#ball-review-points button').count(),ballReport.sequences.length);
  await page.locator('#ball-review-points button').last().click();await decoded(ballReport.sequences.at(-1).core[0]);
  result.checks.push('Sequence/previous/off modes, actual decoded-frame synchronization, one selected ball marker without alternate locations or uncertainty rings, visible video tracks, chronological explanations, held/airborne approximate diagram markers, missing-camera and off-rink explanations, complete single-position CSV and JSON, coverage summary, comparison and sequence navigation');
  const samFolder=path.join(root,'web/data/ball-sam-20261026'),samData=JSON.parse(fs.readFileSync(path.join(samFolder,'positions.json'))),samReport=JSON.parse(fs.readFileSync(path.join(samFolder,'report.json')));
  for(const [name,expected]of [['positions.json',samData],['report.json',samReport]]){const response=await page.request.get(base+'/data/ball-sam-20261026/'+name);assert(response.ok());assert.deepEqual(await response.json(),expected);}
  assert.equal(await page.locator('#ball-sam-comparisons tbody tr').count(),4);
  assert.equal(await page.locator('#ball-sam-result').innerText(),samReport.decision.summary);
  await page.locator('#ball-mode').selectOption('sam');
  const samFrame=samReport.full.audit_selection[0]?.frame??samData.frames.find(r=>r.state==='tracked').frame;
  await seek(samFrame);assert.equal(await page.locator('body').getAttribute('data-ball-state'),samData.frames[samFrame].state);
  assert((await page.locator('#ball-summary').innerText()).includes((samReport.estimated_coverage*100).toFixed(1)+'%'));
  const samCSV=parseCSV(await page.locator('#ball-csv').evaluate(async a=>(await fetch(a.href)).text()));assert.equal(samCSV.length,4501);
  for(let f=0;f<4500;f++){const r=samCSV[f+1],p=samData.frames[f];assert.equal(r.length,16);assert.equal(r[3],p.state);assert.equal(r[6],p.pixel?String(p.pixel[0]*1920):'');assert.equal(r[7],p.pixel?String(p.pixel[1]*1080):'');assert(!['hypotheses','model_radius_px','covariance','pixel_covariance'].some(k=>k in p));}
  for(let f=0;f<4500;f++){const p=ballMap.position(samData.frames[f],replay.frames[f],replay.relative_bounds);assert.equal(samCSV[f+1][13],p.xy?String(p.xy[0]):'');assert.equal(samCSV[f+1][14],p.xy?String(p.xy[1]):'');assert.equal(samCSV[f+1][15],p.method);}
  for(const f of [31,150,1478,1575,3368,4446])await checkMap(f,samData);
  await checkMap(1575,samData);
  await page.screenshot({path:output.replace(/\.json$/,'-sam-ball.png'),fullPage:true});
  await seek(480);assert.equal(await page.locator('body').getAttribute('data-ball-state'),'unknown');
  await page.locator('#ball-mode').selectOption('context');
  result.checks.push('SAM3 numeric report matches deployed source, comparison table, selectable hybrid replay, decoded-frame state, single-position CSV, cut reset and coverage');
  if(replay.players.length>1){await page.locator('#trajectory-player').selectOption(replay.players[1].id);assert((await page.locator('#identity-note').innerText()).includes('Automatic track fragment'));await page.locator('#trajectory-player').selectOption(replay.default_player);}
  await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Mobile overflow');
  await video.evaluate(async v=>{v.muted=true;await v.play();});const before=Number(await page.locator('body').getAttribute('data-overhead-frame'));await page.waitForFunction(f=>Number(document.body.dataset.overheadFrame)>f+10,before);await video.evaluate(v=>v.pause());
  await page.screenshot({path:output.replace(/\.json$/,'-mobile.png'),fullPage:true});result.checks.push('390px mobile layout, fragment selection, actual playback');
  await page.goto(base+'/overhead.html?ball=sam&t='+(samFrame/15+.00001),{waitUntil:'networkidle',timeout:120000});
  await page.waitForFunction(()=>document.body.dataset.overheadReady==='true');await decoded(samFrame);
  assert.equal(await page.locator('#ball-mode').inputValue(),'sam');assert.equal(await page.locator('body').getAttribute('data-ball-state'),samData.frames[samFrame].state);
  result.checks.push('Direct SAM replay link restores method and decoded timestamp');
  assert(await page.locator('#overhead-error').isHidden());assert.deepEqual(result.errors,[]);
 }finally{fs.mkdirSync(path.dirname(output),{recursive:true});fs.writeFileSync(output,JSON.stringify(result,null,2)+'\n');await browser.close();}
 console.log(JSON.stringify(result));
})().catch(e=>{console.error(e);process.exitCode=1;});
