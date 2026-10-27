// Optional browser regression check. Install Playwright or set NODE_PATH to its runtime.
const fs=require('node:fs'),path=require('node:path');
const output=process.env.SOG_TEST_OUTPUT||'build/shots-browser';fs.mkdirSync(output,{recursive:true});
const {chromium}=require('playwright');
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.CHROME_BIN||undefined,headless:true,args:['--no-sandbox']});
 const page=await browser.newPage({viewport:{width:1365,height:1100}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
 const base=process.env.SOG_TEST_BASE||'http://127.0.0.1:8891',model=process.env.SOG_TEST_MODEL||'astra',folder=model.startsWith('terra')?'sog-terra-20261027':'sog-20261027',prefix=model==='terra-max'?'max-':'';await page.goto(base+'/shots.html?model='+model);await page.waitForFunction(()=>document.querySelector('#shots-clip').options.length===3);
 if(await page.locator('#shots-model').inputValue()!==model)throw Error('Wrong model selected');
 const report=await (await page.request.get(base+'/data/'+folder+'/'+prefix+'report.json')).json();
 const result=[];
 for(const id of ['nj-0900','pittsburgh-finals-0600','southjersey-finals-0600']){
  await page.selectOption('#shots-clip',id);await page.waitForFunction(()=>!document.querySelector('#shots-forward').disabled,{},{timeout:120000});
  const duration=await page.locator('video').evaluate(v=>v.duration);if(Math.abs(duration-(id==='nj-0900'?300:90))>0.1)throw Error('Wrong media duration');
  await page.selectOption('#shots-filter','all');const count=await page.locator('.event-button').count();
  const expected=report.clips.find(c=>c.id===id).totals;if(count!==expected.sog+expected.needs_review+expected.not_sog)throw Error('Attempt count differs from report');
  if(count){await page.locator('.event-button').first().click();await page.locator('video').evaluate(v=>v.pause());await page.waitForFunction(()=>!document.querySelector('video').seeking);await page.selectOption('#shots-speed','0.25');
   const before=await page.locator('video').evaluate(v=>v.currentTime);await page.click('#shots-forward');await page.waitForFunction(t=>Math.abs(document.querySelector('video').currentTime-t-1/15)<0.025,before);}
  if(count){await page.locator('#shots-detail summary').click();await page.locator('#shots-detail details button').last().click();await page.waitForFunction(()=>!document.querySelector('video').seeking);if(!await page.locator('video').evaluate(v=>v.paused))throw Error('Evidence review should stay paused');}
  result.push({id,duration,attempts:count,url:page.url(),errors:[...errors]});
 }
 await page.selectOption('#shots-clip','nj-0900');await page.waitForFunction(()=>!document.querySelector('#shots-forward').disabled);await page.selectOption('#shots-filter','all');await page.waitForFunction(()=>!document.querySelector('video').seeking);await page.screenshot({path:path.join(output,'site-desktop.png'),fullPage:true});
 const deep=page.url();await page.reload();await page.waitForFunction(()=>!document.querySelector('#shots-forward').disabled);if(page.url()!==deep)throw Error('Deep link changed');
 await page.waitForFunction(()=>!document.querySelector('video').seeking);await page.setViewportSize({width:390,height:844});await page.screenshot({path:path.join(output,'site-mobile.png'),fullPage:true});const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);if(overflow)throw Error('Mobile horizontal overflow');
 await page.selectOption('#shots-model',model.startsWith('terra')?'astra':'terra');await page.waitForFunction(()=>!document.querySelector('#shots-forward').disabled);if(!await page.locator('#shots-report').getAttribute('href').then(h=>h.includes(model.startsWith('terra')?'sog-20261027':'sog-terra-20261027')))throw Error('Model switch did not update downloads');
 await browser.close();if(errors.length)throw Error(errors.join('\n'));console.log(JSON.stringify({model,results:result,mobileOverflow:overflow,errors},null,2));
})().catch(e=>{console.error(e);process.exitCode=1});
