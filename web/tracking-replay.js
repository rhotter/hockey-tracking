(() => {
 'use strict';
 const video=document.getElementById('tracking-video'),canvas=document.getElementById('tracking-overlay'),ctx=canvas.getContext('2d');
 const clip=document.getElementById('tracking-clip'),method=document.getElementById('tracking-method'),boxes=document.getElementById('tracking-boxes');
 const originalMethods=[...method.options].map(o=>o.value);
 const catalog={
  'gb-tracking-0600':{frames:300,folder:'media/gb-tracking',methods:originalMethods,default:'sam2'},
  'gb-tracking-1800':{frames:300,folder:'media/gb-tracking',methods:originalMethods,default:'sam2'},
  'gb-reid-long-0900':{frames:900,folder:'media/gb-tracking',methods:['botsort','reid-trained','reid-generic','reid-trained-original'],default:'botsort'}
 };
 for(const [id,spec]of Object.entries(window.SportScansTrackingCatalog||{})){
  catalog[id]=spec;clip.add(new Option(spec.label,id));
  for(const [value,label]of Object.entries(spec.methodLabels))if(![...method.options].some(o=>o.value===value))method.add(new Option(label,value));
 }
 const query=new URLSearchParams(location.search);
 let initialSeekPending=true;
 if([...clip.options].some(o=>o.value===query.get('clip')))clip.value=query.get('clip');
 else if(catalog['pittsburgh-0600'])clip.value='pittsburgh-0600';
 if([...method.options].some(o=>o.value===query.get('method')))method.value=query.get('method');
 function availableMethods(){const spec=catalog[clip.value];for(const option of method.options){option.disabled=!spec.methods.includes(option.value);option.hidden=option.disabled;}if(method.selectedOptions[0].disabled)method.value=spec.default;}
 function updateAbout(){
  const spec=catalog[clip.value],vlm=method.value.startsWith('vlm-'),sam3=method.value.startsWith('sam3');
  const methodNote=document.getElementById('tracking-method-note');
  methodNote.textContent=spec.methodDescriptions?.[method.value]||'';
  const custom=Boolean(methodNote.textContent);
  document.getElementById('tracking-about-method').hidden=!custom;
  document.getElementById('tracking-about-sam3').hidden=custom||!sam3;
  document.getElementById('tracking-about-vlm').hidden=custom||!vlm;
  document.getElementById('tracking-about-original').hidden=custom||vlm||sam3||Boolean(spec.description);
  document.getElementById('tracking-about-new').hidden=custom||vlm||sam3||!spec.description||['learned','runtime'].includes(spec.about);
  document.getElementById('tracking-about-learned').hidden=custom||vlm||sam3||spec.about!=='learned';
  document.getElementById('tracking-about-runtime').hidden=custom||vlm||sam3||spec.about!=='runtime';
 }
 function loadVideo(){const name=clip.value,spec=catalog[name];if(initialSeekPending){initialSeekPending=false;const time=Number(query.get('t'));if(query.has('t')&&Number.isFinite(time)&&time>=0)video.addEventListener('loadedmetadata',()=>{if(clip.value===name)video.currentTime=Math.min(time,Math.max(0,video.duration-1/15));},{once:true});}video.poster=`${spec.folder}/${name}.jpg`;SportScansMedia.load(video,`${spec.folder}/${name}.mp4`);document.getElementById('tracking-description').textContent=spec.description||'Great Britain · Yorkshire Iron Rats / Highland Frost · July 26, 2026';updateAbout();}
 availableMethods();loadVideo();
 const status=document.getElementById('tracking-status'),error=document.getElementById('tracking-error'),sourceLink=document.getElementById('tracking-source');
 const sourceIds={pittsburgh:'JlBtScHdcg4',marylandwomen:'ILV4PoQFxGU',philadelphia:'3FNUq6K0Bcc',gb:'X83xW6iQKoc'};
 let data=null,generation=0,lastKey='';const cache=new Map();
 const clock=s=>`${Math.floor(s/60)}:${(s%60).toFixed(2).padStart(5,'0')}`;
 function fail(message){error.textContent=message;error.hidden=false;}
 function draw(time=video.currentTime){
  if(!data)return;
  if(typeof time!=='number')time=video.currentTime;
  const frame=Math.min(data.frames.length-1,Math.max(0,Math.floor((time+.001)*data.fps)));
  const key=`${generation}:${frame}:${boxes.checked}`;if(key===lastKey)return;lastKey=key;
  const row=data.frames[frame];
  // Keep long replays compact in memory; expand only the displayed frame.
  const objects=data.format==='compact_observed_boxes_v1'?row.map(o=>({id:`T${o[0]}`,bbox:o.slice(1).map(v=>v/data.coordinate_scale)})):row.objects;
  ctx.clearRect(0,0,960,540);
  if(boxes.checked)for(const object of objects){
   const [x,y,u,v]=object.bbox.map((a,i)=>a*(i%2?540:960));ctx.strokeStyle='#165b8e';ctx.lineWidth=1.5;ctx.strokeRect(x,y,u-x,v-y);
   ctx.font='13px Arial';const width=ctx.measureText(object.id).width+6;ctx.fillStyle='rgba(255,255,255,.92)';ctx.fillRect(x,Math.max(0,y-17),width,17);ctx.fillStyle='#111';ctx.fillText(object.id,x+3,Math.max(13,y-4));
  }
  status.textContent=`Recording ${clock(data.start_s+frame/data.fps)} · ${objects.length} tracked observations · ${method.selectedOptions[0].textContent}`;
  sourceLink.href=`https://www.youtube.com/watch?v=${data.source_id||catalog[data.clip]?.sourceId||sourceIds[data.clip.split('-')[0]]}&t=${Math.floor(data.start_s+frame/data.fps)}s`;sourceLink.hidden=false;
  document.body.dataset.trackingFrame=String(frame);document.body.dataset.trackingMethod=data.method;document.body.dataset.trackingClip=data.clip;
 }
 async function load(){
  const version=++generation,name=clip.value,selected=method.value;data=null;lastKey='';ctx.clearRect(0,0,960,540);error.hidden=true;sourceLink.hidden=true;status.textContent='Loading tracks…';
  const key=`${name}-${selected}`;
  try{
   if(!cache.has(key)){
    const response=await fetch(`${catalog[name].folder}/${key}.json`);if(!response.ok)throw Error('Track data could not load.');
    const result=await response.json();if(result.frames?.length!==catalog[name].frames||result.fps!==15||result.clip!==name||result.method!==selected)throw Error('Track data is incomplete.');
    if(result.format==='compact_observed_boxes_v1'){
     if(result.coordinate_scale!==10000)throw Error('Track coordinates are invalid.');
     for(const row of result.frames)for(const o of row){if(o.length!==5||!o.every(Number.isFinite)||!Number.isInteger(o[0])||o[0]<1)throw Error('Track data is incomplete.');}
    }
    cache.set(key,result);
    while(cache.size>3)cache.delete(cache.keys().next().value);
   }
   if(version!==generation)return;data=cache.get(key);draw();
  }catch(e){if(version===generation)fail(e.message);}
 }
 clip.addEventListener('change',()=>{availableMethods();loadVideo();load();});
 method.addEventListener('change',()=>{updateAbout();load();});boxes.addEventListener('change',draw);
 for(const event of ['timeupdate','seeked','loadeddata'])video.addEventListener(event,draw);
 video.addEventListener('error',()=>fail('The video could not load. Reload the page to try again.'));
 const stage=document.getElementById('tracking-stage'),full=document.getElementById('tracking-fullscreen');
 if(!stage.requestFullscreen)full.hidden=true;
 full.addEventListener('click',async()=>{try{await stage.requestFullscreen();}catch{fail('Full screen is unavailable in this browser.');}});
 for(const [id,step]of [['tracking-previous',-1],['tracking-next',1]])document.getElementById(id).addEventListener('click',()=>{if(!data)return;video.pause();const frame=Math.max(0,Math.min(data.frames.length-1,Math.floor((video.currentTime+.001)*15)+step));video.currentTime=(frame+.5)/15;});
 function tick(_now,metadata){draw(metadata?.mediaTime??video.currentTime);if('requestVideoFrameCallback'in video)video.requestVideoFrameCallback(tick);else requestAnimationFrame(tick);}
 load();tick();
})();
