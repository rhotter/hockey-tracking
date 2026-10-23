(() => {
 'use strict';
 const video=document.getElementById('rink-video'), overlay=document.getElementById('rink-overlay'), ctx=overlay.getContext('2d');
 SportScansMedia.load(video,video.dataset.mediaSrc);
 const selector=document.getElementById('selected-track'), boxes=document.getElementById('show-boxes'), status=document.getElementById('replay-status'), trackStatus=document.getElementById('track-status'), group=document.getElementById('rink-players');
 const NS='http://www.w3.org/2000/svg';let data=null,lastFrame=-1,lastSelection=null,lastBoxes=null;
 const color=o=>o.color==='red'?'#b82b36':o.color==='white'?'#1c548c':'#555';
 const clock=s=>`${Math.floor(s/60)}:${(s%60).toFixed(2).padStart(5,'0')}`;
 function fail(message){const el=document.getElementById('replay-error');el.textContent=message;el.hidden=false;}
 function draw(time=video.currentTime){
  if(!data)return;
  if(typeof time!=='number')time=video.currentTime;
  const index=Math.min(data.frames.length-1,Math.max(0,Math.floor((time+0.001)*data.fps)));
  const selected=selector.value;
  if(index===lastFrame&&selected===lastSelection&&boxes.checked===lastBoxes)return;
  lastFrame=index;lastSelection=selected;lastBoxes=boxes.checked;
  const row=data.frames[index];ctx.clearRect(0,0,overlay.width,overlay.height);group.replaceChildren();
  let mapped=0;
  for(const o of row.objects){
   const active=!selected||selected===o.id;
   if(boxes.checked){const [x,y,u,v]=o.bbox.map((a,i)=>a*(i%2?720:1280));ctx.globalAlpha=active?1:.3;ctx.strokeStyle=color(o);ctx.lineWidth=selected===o.id?3:1.5;ctx.strokeRect(x,y,u-x,v-y);ctx.font='15px Arial';ctx.fillStyle='rgba(255,255,255,.9)';ctx.fillRect(x,Math.max(0,y-20),ctx.measureText(o.id).width+8,19);ctx.fillStyle=color(o);ctx.fillText(o.id,x+4,Math.max(15,y-5));}
   if(!row.camera_valid||!o.map_xy)continue;
   mapped++;const [x,y]=o.map_xy;const node=document.createElementNS(NS,'g');node.setAttribute('class','track-dot');node.setAttribute('role','button');node.setAttribute('tabindex','0');node.setAttribute('aria-label',`Highlight ${o.id}`);node.setAttribute('opacity',active?'1':'.25');
   const dot=document.createElementNS(NS,'circle');dot.setAttribute('cx',28+x*624);dot.setAttribute('cy',28+y*310);dot.setAttribute('r',selected===o.id?'8':'6');dot.setAttribute('fill',color(o));dot.setAttribute('stroke',selected===o.id?'#111':'white');dot.setAttribute('stroke-width','2');node.append(dot);
   const text=document.createElementNS(NS,'text');text.setAttribute('x',38+x*624);text.setAttribute('y',24+y*310);text.setAttribute('font-size','13');text.textContent=o.id;node.append(text);
   const choose=()=>{selector.value=selected===o.id?'':o.id;draw();};node.addEventListener('click',choose);node.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose();}});group.append(node);
  }
  ctx.globalAlpha=1;
  status.textContent=`Recording ${clock(data.start_s+index/data.fps)} · ${mapped} mapped / ${row.objects.length} tracked in this frame${row.camera_valid?'':' · Camera position unavailable'}`;
  const object=row.objects.find(o=>o.id===selected);
  trackStatus.textContent=selected?(!object?`${selected} has no observation in this frame.`:!row.camera_valid?`${selected}: camera position unavailable.`:!object.map_xy?`${selected}: no accepted foot estimate.`:`${selected}: estimated position shown.`):'Select a dot or track to highlight it in the video.';
  document.body.dataset.replayFrame=String(index);document.body.dataset.mappedCount=String(mapped);
 }
 function frame(_now,metadata){draw(metadata?.mediaTime??video.currentTime);if('requestVideoFrameCallback'in video)video.requestVideoFrameCallback(frame);else requestAnimationFrame(frame);}
 video.addEventListener('timeupdate',draw);video.addEventListener('seeked',draw);video.addEventListener('loadeddata',draw);video.addEventListener('error',()=>fail('The replay video could not load. Reload the page to try again.'));
 selector.addEventListener('change',draw);boxes.addEventListener('change',draw);document.getElementById('replay-speed').addEventListener('change',e=>{video.playbackRate=Number(e.target.value);});
 for(const [id,step]of [['previous-frame',-1],['next-frame',1]])document.getElementById(id).addEventListener('click',()=>{if(!data)return;video.pause();const frame=Math.floor((video.currentTime+.001)*data.fps);video.currentTime=(Math.max(0,Math.min(data.frames.length-1,frame+step))+.5)/data.fps;});
 const stage=document.getElementById('rink-stage'),full=document.getElementById('rink-fullscreen');
 if(!stage.requestFullscreen)full.hidden=true;
 full.addEventListener('click',async()=>{try{await stage.requestFullscreen();}catch{fail('Full screen is unavailable in this browser.');}});
 fetch('data/replay/nj-0900-sam2.json').then(r=>{if(!r.ok)throw Error('Position data could not load.');return r.json();}).then(d=>{
  if(!d.frames?.length||d.fps!==15)throw Error('Position data is incomplete.');data=d;
  for(const id of Object.values(d.track_aliases)){const opt=document.createElement('option');opt.value=id;opt.textContent=id;selector.append(opt);}
  const b=d.calibration.relative_bounds;const x=v=>28+(v-b.left)/(b.right-b.left)*624;
  document.getElementById('center-line').setAttribute('d',`M${x(0)} 28V338`);document.getElementById('blue-lines').setAttribute('d',`M${x(-1)} 28V338 M${x(1)} 28V338`);draw();frame();
 }).catch(e=>fail(e.message));
})();
