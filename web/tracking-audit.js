(() => {
 'use strict';
 const panel=document.getElementById('tracking-audit');if(!panel)return;
 const clip=document.getElementById('audit-clip'),method=document.getElementById('audit-method'),kind=document.getElementById('audit-kind');
 const status=document.getElementById('audit-status'),evidence=document.getElementById('audit-evidence');
 const previous=document.getElementById('audit-prev'),next=document.getElementById('audit-next');
 let loaded=false,loading=false,data,reference,freeze,observations,frames,pairs=[],index=0;
 const clock=s=>`${Math.floor(s/60)}:${(s%60).toFixed(2).padStart(5,'0')}`;
 function render(){
  evidence.replaceChildren();previous.disabled=pairs.length<2;next.disabled=pairs.length<2;
  if(!pairs.length){status.textContent='No checks of this kind for this clip and method.';return;}
  const pair=pairs[index],ref=reference.pairs.find(p=>p.id===pair.id);
  const outcome=pair.predicted_same===null?'At least one reviewed observation was not matched.':pair.predicted_same?'One automatic identity at both times.':'Different automatic identities at the two times.';
  status.textContent=`Check ${index+1} of ${pairs.length}. Reference: ${pair.truth}. ${outcome}`;
  for(const side of ['a','b']){
   const obs=observations.get(ref[side]),frame=frames.get(`${obs.clip}:${obs.frame}`);
   const figure=document.createElement('figure'),stage=document.createElement('div');stage.className='audit-image';
   const image=document.createElement('img');image.alt=`Reviewed ${obs.team} player at recording ${clock(obs.source_time_s)}`;
   image.src=`data/reliability/four-hour-reference-frames/${obs.clip}-${String(obs.frame).padStart(4,'0')}.jpg`;stage.append(image);
   const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox',`0 0 ${frame.width} ${frame.height}`);svg.setAttribute('aria-hidden','true');
   const rect=document.createElementNS(svg.namespaceURI,'rect');const [x,y,u,v]=obs.bbox;
   for(const [key,value]of Object.entries({x,y,width:u-x,height:v-y,fill:'none',stroke:'#ffdf20','stroke-width':4}))rect.setAttribute(key,value);
   svg.append(rect);stage.append(svg);figure.append(stage);
   const caption=document.createElement('figcaption'),track=pair[`track_${side}`];
   caption.textContent=`Recording ${clock(obs.source_time_s)} · ${track===null?'No matched track':`Automatic ID ${track}`}`;
   const spec=freeze.methods.find(m=>m.id===method.value);
   if(!clip.value.includes('stress')||spec.publish_stress){
    const link=document.createElement('a');link.textContent='Open replay here';
    link.href=`tracking-replay.html?clip=${encodeURIComponent(clip.value)}&method=${encodeURIComponent(method.value)}&t=${obs.frame/15}`;
    caption.append(document.createElement('br'),link);
   }
   figure.append(caption);evidence.append(figure);
  }
 }
 function select(){
  if(!loaded)return;
  const row=data.clips.find(r=>r.clip===clip.value&&r.method===method.value);
  pairs=row.pairs.filter(p=>kind.value==='lost'?p.truth==='same'&&p.predicted_same!==true:kind.value==='missing'?p.truth==='same'&&p.predicted_same===null:kind.value==='split'?p.truth==='same'&&p.predicted_same===false:kind.value==='wrong'?p.truth==='different'&&p.predicted_same===true:kind.value==='unknown'?p.truth==='uncertain'&&p.predicted_same===true:p.truth==='same'&&p.predicted_same===true);
  index=0;render();
 }
 panel.addEventListener('toggle',async()=>{
  if(!panel.open||loaded||loading)return;loading=true;status.textContent='Loading reference checks…';
  try{
   [data,reference,freeze]=await Promise.all(['tracking-four-hour-final-results.json','tracking-four-hour-final-reference-v1.json','tracking-four-hour-final-freeze-v1.json'].map(async name=>{const response=await fetch(`data/reliability/${name}`);if(!response.ok)throw Error('Reference files could not load.');return response.json();}));
   if(reference.freeze_sha256!==data.freeze_sha256)throw Error('Reference version mismatch.');
   observations=new Map(reference.observations.map(o=>[o.id,o]));frames=new Map(reference.frames.map(f=>[`${f.clip}:${f.frame}`,f]));
   for(const name of freeze.final_test_clips)clip.add(new Option(name.replaceAll('-',' '),name));
   for(const m of freeze.methods)method.add(new Option(m.label,m.id));method.value=freeze.primary_method;
   loaded=true;select();
  }catch(error){status.textContent=error.message;}finally{loading=false;}
 });
 for(const input of [clip,method,kind])input.addEventListener('change',select);
 previous.addEventListener('click',()=>{if(pairs.length){index=(index+pairs.length-1)%pairs.length;render();}});
 next.addEventListener('click',()=>{if(pairs.length){index=(index+1)%pairs.length;render();}});
})();
