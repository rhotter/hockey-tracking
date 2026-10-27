(() => {
 'use strict';
 const $=id=>document.getElementById(id), video=$('shots-video'), picker=$('shots-clip'), filter=$('shots-filter');
 let dataset, report, clip, selected, ready=false, generation=0, loopEnd=null;
 const statuses={sog:'Estimated SOG',needs_review:'Needs review',not_sog:'Not SOG'};
 const outcomes={goal:'Goal',save:'Saved',blocked:'Blocked',miss:'Missed',post:'Post / crossbar',unknown:'Outcome unclear',not_shot:'Not a shot',disallowed_goal:'Disallowed goal'};
 const time=s=>{const v=Math.max(0,Number(s)||0);return `${Math.floor(v/60)}:${(v%60).toFixed(1).padStart(4,'0')}`;};
 const el=(tag,text,className)=>{const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(className)node.className=className;return node;};
 const team=e=>clip.team_labels[e.team_color]||'Unknown team';
 const events=()=>dataset.events.filter(e=>e.clip===clip.id);
 function error(message){$('shots-error').textContent=message;$('shots-error').hidden=false;}
 function setURL(){const q=new URLSearchParams({clip:clip.id});if(selected)q.set('event',selected.id);history.replaceState(null,'',`${location.pathname}?${q}`);}
 function seek(seconds,play=false){if(!ready)return;video.currentTime=Math.min(Math.max(0,seconds),Math.max(0,video.duration-1/15));if(play)video.play().catch(()=>{});}
 function replay(play=true){if(!selected)return;loopEnd=Math.min(clip.duration_s,selected.outcome_frame/15+3);seek(Math.max(0,selected.clip_seconds-3),play);}
 function detail(event){
  selected=event;const target=$('shots-detail');target.replaceChildren();
  if(!event){target.append(el('p','No attempts were returned for this segment.'));$('shots-replay').disabled=true;setURL();renderList();return;}
  target.append(el('h2',`${team(event)} · ${outcomes[event.outcome]}`));
  target.append(el('p',`${statuses[event.automatic_status]} · Release ${time(event.clip_seconds)} · Recording ${time(event.recording_seconds)}`,`shots-status ${event.automatic_status}`));
  target.append(el('p',event.explanation));target.append(el('p',event.count_reason,'note'));
  target.append(el('p',event.basis==='contextual'?'Context inference · decisive contact may be hidden':event.basis==='direct'?'Direct visual evidence claimed by the model':'Evidence unresolved','shots-evidence'));
  if(event.review_note)target.append(el('p',`Assistant review: ${event.review_note}`,'note'));
  target.append(el('p',event.shooter_jersey?`Shooter number: ${event.shooter_jersey} (automatic reading; unverified)`:'Shooter number unresolved.','note'));
  if(event.evidence_frames?.length){const evidence=el('details'),summary=el('summary','Inspect model-cited frames'),buttons=el('div',undefined,'shots-controls');evidence.append(summary,buttons);for(const frame of [...new Set(event.evidence_frames)].sort((a,b)=>a-b)){const button=el('button',time(frame/15));button.type='button';button.addEventListener('click',()=>{video.pause();$('shots-loop').checked=false;seek(frame/15);});buttons.append(button);}target.append(evidence);}
  const links=el('div',undefined,'shots-links'),source=el('a','Original recording');source.href=`https://www.youtube.com/watch?v=${clip.source_id}&t=${Math.max(0,Math.floor(event.recording_seconds-3))}`;source.target='_blank';source.rel='noopener';links.append(source);
  if(clip.id==='nj-0900'){const overhead=el('a','Player and ball view');overhead.href=`overhead.html?ball=temporal&t=${event.clip_seconds}`;links.append(overhead);}target.append(links);
  $('shots-replay').disabled=!ready;setURL();renderList();
 }
 function renderList(){
  const all=events(),shown=all.filter(e=>filter.value==='all'||e.automatic_status===filter.value),list=$('shots-events');list.replaceChildren();
  $('shots-list-summary').textContent=`${shown.length} shown · ${all.length} detected attempts`;
  if(!shown.length)list.append(el('p',filter.value==='sog'?'No attempts met the automatic SOG evidence rule in this segment. Check “Needs review” for unresolved attempts.':'No attempts in this category.','note'));
  for(const event of shown){const button=el('button',undefined,`event-button${event.id===selected?.id?' selected':''}`);button.type='button';button.dataset.eventId=event.id;button.setAttribute('aria-pressed',String(event.id===selected?.id));
   button.append(el('span',time(event.clip_seconds),'time'));const name=el('span',undefined,'event-name');name.append(el('b',`${team(event)} · ${outcomes[event.outcome]}`));name.append(el('small',statuses[event.automatic_status]));button.append(name);button.addEventListener('click',()=>{detail(event);replay();});list.append(button);
  }
 }
 function totals(){
  const values=events(),target=$('shots-totals');target.replaceChildren();
  const entries=Object.entries(clip.team_labels).filter(([key])=>key!=='unknown');
  for(const [key,label]of entries){const counted=values.filter(e=>e.automatic_status==='sog'&&e.team_color===key),node=el('div');node.append(el('strong',`${counted.length}`),el('span',`${label} estimated SOG`),el('small',`${counted.filter(e=>e.outcome==='goal').length} goal${counted.filter(e=>e.outcome==='goal').length===1?'':'s'} included`));target.append(node);}
  const unresolved=el('div');unresolved.append(el('strong',String(values.filter(e=>e.automatic_status==='needs_review').length)),el('span','Need review'));target.append(unresolved);
  const unknown=values.filter(e=>e.automatic_status==='sog'&&!(e.team_color in clip.team_labels));if(unknown.length){const node=el('div');node.append(el('strong',String(unknown.length)),el('span','SOG · team unknown'));target.append(node);}
 }
 async function selectClip(id,eventId){
  const token=++generation;clip=dataset.clips.find(c=>c.id===id)||dataset.clips[0];picker.value=clip.id;ready=false;loopEnd=null;video.pause();
  $('shots-error').hidden=true;
  for(const name of ['shots-back','shots-forward','shots-replay'])$(name).disabled=true;
  $('shots-scope').textContent=`${clip.title} · Recording ${time(clip.start_s)}–${time(clip.start_s+clip.duration_s)}. Experimental segment counts; unresolved attempts are excluded and totals may undercount.`;
  totals();const initial=events().find(e=>e.id===eventId)||events().find(e=>e.automatic_status==='sog')||events()[0];if(initial&&initial.automatic_status!==filter.value&&filter.value!=='all')filter.value=initial.automatic_status==='not_sog'?'all':initial.automatic_status;
  detail(initial);renderList();const ok=await window.SportScansMedia.load(video,clip.video);if(token!==generation)return;if(!ok)return;
  const loaded=()=>{if(token!==generation)return;ready=true;video.playbackRate=Number($('shots-speed').value);$('shots-back').disabled=false;$('shots-forward').disabled=false;$('shots-replay').disabled=!selected;replay(false);};
  if(video.readyState>=1)loaded();else video.addEventListener('loadedmetadata',loaded,{once:true});
 }
 picker.addEventListener('change',()=>selectClip(picker.value));filter.addEventListener('change',renderList);
 $('shots-speed').addEventListener('change',()=>{video.playbackRate=Number($('shots-speed').value);});
 $('shots-back').addEventListener('click',()=>{video.pause();seek(video.currentTime-1/15);});$('shots-forward').addEventListener('click',()=>{video.pause();seek(video.currentTime+1/15);});$('shots-replay').addEventListener('click',()=>replay());
 video.addEventListener('timeupdate',()=>{if(!clip)return;$('shots-clock').textContent=`Clip ${time(video.currentTime)} / ${time(clip.duration_s)} · Recording ${time(clip.start_s+video.currentTime)}`;if($('shots-loop').checked&&!video.paused&&loopEnd!==null&&video.currentTime>=loopEnd)replay();});
 video.addEventListener('error',()=>error('The video could not be played. The original recording link is available below the attempt.'));
 (async()=>{try{
  const responses=await Promise.all(['events','report'].map(name=>fetch(`data/sog-20261027/${name}.json`)));for(const r of responses)if(!r.ok)throw Error('Results unavailable');[dataset,report]=await Promise.all(responses.map(r=>r.json()));
  picker.replaceChildren(...dataset.clips.map(c=>{const o=el('option',c.title);o.value=c.id;return o;}));picker.disabled=false;
  $('shots-method').textContent=report.method_summary;$('shots-review-note').textContent=report.review_summary;
  const q=new URLSearchParams(location.search);await selectClip(q.get('clip'),q.get('event'));
 }catch(e){error('Shot results could not load. Please reload the page.');}})();
})();
