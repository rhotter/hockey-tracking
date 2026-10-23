'use strict';
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=s=>`${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,'0')}`;
async function json(url){const r=await fetch(url);if(!r.ok)throw Error(`${url}: ${r.status}`);return r.json();}
let game,selectedPlayer=null,requestId=0,selectedEvent=null;
function playerEvents(){return game.events.filter(e=>!selectedPlayer||(e.team_id===selectedPlayer.team_id&&(e.player_id===selectedPlayer.id||e.assists.some(a=>a.id===selectedPlayer.id))));}
function renderPlayers(){const term=$('player-search').value.toLowerCase().trim(),team=$('player-team').value;
 const rows=game.players.filter(p=>(!team||String(p.team_id)===team)&&(!term||`${p.name} ${p.number}`.toLowerCase().includes(term))).sort((a,b)=>b.points-a.points||b.goals-a.goals||a.name.localeCompare(b.name));
 $('nbhl-player-rows').innerHTML=rows.map(p=>`<tr class="${selectedPlayer?.id===p.id?'selected':''}"><td>${p.points?`<button class="nbhl-player-link" data-player="${p.id}" data-team="${p.team_id}">#${esc(p.number)} ${esc(p.name)}</button>`:`#${esc(p.number)} ${esc(p.name)}`}<small>${esc(p.team)}</small></td><td>${p.goals}</td><td>${p.assists}</td><td>${p.points}</td></tr>`).join('')||'<tr><td colspan="4">No matching players.</td></tr>';
 $('nbhl-player-rows').querySelectorAll('[data-player]').forEach(b=>b.addEventListener('click',()=>{selectedPlayer=game.players.find(p=>p.id===Number(b.dataset.player)&&p.team_id===Number(b.dataset.team));renderPlayers();renderEvents();choose(playerEvents()[0]);}));
}
function renderEvents(){const events=playerEvents();$('clear-player').hidden=!selectedPlayer;$('event-count').textContent=`${events.length} goal events${selectedPlayer?' involving '+selectedPlayer.name:''}`;
 $('nbhl-events').innerHTML=events.map(e=>`<button class="event-button ${e.id===selectedEvent?'selected':''}" data-nbhl="${e.id}"><span class="time">${e.remaining}<small>P${e.period} · REMAINING</small></span><span class="event-name"><b>#${esc(e.scorer.number)} ${esc(e.scorer.name)}</b><small>${esc(e.team)} · ${e.score_after.join('–')}${selectedPlayer&&e.player_id!==selectedPlayer.id?' · credited assist':''}</small></span><span>${e.media?'▶':'★'}</span></button>`).join('');
 $('nbhl-events').querySelectorAll('[data-nbhl]').forEach(b=>b.addEventListener('click',()=>{const e=game.events.find(e=>e.id===Number(b.dataset.nbhl));choose(e,true);}));
}
function choose(e,autoplay=false){if(!e)return;selectedEvent=e.id;const v=$('nbhl-video');v.parentElement.querySelectorAll('.media-error').forEach(n=>n.remove());v.pause();v.hidden=!e.media;
 if(e.media){v.poster=e.media.replace('.mp4','.jpg');SportScansMedia.load(v,e.media,{autoplay});}else{SportScansMedia.clear(v);v.removeAttribute('poster');}
 const assists=e.assists.map(p=>`#${p.number} ${p.name}`).join(', ');
 $('nbhl-detail').innerHTML=`<span class="badge official">OFFICIAL NBHL · GOAL ${e.id}</span><h3 style="margin-top:12px">#${esc(e.scorer.number)} ${esc(e.scorer.name)}</h3><p>${esc(e.team)} · score ${e.score_after.join('–')}<br>P${e.period} · ${e.remaining} remaining · ${e.elapsed} elapsed</p><p><b>Credited assists:</b> ${assists?esc(assists):'None credited'}<br><small>Assist times refer to this goal, not the exact passes.</small></p><p>${esc(e.evidence)}</p>${e.video_interval_s?`<p><span class="badge reviewed">VIDEO REVIEW</span> Source window ${fmt(e.video_interval_s[0])}–${fmt(e.video_interval_s[1])}. Names come from the official record.</p>`:'<p class="fine">No reviewed goal clip or source-video clock alignment for this game yet.</p>'}<a href="${game.source}${e.clip_start_s!=null?'&t='+e.clip_start_s:''}" target="_blank" rel="noopener">Open original recording ↗</a>`;
 renderEvents();
}
async function loadGame(){const req=++requestId;try{const d=await json(`data/stats/nbhl-${$('nbhl-select').value}.json`);if(req!==requestId)return;game=d;selectedPlayer=null;selectedEvent=null;
 $('app-error').hidden=true;$('nbhl-title').textContent=d.title;$('nbhl-date').textContent=d.date+' · HOME – GUEST';$('nbhl-official').href=d.official_source;$('nbhl-csv').href=`data/stats/nbhl-${d.id}-players.csv`;
 $('score-line').innerHTML=['home','guest'].map((side,i)=>`<div>${d.score[i]}<span>${esc(d.teams[side])}</span></div>`).join('<i>:</i>');
 $('player-team').innerHTML='<option value="">Both teams</option>'+[...new Map(d.players.map(p=>[p.team_id,p.team])).entries()].map(([id,name])=>`<option value="${id}">${esc(name)}</option>`).join('');$('player-search').value='';renderPlayers();choose(d.events[0]);
 }catch(e){if(req!==requestId)return;$('app-error').hidden=false;$('app-error').textContent='Could not load this game: '+e.message;}}
$('nbhl-select').addEventListener('change',loadGame);$('player-search').addEventListener('input',renderPlayers);$('player-team').addEventListener('change',renderPlayers);$('clear-player').addEventListener('click',()=>{selectedPlayer=null;renderPlayers();renderEvents();});
let tracking;
function loadTrackingGame(){const g=tracking.games[$('track-game').value];$('track-moment').replaceChildren(...g.clips.map((c,i)=>new Option(c.label,String(i))));$('track-moment').value=$('track-game').value==='game2'?'1':'0';$('track-recipe').replaceChildren(...Object.entries(g.views).map(([key,value])=>new Option(value.label,key)));showTracking();}
function showTracking(){const g=tracking.games[$('track-game').value],clip=g.clips[Number($('track-moment').value)],key=$('track-recipe').value,media=clip.media[key],v=$('tracking-video');v.parentElement.querySelectorAll('.media-error').forEach(n=>n.remove());v.poster=media.poster;if($('tracking').open)SportScansMedia.load(v,media.video);else{SportScansMedia.clear(v);v.dataset.mediaSource=media.video;}$('tracking-note').textContent=g.views[key].description+' '+(clip.note||'');}
$('tracking').addEventListener('toggle',()=>{if($('tracking').open&&tracking)showTracking();else $('tracking-video').pause();});
$('track-game').addEventListener('change',loadTrackingGame);$('track-moment').addEventListener('change',showTracking);$('track-recipe').addEventListener('change',showTracking);
for(const v of document.querySelectorAll('video'))v.addEventListener('error',()=>{if(!v.getAttribute('src'))return;const p=document.createElement('p');p.className='fine media-error';p.textContent='This clip could not load. Try another clip or use the original recording link.';if(!v.parentElement.querySelector('.media-error'))v.insertAdjacentElement('afterend',p);});
loadGame();json('data/tracking-current.json').then(d=>{tracking=d;loadTrackingGame();}).catch(e=>{$('tracking-note').textContent='Tracking results could not load: '+e.message;});
