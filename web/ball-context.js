(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  let data, baseline, report, csvURL;
  const colors={detected:'#17633e',tracked:'#17633e',inferred:'#a65500',context:'#156a9c'};
  const active = () => $('ball-mode').value === 'baseline' ? baseline : data;
  const csvCell=value=>'"'+String(value??'').replaceAll('"','""')+'"';
  function ellipse(ctx, center, covariance, scale) {
    const a = covariance[0][0]*scale[0]**2, b = covariance[0][1]*scale[0]*scale[1], d = covariance[1][1]*scale[1]**2;
    const root = Math.hypot(a-d, 2*b);
    ctx.beginPath();ctx.ellipse(...center,2*Math.sqrt(Math.max(0,(a+d+root)/2)),2*Math.sqrt(Math.max(0,(a+d-root)/2)),.5*Math.atan2(2*b,a-d),0,Math.PI*2);
    ctx.globalAlpha=.13;ctx.fill();ctx.globalAlpha=1;ctx.stroke();
  }
  function seek(frame) { $('overhead-video').pause();$('overhead-video').currentTime=frame/data.fps+.00001; }
  function overview() {
    const current=active(), counts={detected:0,tracked:0,inferred:0,context:0,unknown:0};
    const csv=['frame,clip_seconds,recording_seconds,state,x_relative,y_relative,x_pixel,y_pixel,model_radius_pixels,reason,support,carrier,play_state,alternatives_json,evidence'];
    const c=$('ball-timeline'),ctx=c.getContext('2d');ctx.fillStyle='#ececec';ctx.fillRect(0,0,c.width,c.height);
    for(const r of current.frames) {
      counts[r.state]++;
      csv.push([r.frame,r.frame/current.fps,current.start_s+r.frame/current.fps,r.state,...(r.xy||['','']),r.pixel?r.pixel[0]*1920:'',r.pixel?r.pixel[1]*1080:'',r.model_radius_px??'',r.reason,r.review_support,r.hypotheses?.[0]?.carrier_track_id,r.play_state,r.hypotheses?JSON.stringify(r.hypotheses):'',r.evidence].map(csvCell).join(','));
      if(r.state!=='unknown'){ctx.fillStyle=colors[r.state];ctx.fillRect(r.frame/current.frames.length*c.width,0,Math.max(1,c.width/current.frames.length),c.height);}
    }
    const total=current.frames.length,known=total-counts.unknown;
    $('ball-summary').textContent=`${(100*known/total).toFixed(1)}% estimated coverage · ${counts.detected} visual + ${counts.tracked} video-tracked + ${counts.context+counts.inferred} inferred frames. ${counts.unknown} unknown. Coverage is not accuracy.`;
    if(csvURL)URL.revokeObjectURL(csvURL);
    csvURL=URL.createObjectURL(new Blob([csv.join('\n')+'\n'],{type:'text/csv'}));$('ball-csv').href=csvURL;
  }
  async function load() {
    async function get(url){const response=await fetch(url);if(!response.ok)throw Error(`Could not load ${url}`);return response.json();}
    [data,baseline,report]=await Promise.all([get('data/ball-sequence-20261026/replay.json'),get('data/ball-rebuild-20261026/replay.json'),get('data/ball-sequence-20261026/report.json')]);
    overview();$('ball-mode').addEventListener('change',overview);
    const audit=report.output_audit;
    $('ball-quality').textContent=audit?`New-estimate review: ${audit.counts.visible_inside||0} visibly inside the region, ${audit.counts.visible_outside||0} visibly outside, ${audit.counts.context_consistent_unverified||0} plausible but unverified, ${audit.counts.unclear||0} unclear. Assistant review on this development game; no independent human accuracy score.`:'Contextual regions are model estimates, not independently verified ball positions.';
    $('ball-timeline').onclick=e=>{const r=e.currentTarget.getBoundingClientRect();seek(Math.max(0,Math.min(data.frames.length-1,Math.floor((e.clientX-r.left)/r.width*data.frames.length))));};
    for(const button of document.querySelectorAll('[data-ball-seek]'))button.onclick=()=>seek(Number(button.dataset.ballSeek));
    const table=document.createElement('table'),head=table.createTHead().insertRow();
    for(const name of ['Video tracker','Within 25px / answered','Median error']){const th=document.createElement('th');th.textContent=name;head.append(th);}
    const body=table.createTBody();
    for(const result of report.point_comparisons) {
      const row=body.insertRow();for(const value of [result.name,`${result.within_25px} / ${result.answered}`,`${result.median_error_px.toFixed(1)}px`])row.insertCell().textContent=value;
    }
    $('ball-comparisons').replaceChildren(table);
    for(const row of report.sequences){
      const button=document.createElement('button');const t=Math.floor(row.core[0]/data.fps);button.textContent=`${Math.floor(t/60)}:${String(t%60).padStart(2,'0')}`;button.title=row.summary;button.onclick=()=>seek(row.core[0]);$('ball-review-points').append(button);
    }
  }
  function zoom(estimate,color) {
    const canvas=$('ball-zoom'),ctx=canvas.getContext('2d'),video=$('overhead-video');
    ctx.fillStyle='#f1f1f1';ctx.fillRect(0,0,canvas.width,canvas.height);
    if(!estimate.pixel||video.readyState<2){ctx.fillStyle='#555';ctx.font='18px system-ui';ctx.textAlign='center';ctx.fillText('No supported ball estimate',240,140);return;}
    const contextual=estimate.state==='context';const nativeWidth=contextual?Math.max(320,estimate.model_radius_px*4.2):240;
    const sw=nativeWidth/1920*video.videoWidth,sh=sw*270/480;
    ctx.drawImage(video,estimate.pixel[0]*video.videoWidth-sw/2,estimate.pixel[1]*video.videoHeight-sh/2,sw,sh,0,0,480,270);
    ctx.strokeStyle=color;ctx.lineWidth=2;ctx.setLineDash(contextual||estimate.state==='inferred'?[6,4]:[]);
    ctx.beginPath();ctx.arc(240,135,contextual?estimate.model_radius_px/nativeWidth*480:14,0,2*Math.PI);ctx.stroke();ctx.setLineDash([]);
  }
  function draw(row,oc,mc,point) {
    if(!data)return;
    const mode=$('ball-mode').value, estimate=active().frames[row.frame];
    document.body.dataset.ballState=mode==='off'?'off':estimate.state;
    $('ball-evidence').textContent='';
    if(mode==='off'){zoom({},colors.detected);$('ball-status').textContent='Ball overlay hidden';return;}
    const contextual=estimate.state==='context',inferred=contextual||estimate.state==='inferred',color=colors[estimate.state];zoom(estimate,color);
    if(estimate.state==='unknown'){$('ball-status').textContent='Ball unresolved or outside the view';return;}
    const hypotheses=contextual?estimate.hypotheses:[estimate];
    for(const [index,h]of hypotheses.entries()){
      const pixel=h.pixel.map((v,i)=>v*(i?1080:1920));const views=[[oc,pixel,h.pixel_covariance,[1,1],7]];
      if(h.xy)views.push([mc,point(h.xy),h.covariance,[910,455],5]);
      for(const [ctx,p,cov,scale,radius] of views){ctx.save();if(ctx===mc){ctx.beginPath();ctx.roundRect(45,40,910,455,35);ctx.clip();}ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=index?1:2;ctx.setLineDash(inferred?[7,5]:[]);if(cov)ellipse(ctx,p,cov,scale);ctx.beginPath();if(contextual){ctx.moveTo(p[0]-5,p[1]);ctx.lineTo(p[0]+5,p[1]);ctx.moveTo(p[0],p[1]-5);ctx.lineTo(p[0],p[1]+5);}else ctx.arc(...p,radius,0,Math.PI*2);ctx.stroke();ctx.restore();}
    }
    if(estimate.xy){mc.save();mc.beginPath();mc.roundRect(45,40,910,455,35);mc.clip();mc.strokeStyle=color;mc.lineWidth=2;mc.globalAlpha=.5;mc.setLineDash(inferred?[5,4]:[]);mc.beginPath();let started=false;
      for(let f=Math.max(0,row.frame-9);f<=row.frame;f++){const r=active().frames[f];if(!r.xy){started=false;continue;}const p=point(r.xy);if(started)mc.lineTo(...p);else mc.moveTo(...p);started=true;}mc.stroke();mc.restore();}
    const names={detected:'Ball detected · visual evidence',tracked:'Ball tracked · video model + appearance check',inferred:'Ball inferred · short gap',context:'Context estimate'};
    const carrier=estimate.hypotheses?.[0]?.carrier_track_id;
    $('ball-status').textContent=names[estimate.state]+(contextual?` · ${(estimate.play_state||'unknown').replaceAll('_',' ')}${carrier?' · '+carrier:''}${hypotheses.length>1?' · '+hypotheses.length+' alternatives':''}`:'')+` · assumed error radius ${Math.round(estimate.model_radius_px)}px`+(estimate.review_support&&estimate.review_support!=='floor'?` · ${estimate.review_support==='unclear'?'height unclear':estimate.review_support}`:'')+(!estimate.xy?' · rink position unavailable':'');
    if(mode==='context'){
      const sequence=report.sequences.find(s=>s.core[0]<=row.frame&&row.frame<s.core[1]);
      $('ball-evidence').textContent=estimate.evidence||(sequence?sequence.summary:'');
    }
  }
  window.SportScansBall={load,draw};
})();
