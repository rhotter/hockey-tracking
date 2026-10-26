(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  let data, baseline, report, csvURL;
  const green = '#17633e', orange = '#a65500';
  const active = () => $('ball-mode').value === 'baseline' ? baseline : data;
  function ellipse(ctx, center, covariance, scale) {
    const a = covariance[0][0]*scale[0]**2, b = covariance[0][1]*scale[0]*scale[1], d = covariance[1][1]*scale[1]**2;
    const root = Math.hypot(a-d, 2*b);
    ctx.beginPath();ctx.ellipse(...center,2*Math.sqrt(Math.max(0,(a+d+root)/2)),2*Math.sqrt(Math.max(0,(a+d-root)/2)),.5*Math.atan2(2*b,a-d),0,Math.PI*2);
    ctx.globalAlpha=.13;ctx.fill();ctx.globalAlpha=1;ctx.stroke();
  }
  function seek(frame) { $('overhead-video').pause();$('overhead-video').currentTime=frame/data.fps+.00001; }
  function overview() {
    const current=active(), counts={detected:0,inferred:0,unknown:0};
    const csv=['frame,clip_seconds,recording_seconds,state,x_relative,y_relative,x_pixel,y_pixel,model_radius_pixels,reason,review_support,review_frame'];
    const c=$('ball-timeline'),ctx=c.getContext('2d');ctx.fillStyle='#ececec';ctx.fillRect(0,0,c.width,c.height);
    for(const r of current.frames) {
      counts[r.state]++;
      csv.push([r.frame,r.frame/current.fps,current.start_s+r.frame/current.fps,r.state,...(r.xy||['','']),r.pixel?r.pixel[0]*1920:'',r.pixel?r.pixel[1]*1080:'',r.model_radius_px??'',r.reason,r.review_support??'',r.review_frame??''].join(','));
      if(r.state!=='unknown'){ctx.fillStyle=r.state==='detected'?green:orange;ctx.fillRect(r.frame/current.frames.length*c.width,0,Math.max(1,c.width/current.frames.length),c.height);}
    }
    $('ball-summary').textContent=`${counts.detected} detected frames + ${counts.inferred} inferred / 4,500. ${counts.unknown} remain unknown. Coverage is not accuracy.`;
    if(csvURL)URL.revokeObjectURL(csvURL);
    csvURL=URL.createObjectURL(new Blob([csv.join('\n')+'\n'],{type:'text/csv'}));$('ball-csv').href=csvURL;
  }
  async function load() {
    async function get(url){const response=await fetch(url);if(!response.ok)throw Error(`Could not load ${url}`);return response.json();}
    [data,baseline,report]=await Promise.all([get('data/ball-rebuild-20261026/replay.json'),get('data/ball-context-20261026/replay.json'),get('data/ball-rebuild-20261026/report.json')]);
    overview();$('ball-mode').addEventListener('change',overview);
    const test=report.review.improved.test,old=report.review.baseline.test;
    const audit=report.output_audit?.counts;
    $('ball-quality').textContent=(audit?`Added-position audit: ${audit.visible_ball||0} visible ball locations, ${audit.wrong_location||0} wrong, ${audit.unclear||0} unclear. `:'')+`Reused nine-frame check: ${test.tp}/${test.tp+test.fn} located, versus ${old.tp}/${old.tp+old.fn} previously. Assistant review, not independently verified game-wide accuracy.`;
    $('ball-timeline').onclick=e=>{const r=e.currentTarget.getBoundingClientRect();seek(Math.max(0,Math.min(4499,Math.floor((e.clientX-r.left)/r.width*4500))));};
    for(const button of document.querySelectorAll('[data-ball-seek]'))button.onclick=()=>seek(Number(button.dataset.ballSeek));
    const table=document.createElement('table'),head=table.createTHead().insertRow();
    for(const name of ['Detector input','Found / 59','Wrong location']){const th=document.createElement('th');th.textContent=name;head.append(th);}
    const names={full1280:'Full frame · 1280',full1920:'Full frame · 1920',tiles640:'640px crops',tiles960:'Crops enlarged to 960','multi-view':'Multi-view ensemble','verified-multi-view':'Ensemble + crop verifier','heatmap-rgb':'RGB heatmap','heatmap-temporal3':'Three-frame heatmap'};
    const body=table.createTBody();
    for(const [name,result] of Object.entries(report.regression)) {
      const row=body.insertRow();for(const value of [names[name]||name,`${result.regression.tp}/59`,result.regression.fp])row.insertCell().textContent=value;
    }
    $('ball-comparisons').replaceChildren(table);
    for(const row of report.review_points){
      const button=document.createElement('button');button.textContent=`${Math.floor(row.frame/900)}:${String(Math.floor(row.frame/15)%60).padStart(2,'0')} · ${row.split==='development'?'training':'reused review'}`;button.onclick=()=>seek(row.frame);$('ball-review-points').append(button);
    }
  }
  function zoom(estimate,color) {
    const canvas=$('ball-zoom'),ctx=canvas.getContext('2d'),video=$('overhead-video');
    ctx.fillStyle='#f1f1f1';ctx.fillRect(0,0,canvas.width,canvas.height);
    if(!estimate.pixel||video.readyState<2){ctx.fillStyle='#555';ctx.font='18px system-ui';ctx.textAlign='center';ctx.fillText('No supported ball estimate',240,140);return;}
    const sw=240/1920*video.videoWidth,sh=135/1080*video.videoHeight;
    ctx.drawImage(video,estimate.pixel[0]*video.videoWidth-sw/2,estimate.pixel[1]*video.videoHeight-sh/2,sw,sh,0,0,480,270);
    ctx.strokeStyle=color;ctx.lineWidth=2;ctx.setLineDash(estimate.state==='inferred'?[6,4]:[]);
    ctx.beginPath();ctx.arc(240,135,14,0,2*Math.PI);ctx.stroke();ctx.setLineDash([]);
  }
  function draw(row,oc,mc,point) {
    if(!data)return;
    const mode=$('ball-mode').value, estimate=active().frames[row.frame];
    document.body.dataset.ballState=mode==='off'?'off':estimate.state;
    if(mode==='off'){zoom({},green);$('ball-status').textContent='Ball overlay hidden';return;}
    const inferred=estimate.state==='inferred',color=inferred?orange:green;zoom(estimate,color);
    if(estimate.state==='unknown'){$('ball-status').textContent='Ball unknown · insufficient visual evidence';return;}
    const pixel=estimate.pixel.map((v,i)=>v*(i?1080:1920));
    const views=[[oc,pixel,estimate.pixel_covariance,[1,1],7]];
    if(estimate.xy)views.push([mc,point(estimate.xy),estimate.covariance,[910,455],5]);
    for(const [ctx,p,cov,scale,radius] of views){ctx.save();ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2;ctx.setLineDash(inferred?[7,5]:[]);if(cov)ellipse(ctx,p,cov,scale);ctx.beginPath();ctx.arc(...p,radius,0,Math.PI*2);ctx.stroke();ctx.restore();}
    if(estimate.xy){mc.save();mc.strokeStyle=color;mc.lineWidth=2;mc.globalAlpha=.5;mc.beginPath();let started=false;
      for(let f=Math.max(0,row.frame-9);f<=row.frame;f++){const r=active().frames[f];if(!r.xy){started=false;continue;}const p=point(r.xy);if(started)mc.lineTo(...p);else mc.moveTo(...p);started=true;}mc.stroke();mc.restore();}
    $('ball-status').textContent=(inferred?`Ball inferred · nearby evidence ${estimate.nearest_evidence_s.toFixed(2)}s away`:'Ball detected · visual path')+` · assumed error radius ${Math.round(estimate.model_radius_px)}px`+(estimate.review_support&&estimate.review_support!=='floor'?` · model review: ${estimate.review_support==='unclear'?'height unclear':estimate.review_support}`:'')+(!estimate.xy?' · rink position unavailable':'');
  }
  window.SportScansBall={load,draw};
})();
