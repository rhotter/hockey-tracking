(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  let data, report;
  const green = '#17633e', orange = '#a65500';
  function ellipse(ctx, center, covariance, scale) {
    const a = covariance[0][0]*scale[0]**2;
    const b = covariance[0][1]*scale[0]*scale[1];
    const d = covariance[1][1]*scale[1]**2;
    const root = Math.hypot(a-d, 2*b);
    const major = 2*Math.sqrt(Math.max(0, (a+d+root)/2));
    const minor = 2*Math.sqrt(Math.max(0, (a+d-root)/2));
    ctx.beginPath();
    ctx.ellipse(...center, major, minor, .5*Math.atan2(2*b,a-d), 0, Math.PI*2);
    ctx.globalAlpha = .13; ctx.fill(); ctx.globalAlpha = 1; ctx.stroke();
  }
  async function load() {
    [data,report] = await Promise.all(['replay','report'].map(async name => {
      const response = await fetch(`data/ball-context-20261026/${name}.json`);
      if (!response.ok) throw Error(`Could not load ball ${name}`);
      return response.json();
    }));
    const states = report.variants.context.states;
    $('ball-summary').textContent = `${states.detected} detected frames + ${states.inferred} inferred frames / 4,500. Most of this five-minute segment remains unknown.`;
    const csv = ['frame,clip_seconds,recording_seconds,state,x_relative,y_relative,x_pixel,y_pixel,model_radius_pixels,reason'];
    for (const r of data.frames) csv.push([r.frame,r.frame/data.fps,data.start_s+r.frame/data.fps,r.state,...(r.xy||['','']),r.pixel?r.pixel[0]*1920:'',r.pixel?r.pixel[1]*1080:'',r.model_radius_px??'',r.reason].join(','));
    $('ball-csv').href = URL.createObjectURL(new Blob([csv.join('\n')+'\n'], {type:'text/csv'}));
    const c=$('ball-timeline'), ctx=c.getContext('2d');
    ctx.fillStyle='#ececec';ctx.fillRect(0,0,c.width,c.height);
    for (const r of data.frames) if(r.state!=='unknown') {
      ctx.fillStyle=r.state==='detected'?green:orange;
      ctx.fillRect(r.frame/data.frames.length*c.width,0,Math.max(1,c.width/data.frames.length),c.height);
    }
    c.onclick=e=>{const box=c.getBoundingClientRect();$('overhead-video').currentTime=Math.max(0,Math.min(299.933,(e.clientX-box.left)/box.width*300));};
    for(const button of document.querySelectorAll('[data-ball-seek]')) button.onclick=()=>{
      $('ball-mode').value='context';$('overhead-video').pause();
      $('overhead-video').currentTime=Number(button.dataset.ballSeek)/data.fps+.00001;
      $('ball-mode').dispatchEvent(new Event('change'));
    };
  }
  function draw(row, oc, mc, point) {
    if(!data)return;
    const mode=$('ball-mode').value;
    const estimate=mode==='context'?data.frames[row.frame]:row.ball?{...row.ball,state:'detected'}:{state:'unknown'};
    document.body.dataset.ballState=mode==='off'?'off':estimate.state;
    if(mode==='off'){$('ball-status').textContent='Ball overlay hidden';return;}
    if(estimate.state==='unknown'){
      $('ball-status').textContent=row.camera_valid?'Ball unknown · insufficient evidence':'Ball unknown · camera mapping unavailable';return;
    }
    const inferred=estimate.state==='inferred', color=inferred?orange:green;
    for(const ctx of [oc,mc]){ctx.save();ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2;ctx.setLineDash(inferred?[7,5]:[]);}
    const pixel=estimate.pixel.map((v,i)=>v*(i?1080:1920));
    const p=point(estimate.xy);
    if(estimate.covariance){ellipse(oc,pixel,estimate.pixel_covariance,[1,1]);ellipse(mc,p,estimate.covariance,[910,455]);}
    for(const [ctx,q,radius] of [[oc,pixel,5],[mc,p,4]]){ctx.beginPath();ctx.arc(...q,radius,0,Math.PI*2);ctx.stroke();ctx.restore();}
    const reason={short_gap_bridge:'between detections',occlusion_bridge:'possible player occlusion',motion_prediction:'motion prediction',occlusion_prediction:'prediction behind a player'}[estimate.reason];
    const radius=estimate.model_radius_px;
    $('ball-status').textContent=inferred
      ? `Ball inferred · ${reason} · nearest evidence ${estimate.nearest_evidence_s.toFixed(2)}s away · model radius ${Math.round(radius)}px`
      : `Ball detected${radius?` · model radius ${Math.round(radius)}px`:''}`;
  }
  window.SportScansBall={load,draw};
})();
