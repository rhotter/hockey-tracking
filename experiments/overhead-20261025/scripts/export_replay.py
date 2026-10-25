"""Export observed-only overhead coordinates, plus explicit reviewed identity associations."""
from pathlib import Path
import json
import numpy as np
import cv2
from geometry import project,usable,floor_point,footpoint,segment_path,visible_polygon,BOUNDS

def cameras(registration,method='floor',count=4500):
    keys={r['frame']:r['methods'][method]['selected']for r in registration['keyframes']}
    indices=sorted(keys);quad=np.array([[0,550],[1920,550],[1920,1000],[0,1000]],np.float32)
    result=[];j=0
    for f in range(count):
        while j+1<len(indices)and indices[j+1]<=f:j+=1
        lower=indices[j];upper=indices[min(j+1,len(indices)-1)]if lower!=f else lower
        a,b=keys[lower],keys[upper];H=None
        if a and b and upper-lower<=registration['stride']and usable(a['H'])and usable(b['H']):
            A,B=np.asarray(a['H']),np.asarray(b['H'])
            if lower==upper:H=A
            else:
                aa,bb=project(A,quad),project(B,quad)
                # A cut or an extreme camera fit may not be interpolated.
                scaled=np.abs(aa-bb)/np.array([7.637,2.])
                if np.quantile(scaled,.9)<.25:
                    alpha=(f-lower)/(upper-lower)
                    H=cv2.getPerspectiveTransform(quad,((1-alpha)*aa+alpha*bb).astype(np.float32))
            if not usable(H):H=None
        result.append(H)
    return result

def export(variant,volume):
    import shutil,tarfile,hashlib
    from PIL import Image,ImageDraw
    from overhead import BASE,OUT,SOURCE,START,DURATION,FPS,read,dump,sha
    from evaluate_geometry import evaluate
    config=read('/work/selection.json');identity=read('/work/player-review.json')
    registration=read(OUT/'registration-final.json');tracks=read(OUT/'tracks-sam2.json')['frames'];det=read(OUT/'detections.json')['frames'];roles=read(OUT/'roles.json')['tracks']
    Hs=cameras(registration,config['camera_method']);ball=read(OUT/'ball.json')['frames']
    for gap in read('/work/transition-review.json')['intervals']:
        for f in range(gap['first_frame'],gap['last_frame']+1):Hs[f]=None
    mapping={}
    for part in identity['associations']:
        for f in range(part['first_frame'],part['last_frame']+1):
            assert f not in mapping,'Overlapping reviewed identity fragments'
            mapping[f]=part['track_id']
    b=registration['calibration']['relative_bounds'];bounds=(b['left'],b['right'],b['far'],b['near']);lo,hi,_,_=bounds
    rows=[];bytrack={};ball_candidates=[]
    for f,(row,H)in enumerate(zip(tracks,Hs)):
        objects=[]
        for o in row['objects']:
            tid=o['track_id'];chosen=mapping.get(f)==tid
            if not chosen and not roles.get(tid,{}).get('keep',False):continue
            info=det[f];idx=o['detection_index'];pose=info['poses'].get(str(idx));foot=footpoint(pose,o['bbox'])
            if foot and o.get('mask_bbox'):
                a,b,c,d=o['mask_bbox']
                if foot[1]>d+8 or not a-8<=foot[0]<=c+8:foot=None
            xy=floor_point(H,foot,bounds)
            color=info['colors'][idx];color='red'if color['red']>.18 else'white'if color['white']>.3 else'unknown'
            # Reject people whose observed feet are outside the rink. No invented fallback feet.
            if foot is not None and H is not None and xy is None and not chosen:continue
            obj={'track_id':tid,'player_id':identity['id']if chosen else'T'+tid,'label':'T'+tid,
                 'bbox':[round(v/s,5)for v,s in zip(o['bbox'],[1920,1080,1920,1080])],
                 'foot':[round(foot[0]/1920,6),round(foot[1]/1080,6)]if foot else None,
                 'xy':xy,'color':color,'segment':None}
            objects.append(obj);bytrack.setdefault(obj['player_id'],[]).append({'frame':f,**obj})
        candidate=None
        options=sorted([d for d in ball[f]['detections']if d[4]>=.6],key=lambda d:-d[4])
        if options and(len(options)==1 or options[0][4]-options[1][4]>=.15):
            d=options[0];p=[(d[0]+d[2])/2,(d[1]+d[3])/2];xy=floor_point(H,p,bounds)
            if xy is not None:candidate={'pixel':[p[0]/1920,p[1]/1080],'xy':xy,'confidence':d[4]}
        ball_candidates.append(candidate)
        viewport=None;grid=None
        if H is not None:
            viewport=visible_polygon(H,bounds=bounds)
            G=np.linalg.inv(H);grid=[]
            for a,b in [([x,-1],[x,1])for x in [-1,0,1]]+[([lo,-1],[hi,-1])]:
                q=project(G,[a,b])
                if np.isfinite(q).all()and abs(q).max()<100000:grid.append((q/np.array([1920,1080])).round(6).tolist())
        rows.append({'frame':f,'camera_valid':H is not None,'image_to_floor':H.tolist()if H is not None else None,
                     'viewport':viewport,'grid':grid,'objects':objects,'ball':None})
    for pid,observations in bytrack.items():
        segmented=segment_path(observations)
        for ob in segmented:
            dest=next(o for o in rows[ob['frame']]['objects']if o['player_id']==pid)
            dest['segment']=ob['segment']
            if dest['xy']:dest['xy']=[round(v,6)for v in dest['xy']]
    for f,c in enumerate(ball_candidates):
        if not c:continue
        support=0
        for j in range(max(0,f-4),min(len(rows),f+5)):
            other=ball_candidates[j]
            if j==f or other is None:continue
            delta=abs(np.array(c['xy'])-np.array(other['xy']))
            if delta[0]<=.015*abs(j-f)+.01 and delta[1]<=.035*abs(j-f)+.025:support+=1
        if support>=2:rows[f]['ball']=c
    out=OUT/'public';(out/'data/overhead-20261025').mkdir(parents=True,exist_ok=True);media=out/'media/overhead-20261025';media.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(OUT/'media/nj-0900-5min.mp4',media/'nj-0900-5min.mp4')
    players=[{'id':identity['id'],'label':identity['label'],'short_label':identity['short_label'],'note':identity['disclosure'],'evidence':[]}]
    for evidence in identity['evidence']:
        f=evidence['frame'];tid=evidence['track_id'];o=next(o for o in tracks[f]['objects']if o['track_id']==tid)
        im=Image.open(OUT/'frames'/f'{f+1:06d}.jpg');x,y,u,v=o['bbox'];w,h=u-x,v-y
        crop=im.crop((max(0,x-.2*w),max(0,y-.08*h),min(1920,u+.2*w),min(1080,v+.08*h)))
        crop.thumbnail((400,500));name=f'player-{f:04d}.jpg';crop.save(media/name,quality=95)
        players[0]['evidence'].append({'frame':f,'track_id':tid,'image':'media/overhead-20261025/'+name,'caption':evidence['caption']})
    for pid,observations in sorted(bytrack.items(),key=lambda v:-sum(o['xy']is not None for o in v[1])):
        if pid==identity['id']or sum(o['xy']is not None for o in observations)<45:continue
        players.append({'id':pid,'label':pid+' · automatic fragment','short_label':pid,
                        'note':'Automatic track fragment. This ID is not a verified player identity or jersey number.','evidence':[]})
    replay={'source_id':SOURCE,'start_s':START,'duration_s':DURATION,'fps':FPS,'frames':rows,'players':players,
            'default_player':identity['id'],'rink':{'center':-lo/(hi-lo),'left_blue':(-1-lo)/(hi-lo),'right_blue':(1-lo)/(hi-lo)},
            'coordinate_units':'relative_rink','relative_bounds':list(bounds),'metric_distances_allowed':False,'method':config['camera_method']}
    # Keep compact frame sequences small enough for a static, client-only replay.
    (out/'data/overhead-20261025/replay.json').write_text(json.dumps(replay,separators=(',',':'))+'\n')
    geometry=read(OUT/'geometry-final.json')['heldout']
    selected=[next((o for o in r['objects']if o['player_id']==identity['id']),None)for r in rows]
    report={'source_id':SOURCE,'start_s':START,'duration_s':DURATION,'fps':FPS,'frames':len(rows),
            'camera_method':config['camera_method'],'camera_valid_frames':sum(r['camera_valid']for r in rows),
            'selected_player':identity['label'],'selected_observed_frames':sum(o is not None for o in selected),
            'selected_mapped_frames':sum(o is not None and o['xy']is not None for o in selected),
            'selected_segments':len({o['segment']for o in selected if o and o['segment']is not None}),
            'ball_emitted_frames':sum(r['ball']is not None for r in rows),'geometry':geometry,
            'identity_review':identity,'selection':config,'inputs':{n:sha(OUT/n)for n in ['registration-final.json','tracks-sam2.json','detections.json','roles.json','ball.json']},
            'display_summary':[],'independent_human_ground_truth':False}
    m=geometry['methods'][config['camera_method']]
    report['display_summary']=[
        f"Processed all 4,500 frames at 15 fps. Camera mapping accepted on {report['camera_valid_frames']/45:.1f}% of this recording interval.",
        f"Floor-line check: {m['answered_points']}/{m['marked_points']} marked points answered; median {m['median_error_px']:.1f} px and 90th percentile {m['p90_error_px']:.1f} px error in the 1920×1080 source. These are image-to-line errors, not metre accuracy.",
        f"{identity['short_label']}: {report['selected_mapped_frames']/15:.1f} seconds of accepted floor positions in the five-minute recording. This is observed coverage, not time on floor.",
        identity['disclosure'],
        f"Ball estimates appear on {report['ball_emitted_frames']}/4,500 frames after confidence, floor and temporal-support gates. This segment's ball accuracy has not been established."
    ]
    dump(out/'data/overhead-20261025/report.json',report)
    for name in ['player-review.json','floor-line-review-v1.json','floor-line-review-v2.json','selection.json','calibration-v1.json','transition-review.json']:
        shutil.copyfile('/work/'+name,out/'data/overhead-20261025'/name)
    shutil.copyfile(OUT/'geometry-final.json',out/'data/overhead-20261025/geometry-final.json')
    from render_trajectory import render
    render(replay,report,media)
    manifest={str(p.relative_to(out)):sha(p)for p in sorted(out.rglob('*'))if p.is_file()and p.name!='manifest.json'}
    dump(out/'manifest.json',manifest)
    with tarfile.open(OUT/'public.tar.gz','w:gz')as tar:
        for p in out.rglob('*'):
            if p.is_file():tar.add(p,arcname=str(p.relative_to(out)))
    return {k:v for k,v in report.items()if k not in ['geometry','identity_review','inputs']}|{'archive':str(OUT/'public.tar.gz'),'archive_sha256':sha(OUT/'public.tar.gz'),'bytes':(OUT/'public.tar.gz').stat().st_size}
