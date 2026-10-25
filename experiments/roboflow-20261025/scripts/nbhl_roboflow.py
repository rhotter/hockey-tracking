"""NBHL adapters around the actual pinned Roboflow implementations."""
import hashlib
import json
import time
from pathlib import Path

BASE=Path('/data/ball-hockey/reliability-20261018')
OLD=BASE/'tracking-four-hour-v1'
OUT=BASE/'roboflow-20261025-v1'
PROTOCOL=Path('/work/roboflow-replacement-protocol-20261025.json')


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def metadata(phase,clip,labels=False):
    assert clip in read(PROTOCOL)[phase]
    if phase=='development':path=OLD/'features/osnet_parts_generic'/clip/'metadata.json'
    else:path=OLD/'final-test/features/osnet_parts_generic'/clip/'metadata.json'
    data=read(path)
    source=data['summary']['source_id']
    registry={r['id']:r for r in read('/work/sources.json')}
    assert registry[source]['sport']=='ball_hockey' and registry[source]['league']=='NBHL',source
    if phase=='development':
        cuts=set(read(OLD/'cuts'/(clip+'.json'))['reset_frames']);scene=0
        for f,row in enumerate(data['frames']):
            scene+=f in cuts;row['scene']=scene
    if not labels:data={k:v for k,v in data.items() if k not in ['observations','pairs']}
    return data


def folder(phase,meta):
    if phase=='development' and meta['long']:return BASE/'long-sam2-v1'/meta['summary']['source_id']/'frames'
    return Path('/data/sequences')/meta['summary']['clip']/'img1'


def iter_frames(paths,enabled=True):
    """Bounded ordered prefetch hides network-volume latency, without frame skips."""
    if not enabled:
        yield from (None for _ in paths);return
    import cv2
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as pool:
        pending=deque();source=iter(paths)
        for path in __import__('itertools').islice(source,16):pending.append(pool.submit(cv2.imread,str(path)))
        while pending:
            frame=pending.popleft().result();assert frame is not None
            path=next(source,None)
            if path is not None:pending.append(pool.submit(cv2.imread,str(path)))
            yield frame


def prefetch_check(phase,clip,volume):
    import cv2,numpy as np
    meta=metadata('regression','pittsburgh-stress-0900')
    paths=sorted(folder('regression',meta).glob('*.jpg'))[:32]
    for path,frame in zip(paths,iter_frames(paths)):assert np.array_equal(cv2.imread(str(path)),frame)
    report={'ordered_frames_equal':len(paths),'maximum_prefetch_frames':16,'threads':8,'frame_skips':0}
    dump(OUT/'prefetch-check.json',report);return report


def preflight(phase,clip,volume):
    import importlib.metadata as im,inspect
    import trackers,supervision as sv
    from sports.common.team import TeamClassifier
    from sports.common.view import ViewTransformer
    from sports.common.ball import BallTracker
    p=read(PROTOCOL);rows=[]
    for phase in ['development','regression']:
        for name in p[phase]:
            meta=metadata(phase,name);paths=sorted(folder(phase,meta).glob('*.jpg'))
            assert len(paths)==len(meta['frames']),(name,len(paths),len(meta['frames']))
            rows.append({'phase':phase,'clip':name,'source_id':meta['summary']['source_id'],
                         'frames':len(paths),'shape':meta['shape'],'first_frame_sha256':sha(paths[0])})
    assert sha(BASE/p['rfdetr_weights'])==p['rfdetr_weights_sha256']
    classes={'sports_bytetrack':sv.ByteTrack,'rf_bytetrack':trackers.ByteTrackTracker,
             'rf_ocsort':trackers.OCSORTTracker,'rf_cbiou':trackers.CBIoUTracker,'rf_botsort':trackers.BoTSORTTracker}
    report={'protocol_sha256':sha(PROTOCOL),'clips':rows,
            'versions':{x:im.version(x) for x in ['sports','trackers','supervision','transformers','numpy','torch','umap-learn','scikit-learn']},
            'constructors':{k:str(inspect.signature(v)) for k,v in classes.items()},
            'upstream_source_sha256':{c.__name__:sha(inspect.getfile(c)) for c in [TeamClassifier,ViewTransformer,BallTracker,*classes.values()]}}
    dump(OUT/'preflight.json',report);return report


def detector_audit(phase,clip,volume):
    from collections import Counter
    p=read(PROTOCOL);rows=[]
    for phase in ['development','regression']:
        for clip in p[phase]:
            detections=read(OUT/'detections'/phase/clip/'detections.json')
            counts={str(t):dict(Counter(int(d[5]) for frame in detections for d in frame if d[4]>=t)) for t in [.025,.1,.25,.35,.6,.7]}
            rows.append({'phase':phase,'clip':clip,'classes_at_confidence':counts,
                         'maximum_class0_confidence':max([d[4] for frame in detections for d in frame if d[5]==0] or [0])})
    training=read(BASE/p['rfdetr_weights'].rsplit('/',1)[0]/'completed.json')
    result={'protocol_sha256':sha(PROTOCOL),'training_provenance':training['provenance'],'rfdetr_version':training['rfdetr_version'],'rows':rows}
    dump(OUT/'detector-audit.json',result);return result


def detect(phase,clip,volume):
    import torch
    from PIL import Image
    from rfdetr import RFDETR
    p=read(PROTOCOL);dest=OUT/'detections'/phase/clip
    if (dest/'report.json').exists():
        record=read(dest/'report.json');assert record['protocol_sha256']==sha(PROTOCOL)
        assert record['detections_sha256']==sha(dest/'detections.json');return record
    torch.set_num_threads(4)
    meta=metadata(phase,clip);paths=sorted(folder(phase,meta).glob('*.jpg'))
    checkpoint=BASE/p['rfdetr_weights'];assert sha(checkpoint)==p['rfdetr_weights_sha256']
    model=RFDETR.from_checkpoint(str(checkpoint))
    started=time.monotonic();rows=[]
    for f,path in enumerate(paths):
        pred=model.predict(Image.open(path).convert('RGB'),threshold=.025)
        assert pred.class_id is not None
        rows.append([list(map(float,b))+[float(s),int(c)] for b,s,c in zip(pred.xyxy,pred.confidence,pred.class_id)])
        if f%300==0:print(json.dumps({'task':'detect','clip':clip,'frame':f,'total':len(paths)}),flush=True)
    dump(dest/'detections.json',rows)
    report={'clip':clip,'phase':phase,'frames':len(rows),'seconds':time.monotonic()-started,
            'weights_sha256':sha(checkpoint),'protocol_sha256':sha(PROTOCOL),
            'detections_sha256':sha(dest/'detections.json'),'class_ids':sorted({r[5] for frame in rows for r in frame})}
    dump(dest/'report.json',report);volume.commit();return report


def make_tracker(family,profile):
    import supervision as sv
    import trackers
    classes={'sports_bytetrack':sv.ByteTrack,'rf_bytetrack':trackers.ByteTrackTracker,
             'rf_ocsort':trackers.OCSORTTracker,'rf_cbiou':trackers.CBIoUTracker,'rf_botsort':trackers.BoTSORTTracker}
    options={'frame_rate':15}
    if family=='sports_bytetrack':options['minimum_consecutive_frames']=3
    if profile=='long_buffer':options['lost_track_buffer']=120
    return classes[family](**options)


def score_result(phase,clip,result):
    from nbhl_tracking_fourhour import evaluate
    from nbhl_tracking_reference_audit import corrected_report
    report=evaluate(metadata(phase,clip,labels=True),result)
    if phase=='development':
        corrections={r['pair_id']:r for r in read('/work/tracking-four-hour-reference-corrections-v1.json')['corrections']}
        needed={p['id']:corrections[p['id']] for p in report['pairs'] if p['id'] in corrections and p['truth']!=corrections[p['id']]['truth']}
        report=corrected_report(report,needed)
    return report


def track(phase,clip,volume):
    import cv2,numpy as np,supervision as sv
    cv2.setNumThreads(2)
    p=read(PROTOCOL);meta=metadata(phase,clip);paths=sorted(folder(phase,meta).glob('*.jpg'))
    selection=read(OUT/'selection.json') if phase=='regression' else None
    if selection:assert selection['protocol_sha256']==sha(PROTOCOL)
    reports=[]
    for detector in p['detectors']:
        configs=[(f,profile) for f in p['trackers'] for profile in p['profiles']]
        if selection:configs=sorted({('sports_bytetrack','default'),tuple(selection['winners'][detector]['setting'])})
        detections=read(OUT/'detections'/phase/clip/'detections.json') if detector=='rfdetr_nbhl' else [r['detections'] for r in meta['frames']]
        assert len(detections)==len(paths)
        active={}
        for family,profile in configs:
            key=detector+'-'+family+'-'+profile;dest=OUT/'tracking'/phase/key/clip
            if (dest/'metrics.json').exists():
                old=read(dest/'metrics.json');assert old['protocol_sha256']==sha(PROTOCOL)
                assert old['tracks_sha256']==sha(dest/'tracks.json')
                reports.append({k:v for k,v in old.items() if k!='pairs'});continue
            active[key]={'tracker':make_tracker(family,profile),'family':family,'profile':profile,'frames':[],'seconds':0.,'scene':None,'dest':dest}
        started=time.monotonic()
        decoded=iter_frames(paths,any(s['family']=='rf_botsort' for s in active.values()))
        for f,raw in enumerate(detections):
            frame=next(decoded)
            arr=np.asarray(raw,dtype=np.float32).reshape(-1,6)
            # Class-agnostic association on the one-class-trained checkpoint's
            # returned proposals. Raw RF-DETR class logits are retained for audit.
            arr=arr[arr[:,4]>=.025]
            for key,state in active.items():
                scene=meta['frames'][f].get('scene',0)
                if state['scene'] is not None and state['scene']!=scene:state['tracker']=make_tracker(state['family'],state['profile'])
                state['scene']=scene
                det=sv.Detections(xyxy=arr[:,:4].copy(),confidence=arr[:,4].copy(),class_id=np.zeros(len(arr),dtype=int))
                began=time.monotonic()
                if state['family']=='sports_bytetrack':out=state['tracker'].update_with_detections(det)
                elif state['family']=='rf_botsort':out=state['tracker'].update(det,frame=frame)
                else:out=state['tracker'].update(det)
                state['seconds']+=time.monotonic()-began
                objects=[]
                if out.tracker_id is not None:
                    for b,s,tid in zip(out.xyxy,out.confidence,out.tracker_id):
                        if int(tid)<0:continue
                        # Ensure library output has not invented a currently invisible box.
                        assert np.any(np.max(np.abs(arr[:,:4]-b),axis=1)<.01)
                        objects.append({'track_id':f'{scene}:{int(tid)}','bbox':list(map(float,b)),'confidence':float(s)})
                assert len({o['track_id'] for o in objects})==len(objects)
                state['frames'].append({'frame':f,'objects':objects})
            if f%300==0:print(json.dumps({'task':'track','clip':clip,'detector':detector,'frame':f,'variants':len(active)}),flush=True)
        for key,state in active.items():
            result={'clip':clip,'method':key,'frames':state['frames']}
            dest=state['dest'];dump(dest/'tracks.json',result)
            # No reference labels are accessed until all inference for this variant is done.
            report=score_result(phase,clip,result)|{'method':key,'detector':detector,
                'setting':[state['family'],state['profile']],'tracking_seconds':state['seconds'],
                'shared_decode_and_tracking_seconds':time.monotonic()-started,'protocol_sha256':sha(PROTOCOL),
                'tracks_sha256':sha(dest/'tracks.json')}
            dump(dest/'metrics.json',report);reports.append({k:v for k,v in report.items() if k!='pairs'})
        volume.commit()
    print(json.dumps({'clip':clip,'results':[{'method':r['method'],'counts':r['counts']} for r in reports]}),flush=True)
    return {'clip':clip,'reports':reports}


def select(phase,clip,volume):
    assert phase=='development';p=read(PROTOCOL);rows=[]
    for detector in p['detectors']:
        for family in p['trackers']:
            for profile in p['profiles']:
                key=detector+'-'+family+'-'+profile
                reports=[read(OUT/'tracking/development'/key/c/'metrics.json') for c in p['development']]
                assert all(r['protocol_sha256']==sha(PROTOCOL) for r in reports)
                counts={k:sum(r['counts'][k] for r in reports) for k in reports[0]['counts']}
                rows.append({'method':key,'detector':detector,'setting':[family,profile],'counts':counts,
                             'fragments':sum(r['fragments'] for r in reports),'tracking_seconds':sum(r['tracking_seconds'] for r in reports)})
    def objective(r):
        c=r['counts'];return (-(c['correct']-10*c['wrong']),-c['matched_observations'],c['unknown_same'],r['fragments'],r['setting'][1]!='default',r['method'])
    result={'protocol_sha256':sha(PROTOCOL),'selection_uses_regression':False,
            'frozen_at':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
            'winners':{d:sorted([r for r in rows if r['detector']==d],key=objective)[0] for d in p['detectors']},'development':rows}
    path=OUT/'selection.json'
    if path.exists():
        old=read(path);assert old['winners']==result['winners'] and old['protocol_sha256']==result['protocol_sha256'];return old
    dump(path,result);return result


def team_gate(distances,radius):
    """Abstain on ambiguous or far-from-training cluster assignments."""
    import numpy as np
    distances=np.asarray(distances);order=np.sort(distances,axis=1)
    margin=(order[:,1]-order[:,0])/np.maximum(order[:,1],1e-9)
    return (margin>=.2)&(order[:,0]<=radius),margin


def teams(phase,clip,volume):
    import cv2,numpy as np,torch
    from scipy.optimize import linear_sum_assignment
    from huggingface_hub import snapshot_download
    from nbhl_detection_metrics import iou
    import sports.common.team as upstream
    cv2.setNumThreads(2);torch.set_num_threads(4)
    variant=clip or 'full_box';assert variant in ['full_box','torso']
    suffix='' if variant=='full_box' else '-torso'
    p=read(PROTOCOL);started=time.monotonic()
    upstream.SIGLIP_MODEL_PATH=snapshot_download(p['siglip_model'],revision=p['siglip_revision'])
    groups={}
    for phase in ['development','regression']:
        for clip in p[phase]:
            if variant=='torso' and phase=='development':continue
            meta=metadata(phase,clip)
            groups.setdefault(meta['summary']['source_id'],[]).append((phase,clip,meta))
    reports=[]
    def crop(path,box):
        image=cv2.imread(str(path));h,w=image.shape[:2]
        if variant=='torso':
            x1,y1,x2,y2=box;wbox=x2-x1;hbox=y2-y1
            box=[x1+.15*wbox,y1+.12*hbox,x1+.85*wbox,y1+.55*hbox]
        x1,y1,x2,y2=np.round(box).astype(int)
        return image[max(0,y1):min(h,y2),max(0,x1):min(w,x2)].copy()
    for source,inputs in groups.items():
        dest=OUT/('teams'+suffix)/(source+'.json')
        if dest.exists():
            old=read(dest);assert old['protocol_sha256']==sha(PROTOCOL);reports.append(old);continue
        candidates=[]
        for phase,clip,meta in inputs:
            for f in range(7,len(meta['frames']),15):
                for d,raw in enumerate(meta['frames'][f]['detections']):
                    if raw[4]>=.65:candidates.append((phase,clip,f,d,raw[:4],folder(phase,meta)/f'{f+1:06d}.jpg'))
        indices=np.random.default_rng(p['seed']).choice(len(candidates),size=min(512,len(candidates)),replace=False)
        sampled=[candidates[i] for i in sorted(indices)]
        crops=[crop(r[-1],r[-2]) for r in sampled];assert all(c.size for c in crops)
        model=upstream.TeamClassifier(device='cuda',batch_size=32)
        model.reducer.set_params(random_state=p['seed'],transform_seed=p['seed'],n_jobs=1)
        model.cluster_model.set_params(random_state=p['seed'],n_init=10)
        model.fit(crops)
        distances=model.cluster_model.transform(model.reducer.embedding_)
        radius=float(np.quantile(np.min(distances,axis=1),.95))
        # References are accessed only after unsupervised fitting. All classification
        # crops below still come from automatic boxes, not reference boxes.
        automatic=[];matches=[];unmatched=[]
        for phase,clip,meta in inputs:
            labeled=metadata(phase,clip,labels=True);by_frame={}
            for obs in labeled['observations']:
                if 'team' in obs:by_frame.setdefault(obs['frame'],[]).append(obs)
            for f,observations in sorted(by_frame.items()):
                detections=[d for d in meta['frames'][f]['detections'] if d[4]>=.65]
                offset=len(automatic)
                for d in detections:
                    automatic.append({'clip':clip,'phase':phase,'frame':f,'bbox':d[:4],
                                      'crop':crop(folder(phase,meta)/f'{f+1:06d}.jpg',d[:4])})
                overlap=np.array([[iou(o['bbox'],d[:4]) for d in detections] for o in observations])
                used=set()
                if detections:
                    aa,bb=linear_sum_assignment(np.where(overlap>=.5,1-overlap,1e6))
                    for a,b in zip(aa,bb):
                        if overlap[a,b]>=.5:
                            matches.append({'id':observations[a]['id'],'truth':observations[a]['team'],
                                            'automatic_index':offset+int(b)});used.add(int(a))
                unmatched.extend(o['id'] for i,o in enumerate(observations) if i not in used)
        if not automatic:
            report={'source_id':source,'protocol_sha256':sha(PROTOCOL),'fit_crops':len(crops),
                    'counts':None,'status':'Fitted; this recording has no existing per-observation team reference.',
                    'fit_uses_reference_labels':False,'fit_uses_unlabeled_evaluation_video':True}
            dump(dest,report);reports.append(report);volume.commit();del model;torch.cuda.empty_cache();continue
        features=model.extract_features([r.pop('crop') for r in automatic])
        projections=model.reducer.transform(features)
        labels=model.cluster_model.predict(projections)
        distances=model.cluster_model.transform(projections)
        accepted,margins=team_gate(distances,radius)
        for i,row in enumerate(automatic):
            row.update(cluster=int(labels[i]),accepted=bool(accepted[i]),margin=float(margins[i]),distance=float(min(distances[i])))
        names=sorted({m['truth'] for m in matches});assert len(names)==2,names
        # A cluster has no semantic team name. This permutation is scoring only.
        mappings=[dict(enumerate(names)),dict(enumerate(reversed(names)))]
        mapping=max(mappings,key=lambda mp:sum(mp[int(labels[m['automatic_index']])]==m['truth'] for m in matches))
        scored=[]
        for m in matches:
            a=automatic[m['automatic_index']]
            scored.append(m|{'prediction':mapping[a['cluster']],'correct':mapping[a['cluster']]==m['truth'],'accepted':a['accepted']})
        totals={'reference_observations':len(matches)+len(unmatched),'matched':len(matches),
                'correct_all':sum(r['correct'] for r in scored),'accepted':sum(r['accepted'] for r in scored),
                'correct_accepted':sum(r['correct'] and r['accepted'] for r in scored),
                'unmatched':len(unmatched)}
        report={'source_id':source,'protocol_sha256':sha(PROTOCOL),'variant':variant,'fit_crops':len(crops),
                'fit_observations':[{'clip':r[1],'frame':r[2],'detection':r[3]} for r in sampled],
                'training_distance_95pct':radius,'scoring_label_permutation':mapping,'counts':totals,
                'automatic_predictions':automatic,'matches':scored,'unmatched_observation_ids':unmatched,
                'fit_uses_reference_labels':False,'fit_uses_unlabeled_evaluation_video':True,
                'independent_human_ground_truth':False}
        dump(dest,report);reports.append(report);volume.commit()
        print(json.dumps({'task':'teams','source':source,'counts':totals}),flush=True)
        del model;torch.cuda.empty_cache()
    result={'sources':reports,'variant':variant,'seconds':time.monotonic()-started,'protocol_sha256':sha(PROTOCOL)}
    if variant=='torso':result['followup_protocol_sha256']=sha('/work/roboflow-team-torso-followup-20261025.json')
    dump(OUT/('teams'+suffix+'.json'),result);return {k:v for k,v in result.items() if k!='sources'}|{'sources':[{'source_id':r['source_id'],'counts':r['counts']} for r in reports]}


def three_readings(parsed,images):
    """Tutorial-style three consecutive chronological readings, with abstention."""
    from evaluate_nbhl_jersey import normalize
    if not isinstance(parsed,dict) or parsed.get('identity_consistent')!='yes':return None
    rows=parsed.get('readings',[])
    if not isinstance(rows,list) or not all(isinstance(r,dict) and type(r.get('image')) is int for r in rows):return None
    if len(rows)!=len(images) or sorted(r['image'] for r in rows)!=list(range(1,len(images)+1)):return None
    numbers={r['image']-1:normalize(r.get('number')) for r in rows}
    order=sorted(range(len(images)),key=lambda i:images[i]['frame'])
    values=[numbers[i] for i in order]
    if 'invalid' in values:return None
    answers={values[i] for i in range(2,len(values)) if values[i] is not None and values[i]==values[i-1]==values[i-2]}
    return next(iter(answers)) if len(answers)==1 else None


def utilities(phase,clip,volume):
    import cv2,numpy as np,supervision as sv
    from sports.common.view import ViewTransformer
    from sports.common.ball import BallTracker
    from evaluate_nbhl_jersey import counts
    calibration=read('/work/rink-relative-nj-v1.json')
    source=np.asarray(calibration['source_points'],dtype=np.float32)
    target=np.asarray(calibration['target_points'],dtype=np.float32)
    transformer=ViewTransformer(source=source,target=target)
    # Only these four correspondences are in the anchor camera. End-board points
    # are in different camera frames and require the existing registration stage.
    points=source
    projected=transformer.transform_points(points)
    previous=cv2.perspectiveTransform(points.reshape(-1,1,2),np.asarray(calibration['image_to_relative_floor'])).reshape(-1,2)
    geometry={'calibration_sha256':sha('/work/rink-relative-nj-v1.json'),
              'point_count':len(points),'maximum_difference_relative_units':float(np.max(np.abs(projected-previous))),
              'source_points':source.tolist(),'target_points':target.tolist(),'output_points':projected.tolist(),
              'dimensions_verified':False,'metric_distances_allowed':False,
              'scope':'Numerical compatibility of the projection utility only. Manual correspondences, camera registration, visibility gating and dimensions remain external.'}
    assert geometry['maximum_difference_relative_units']<1e-5
    # Existing fixed .6 operating point, unchanged ball detector. Reset at each
    # independent ten-second window, update every frame, score reviewed-visible only.
    ball_reference=read('/work/ball-neighbor-reference-report.json');ball_rows=[];inputs=[]
    registry={r['id']:r for r in read('/work/sources.json')}
    for file in ball_reference['prediction_files']:
        cid=file['clip'];path=BASE/'ball/baselines/yolo11m-crops640-s17-f10-e80-c6ab3e44-full1280-dense'/(cid+'.json')
        assert sha(path)==file['sha256'];data=read(path)
        assert registry[data['clip']['source_id']]['league']=='NBHL'
        known={r['frame']:(split,r) for split,v in ball_reference['splits'].items() for r in v['frames'] if r['clip']==cid}
        tracker=BallTracker(buffer_size=10)
        inputs.append({'clip':cid,'sha256':sha(path),'frames':len(data['frames'])})
        for row in data['frames']:
            raw=[r for r in row['predictions'] if r['confidence']>=.6]
            det=sv.Detections(xyxy=np.asarray([r['bbox'] for r in raw],dtype=np.float32).reshape(-1,4),
                              confidence=np.asarray([r['confidence'] for r in raw]),class_id=np.zeros(len(raw),dtype=int))
            out=tracker.update(det)
            if row['frame'] not in known:continue
            split,ref=known[row['frame']];assert row['sha256']==ref['sha256']
            cx,cy=ref['ball']['center'];x1,y1,x2,y2=ref['ball']['bbox'];radius=max(4,min(12,((x2-x1)+(y2-y1))/4))
            def evaluate_box(box):
                if box is None:return {'answered':False,'correct':False,'error_pixels':None,'bbox':None}
                center=(np.asarray(box[:2])+np.asarray(box[2:]))/2
                error=float(np.linalg.norm(center-np.asarray([cx,cy])))
                return {'answered':True,'correct':error<=radius,'error_pixels':error,'bbox':list(map(float,box))}
            top=max(raw,key=lambda r:r['confidence'])['bbox'] if raw else None
            picked=out.xyxy[0] if len(out) else None
            ball_rows.append({'clip':cid,'frame':row['frame'],'split':split,'frame_sha256':row['sha256'],
                              'reference_center':[cx,cy],'radius':radius,'candidate_count':len(raw),
                              'top_confidence':evaluate_box(top),'sports_balltracker':evaluate_box(picked)})
    ball_counts={}
    for split in ['all',*ball_reference['splits']]:
        rows=[r for r in ball_rows if split=='all' or r['split']==split]
        ball_counts[split]={method:{'reviewed_visible':len(rows),'answered':sum(r[method]['answered'] for r in rows),
                                   'correct':sum(r[method]['correct'] for r in rows)} for method in ['top_confidence','sports_balltracker']}
    ball={'confidence_threshold':.6,'buffer_size':10,'counts':ball_counts,'rows':ball_rows,'inputs':inputs,
          'unknown_visibility_excluded':True,'independent_human_ground_truth':False,
          'scope':'Existing ball detector; conditional localization on 108 correlated assistant-reviewed visible frames. No shot, possession or goal accuracy claim.'}
    evidence_path=BASE/'jersey-temporal-v1/evidence.json';pred_path=BASE/'jersey-temporal-v1/predictions-235b-fp8.json'
    evidence=read(evidence_path);pred=read(pred_path);assert pred['evidence_sha256']==sha(evidence_path)
    cases={r['id']:r for r in evidence['cases']};predictions={r['id']:r for r in pred['results']};jersey_rows=[]
    for label in read('/work/jersey-number-review-v2.json')['observations']:
        if registry[label['source_id']]['league']!='NBHL':continue
        case=cases[label['id']];prediction=predictions[label['id']]
        jersey_rows.append(label|{'prediction':three_readings(prediction.get('parsed'),case['images'])})
    jersey={'counts':counts(jersey_rows),'rows':jersey_rows,'evidence_sha256':sha(evidence_path),'prediction_sha256':sha(pred_path),
            'new_model_inference':False,'independent_human_ground_truth':False,
            'scope':'Tutorial-style three consecutive readings applied to archived VLM outputs in chronological image order. Existing assistant labels are not blind; unknown crops are not negatives. This does not implement or validate Roboflow number detection or OCR.'}
    result={'protocol_sha256':sha(PROTOCOL),'geometry':geometry,'ball':ball,'jersey':jersey}
    dump(OUT/'utilities.json',result);return {k:v for k,v in result.items() if k not in ['ball','jersey']}|{'ball':ball_counts,'jersey':jersey['counts']}


def export(phase,clip,volume):
    """Compact evidence and browser overlays; no video or model redistribution."""
    import tarfile,math
    p=read(PROTOCOL);selection=read(OUT/'selection.json');dest=OUT/'public'
    assert selection['protocol_sha256']==sha(PROTOCOL)
    for name in ['preflight','selection','teams','teams-torso','utilities','detector-audit','prefetch-check']:
        dump(dest/'reports'/(name+'.json'),read(OUT/(name+'.json')))
    methods=sorted({d+'-'+f+'-'+v for d in p['detectors'] for f,v in [('sports_bytetrack','default'),selection['winners'][d]['setting']]})
    methods.append(selection['winners']['yolo_cached']['method']+'-appearance')
    summaries=[];files=[]
    for method in methods:
        reports=[]
        for clip in p['regression']:
            origin=OUT/'tracking/regression'/method/clip
            report=read(origin/'metrics.json');raw=read(origin/'tracks.json')
            assert report['tracks_sha256']==sha(origin/'tracks.json')
            meta=metadata('regression',clip,labels=True)
            ref_frames={o['frame'] for o in meta['observations']}
            dump(dest/'reports/pairs'/(clip+'-'+method+'.json'),report)
            dump(dest/'reports/sampled-predictions'/(clip+'-'+method+'.json'),
                 {k:v for k,v in raw.items() if k!='frames'}|{'frames':[r for r in raw['frames'] if r['frame'] in ref_frames]})
            height,width=meta['shape'];ids={};frames=[]
            for i,row in enumerate(raw['frames']):
                assert i==row['frame'];objects=[]
                for obj in row['objects']:
                    box=obj['bbox'];assert all(math.isfinite(v) for v in box) and box[0]<box[2] and box[1]<box[3]
                    objects.append([ids.setdefault(obj['track_id'],len(ids)+1),
                                    *[max(0,min(10000,round(v/s*10000))) for v,s in zip(box,[width,height,width,height])]])
                frames.append(objects)
            replay={'format':'compact_observed_boxes_v1','coordinate_scale':10000,'clip':clip,'method':'roboflow-'+method,
                    'fps':15,'start_s':900 if 'stress' in clip else int(clip.rsplit('-',1)[1]),'frames':frames,
                    'fragment_ids_only':True,'manually_corrected':False,'interpolated_positions':False,
                    'track_id_map':ids,'source_tracks_sha256':report['tracks_sha256'],
                    'observation_definition':'Current automatic detector boxes with Roboflow tracker IDs. Visibility and identity are not independently verified.'}
            path=dest/'replays'/(clip+'-roboflow-'+method+'.json');path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(replay,separators=(',',':'))+'\n')
            reports.append({k:v for k,v in report.items() if k!='pairs'})
        summaries.append({'method':method,'counts':{k:sum(r['counts'][k] for r in reports) for k in reports[0]['counts']},
                          'fragments':sum(r['fragments'] for r in reports),'tracking_seconds':sum(r['tracking_seconds'] for r in reports),
                          'clips':reports})
    report={'id':p['id'],'protocol_sha256':sha(PROTOCOL),'selection_sha256':sha(OUT/'selection.json'),
            'hybrid_protocol_sha256':sha('/work/roboflow-hybrid-protocol-20261025.json'),
            'existing_pipeline':read('/work/roboflow-existing-baseline-20261025.json'),
            'tracking':summaries,'independent_human_ground_truth':False,'total_regression_frames':6300,
            'detector_runs':[read(OUT/'detections'/phase/clip/'report.json') for phase in ['development','regression'] for clip in p[phase]]}
    dump(dest/'reports/results.json',report)
    from PIL import Image,ImageDraw,ImageFont
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16) if Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf').exists() else ImageFont.load_default(size=16)
    for clip in ['pittsburgh-0600','marylandwomen-0600']:
        meta=metadata('regression',clip,labels=True);source=meta['summary']['source_id']
        scale=[960/meta['shape'][1],540/meta['shape'][0]]
        team=read(OUT/'teams'/(source+'.json'));pic=Image.open(folder('regression',meta)/'000001.jpg').convert('RGB').resize((960,540))
        sheet=Image.new('RGB',(1920,596),'white');sheet.paste(pic,(0,56));sheet.paste(pic,(960,56));draw=ImageDraw.Draw(sheet)
        draw.text((12,8),'Frozen assistant reference: team + review identity (not independent human ground truth)',fill='black',font=font)
        draw.text((972,8),'Roboflow team clusters: A / B; ? = confidence gate abstained',fill='black',font=font)
        draw.text((12,30),clip+' | first frame, fixed choice | boxes and IDs were not used to fit clusters',fill='black',font=font)
        for o in meta['observations']:
            if o['frame']!=0:continue
            box=[v*scale[i%2]+(56 if i%2 else 0) for i,v in enumerate(o['bbox'])]
            draw.rectangle(box,outline='#ffee00',width=3);draw.text((box[0],max(56,box[1]-19)),o['team']+' '+o['identity'],fill='#ffee00',font=font,stroke_width=1,stroke_fill='black')
        for o in team['automatic_predictions']:
            if o['clip']!=clip or o['frame']!=0:continue
            box=[v*scale[i%2]+(56 if i%2 else 960) for i,v in enumerate(o['bbox'])]
            color=('#00e5ff' if o['cluster']==0 else '#ff66ff') if o['accepted'] else '#bbbbbb'
            draw.rectangle(box,outline=color,width=3);draw.text((box[0],max(56,box[1]-19)),('A' if o['cluster']==0 else 'B')+('' if o['accepted'] else ' ?'),fill=color,font=font,stroke_width=1,stroke_fill='black')
        path=dest/'review'/(clip+'-teams.jpg');path.parent.mkdir(parents=True,exist_ok=True);sheet.save(path,quality=88)
    for path in sorted(dest.rglob('*')):
        if path.is_file() and path.name!='manifest.json':files.append({'name':str(path.relative_to(dest)),'sha256':sha(path),'bytes':path.stat().st_size})
    dump(dest/'manifest.json',{'files':files,'protocol_sha256':sha(PROTOCOL)})
    archive=OUT/'roboflow-public.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for path in sorted(dest.rglob('*')):
            if path.is_file():tar.add(path,arcname=str(path.relative_to(dest)))
    volume.commit()
    return {'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(files)}


def hybrid(phase,clip,volume):
    import numpy as np
    from nbhl_tracking_fourhour import tracklet_evidence,link_candidates
    from roboflow_existing_linker import global_links
    assert phase=='regression'
    protocol=read('/work/roboflow-hybrid-protocol-20261025.json')
    selection=read(OUT/'selection.json');method=selection['winners']['yolo_cached']['method']
    dest=OUT/'tracking/regression'/(method+'-appearance')/clip
    if (dest/'metrics.json').exists():
        old=read(dest/'metrics.json');assert old['hybrid_protocol_sha256']==sha('/work/roboflow-hybrid-protocol-20261025.json')
        assert old['tracks_sha256']==sha(dest/'tracks.json');return {k:v for k,v in old.items() if k!='pairs'}
    meta=metadata(phase,clip);raw=read(OUT/'tracking/regression'/method/clip/'tracks.json')
    feature_path=OLD/'final-test/features/osnet_parts_generic'/clip/'features.npz'
    assert sha(feature_path)==meta['summary']['feature_sha256']
    started=time.monotonic()
    for row in raw['frames']:
        detections=np.asarray(meta['frames'][row['frame']]['detections'])
        for obj in row['objects']:
            differences=np.max(np.abs(detections[:,:4]-obj['bbox']),axis=1)
            index=int(np.argmin(differences));assert differences[index]<.01
            obj['detection_index']=index
    with np.load(feature_path) as loaded:features=dict(loaded)
    evidence=tracklet_evidence(meta,features,raw);candidates=link_candidates(meta,features,evidence)
    linked=global_links(raw,evidence,candidates,protocol['linking']);linked['method']=method+'-appearance'
    dump(dest/'tracks.json',linked)
    report=score_result(phase,clip,linked)|{'method':linked['method'],'detector':'yolo_cached','setting':['existing_osnet_global_links','global-005'],
        'base_method':method,'tracking_seconds':time.monotonic()-started,'links':len(linked['links']),
        'feature_sha256':sha(feature_path),'protocol_sha256':sha(PROTOCOL),'hybrid_protocol_sha256':sha('/work/roboflow-hybrid-protocol-20261025.json'),
        'tracks_sha256':sha(dest/'tracks.json')}
    dump(dest/'metrics.json',report);volume.commit();return {k:v for k,v in report.items() if k!='pairs'}
