"""New recording adapters; old methods/settings are kept fixed."""
import hashlib
import json
import time
from pathlib import Path

BASE=Path('/data/ball-hockey/reliability-20261018')
OUT=BASE/'new-games-20261025-v1'
PROTOCOL=Path('/work/new-games-protocol.json')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def specification(clip):return next(c for c in read(PROTOCOL)['clips']if c['clip']==clip)
def folder(clip):return OUT/'frames'/clip
def guard():
    p=read(PROTOCOL);r=read('/work/reference.json');review=read('/work/source-review.json')
    assert r['status']=='sealed_before_inference' and r['protocol_sha256']==sha(PROTOCOL)
    assert review['protocol_sha256']==sha(PROTOCOL)
    assert all(x['confirmed_ball_hockey']for x in review['sources'])
    assert {s['source_id']for s in review['sources']}=={s['id']for s in p['sources']}
    assert not ({s['id']for s in p['sources']}&set(p['previous_source_ids']))
    return p,r

def record_start(task,clip):
    import datetime
    _,reference=guard()
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    assert now>reference['sealed_at']
    dest=OUT/'execution'/task/(clip+'.json')
    if not dest.exists():
        dump(dest,{'task':task,'clip':clip,'started_at':now,'reference_sha256':sha('/work/reference.json'),'protocol_sha256':sha(PROTOCOL),'adapter_sha256':sha(__file__)})

def prepare(source,payload,volume):
    import urllib.request,shutil,subprocess,tarfile
    from PIL import Image,ImageDraw
    p=read(PROTOCOL);assert source in [r['id']for r in p['sources']]
    OUT.mkdir(parents=True,exist_ok=True)
    raw=Path('/data/raw')/(source+'.mp4')
    if not raw.exists():
        part=raw.with_suffix('.part')
        try:
            req=urllib.request.Request(payload['url'],headers=payload['headers'])
            with urllib.request.urlopen(req,timeout=45)as response,part.open('wb')as f:
                shutil.copyfileobj(response,f,1024*1024)
            assert part.stat().st_size==payload['expected_bytes']
            part.rename(raw)
        except Exception as e:
            part.unlink(missing_ok=True)
            return {'source_id':source,'download':'failed','error_type':type(e).__name__,'status_code':getattr(e,'code',None),'processing_performed':False}
    assert raw.stat().st_size==payload['expected_bytes']
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(raw)]))
    video=next(s for s in probe['streams']if s['codec_type']=='video')
    dest=OUT/'review'/source;dest.mkdir(parents=True,exist_ok=True)
    clips=[]
    for c in [c for c in p['clips']if c['source_id']==source]:
        frames=folder(c['clip']);frames.mkdir(parents=True,exist_ok=True)
        count=c['duration_s']*c['fps']
        if len(list(frames.glob('*.jpg')))!=count:
            subprocess.run(['ffmpeg','-v','error','-threads','4','-ss',str(c['start_s']),'-i',str(raw),'-t',str(c['duration_s']),'-an','-vf','setpts=PTS-STARTPTS,fps=15:start_time=0','-frames:v',str(count),'-q:v','2','-y',str(frames/'%06d.jpg')],check=True)
        paths=sorted(frames.glob('*.jpg'));assert len(paths)==count
        w,h=Image.open(paths[0]).size
        samples=[];sheet=Image.new('RGB',(960,6*294),'white');draw=ImageDraw.Draw(sheet)
        for i,f in enumerate(c['reference_frames']):
            path=paths[f];im=Image.open(path).convert('RGB');im.thumbnail((960,270));sheet.paste(im,(0,i*294+24))
            draw.text((8,i*294+5),f"{c['clip']} frame={f} source={c['start_s']+f/15:.2f}s  native={w}x{h}",fill='black')
            samples.append({'frame':f,'source_time_s':c['start_s']+f/15,'sha256':sha(path),'width':w,'height':h})
        sheet.save(dest/(c['clip']+'-sheet.jpg'),quality=94)
        clips.append(c|{'width':w,'height':h,'frame_count':count,'samples':samples})
        # Reviewable complete selected interval, not an edited highlight.
        media=OUT/'media'/(c['clip']+'.mp4');media.parent.mkdir(exist_ok=True)
        if not media.exists():
            subprocess.run(['ffmpeg','-v','error','-threads','4','-ss',str(c['start_s']),'-i',str(raw),'-t',str(c['duration_s']),'-an','-vf','setpts=PTS-STARTPTS,fps=15:start_time=0,scale=-2:720','-c:v','libx264','-preset','fast','-crf','25','-movflags','+faststart','-y',str(media)],check=True)
    result={'source_id':source,'protocol_sha256':sha(PROTOCOL),'raw_sha256':sha(raw),'raw_bytes':raw.stat().st_size,'video':{k:video.get(k)for k in ['width','height','r_frame_rate','duration']},'clips':clips,'inference_performed':False}
    dump(dest/'reservation.json',result)
    with tarfile.open(OUT/(source+'-review.tar.gz'),'w:gz')as t:
        for path in dest.glob('*'):t.add(path,arcname=path.name)
        for c in clips:
            for s in c['samples']:
                path=folder(c['clip'])/f"{s['frame']+1:06d}.jpg";t.add(path,arcname=c['clip']+f"/{s['frame']:06d}.jpg")
    volume.commit()
    return {k:v for k,v in result.items()if k!='clips'}|{'clips':[c['clip']for c in clips],'archive':str(OUT/(source+'-review.tar.gz'))}

def metadata(clip):return read(OUT/'detections'/clip/'metadata.json')

def detect(clip,payload,volume):
    import cv2,numpy as np,torch
    from PIL import Image
    from ultralytics import YOLO
    from ultralytics.utils import ROOT as UROOT,YAML
    from ultralytics.trackers.utils.gmc import GMC
    from rfdetr import RFDETR
    p,_=guard();spec=specification(clip);dest=OUT/'detections'/clip
    if (dest/'report.json').exists():
        cached=read(dest/'report.json');assert cached['protocol_sha256']==sha(PROTOCOL)
        for name,digest in cached['files'].items():assert sha(dest/name)==digest
        return cached
    torch.set_num_threads(4);cv2.setNumThreads(2)
    for detector in ['detector','rfdetr']:assert sha(p[detector]['weights'])==p[detector]['weights_sha256']
    yolo=YOLO(p['detector']['weights']);rfdetr=RFDETR.from_checkpoint(p['rfdetr']['weights'])
    config=YAML.load(UROOT/'cfg/trackers/botsort.yaml')
    config.update(track_high_thresh=.65,track_low_thresh=.1,new_track_thresh=.65,track_buffer=30,with_reid=False)
    gmc=GMC(method=config['gmc_method']);previous=None;scene=0
    paths=sorted(folder(clip).glob('*.jpg'));assert len(paths)==spec['duration_s']*15
    rows=[];rfrows=[];warps=[];resets=[];offsets=[0];start=time.monotonic()
    for f,path in enumerate(paths):
        frame=cv2.imread(str(path));assert frame is not None
        h,w=frame.shape[:2];gray=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(160,90))
        reset=previous is not None and float(np.mean(cv2.absdiff(gray,previous)))>55.
        previous=gray
        if reset:scene+=1;gmc=GMC(method=config['gmc_method'])
        a=yolo.predict(frame,imgsz=1280,conf=.025,iou=.65,classes=[0],max_det=100,device=0,verbose=False)[0].boxes.data.cpu().numpy().reshape(-1,6)
        try:warp=gmc.apply(frame,a[a[:,4]>=.65,:4])
        except Exception:warp=np.eye(2,3)
        pred=rfdetr.predict(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),threshold=.025)
        rfrows.append([list(map(float,b))+[float(s),int(c)]for b,s,c in zip(pred.xyxy,pred.confidence,pred.class_id)])
        rows.append({'frame':f,'source_time_s':spec['start_s']+f/15,'detections':a.tolist(),'scene':scene})
        warps.append(warp);resets.append(reset);offsets.append(offsets[-1]+len(a))
        if f%300==0:print(json.dumps({'task':'detect','clip':clip,'frame':f,'seconds':time.monotonic()-start}),flush=True)
    dump(dest/'yolo.json',[r['detections']for r in rows]);dump(dest/'rfdetr.json',rfrows)
    np.savez_compressed(dest/'motion.npz',warps=np.array(warps),resets=np.array(resets),offsets=np.array(offsets))
    meta={'summary':{'clip':clip,'source_id':spec['source_id'],'input_sha256':sha(dest/'yolo.json')},'config':config,'frames':rows,'shape':[h,w],'long':len(paths)>900}
    dump(dest/'metadata.json',meta)
    report={'clip':clip,'frames':len(paths),'seconds':time.monotonic()-start,'protocol_sha256':sha(PROTOCOL),'reference_sha256':sha('/work/reference.json'),'files':{n:sha(dest/n)for n in ['yolo.json','rfdetr.json','motion.npz','metadata.json']}}
    dump(dest/'report.json',report);return report

def track(clip,payload,volume):
    import cv2,numpy as np,supervision as sv
    from trackers import BoTSORTTracker
    p,_=guard();meta=metadata(clip);cv2.setNumThreads(2);reports=[]
    for detector in ['rfdetr','yolo']:
        method='roboflow-'+detector+'-botsort';dest=OUT/'tracks'/method/(clip+'.json')
        if dest.exists():
            x=read(dest);assert x['protocol_sha256']==sha(PROTOCOL);reports.append({'method':method,'clip':clip,'cached':True});continue
        tracker=BoTSORTTracker(frame_rate=15);previous=None;rows=[];start=time.monotonic()
        detections=read(OUT/'detections'/clip/(detector+'.json'))
        for f,raw in enumerate(detections):
            scene=meta['frames'][f]['scene']
            if previous is not None and previous!=scene:tracker=BoTSORTTracker(frame_rate=15)
            previous=scene;arr=np.asarray(raw,dtype=np.float32).reshape(-1,6)
            arr=arr[arr[:,4]>=.025]
            frame=cv2.imread(str(folder(clip)/f'{f+1:06d}.jpg'));assert frame is not None
            result=tracker.update(sv.Detections(xyxy=arr[:,:4].copy(),confidence=arr[:,4].copy(),class_id=np.zeros(len(arr),dtype=int)),frame=frame)
            objects=[]
            if result.tracker_id is not None:
                for box,confidence,tid in zip(result.xyxy,result.confidence,result.tracker_id):
                    if tid<0:continue
                    assert np.any(np.max(np.abs(arr[:,:4]-box),axis=1)<.01)
                    objects.append({'track_id':f'{scene}:{int(tid)}','bbox':list(map(float,box)),'confidence':float(confidence)})
            assert len({o['track_id']for o in objects})==len(objects)
            rows.append({'frame':f,'objects':objects})
        report={'clip':clip,'method':method,'protocol_sha256':sha(PROTOCOL),'seconds':time.monotonic()-start,'frames':rows,'labels_used_for_inference':False}
        dump(dest,report);reports.append({k:v for k,v in report.items()if k!='frames'})
        volume.commit()
    return reports

def sam(clip,payload,volume):
    import numpy as np,torch
    from PIL import Image
    from nbhl_identity_contrastive import load_model,preprocess
    from nbhl_sam2_tracking_v2 import track as sam_track
    from nbhl_sam2_lazy_frames import install
    p,_=guard();meta=metadata(clip);torch.set_num_threads(4)
    dest=OUT/'features'/clip;dest.mkdir(parents=True,exist_ok=True)
    features_path=dest/'features.npz'
    if not features_path.exists():
        model,provenance=load_model(BASE,'osnet_parts')
        assert provenance['weights_sha256']==p['osnet_sha256']
        transform=preprocess('osnet_parts');cached=dict(np.load(OUT/'detections'/clip/'motion.npz'))
        z=np.zeros((int(cached['offsets'][-1]),1536),np.float32);batch=[];indices=[];start=time.monotonic()
        def flush():
            if not batch:return
            with torch.inference_mode():values=model(torch.stack(batch).cuda()).float().cpu().numpy()
            assert np.isfinite(values).all() and np.allclose(np.linalg.norm(values,axis=1),1.,atol=1e-4)
            z[indices]=values;batch.clear();indices.clear()
        for f,row in enumerate(meta['frames']):
            im=Image.open(folder(clip)/f'{f+1:06d}.jpg').convert('RGB');w,h=im.size
            for j,d in enumerate(row['detections']):
                if d[4]<.1:continue
                batch.append(transform(im.crop((max(0.,d[0]),max(0.,d[1]),min(w,d[2]),min(h,d[3])))))
                indices.append(int(cached['offsets'][f])+j)
                if len(batch)==32:flush()
        flush();cached['features']=z;np.savez_compressed(features_path,**cached)
        meta['summary']['feature_sha256']=sha(features_path)
        dump(dest/'metadata.json',meta)
        dump(dest/'report.json',{'clip':clip,'seconds':time.monotonic()-start,'provenance':provenance,'features_sha256':sha(features_path)})
        del model;torch.cuda.empty_cache();volume.commit()
    meta=read(dest/'metadata.json');assert sha(features_path)==meta['summary']['feature_sha256']
    features=dict(np.load(features_path));target=OUT/'tracks/existing-sam2-local'/(clip+'.json')
    assert sha(p['sam2_weights'])==p['sam2_sha256']
    if not target.exists():
        restore,caches=install(capacity=4)
        try:result=sam_track(meta,features,folder(clip),p['sam2_weights'],'large',.1)
        finally:restore()
        result['protocol_sha256']=sha(PROTOCOL);result['frame_cache']=[c.statistics()for c in caches]
        dump(target,result)
    result=read(target);assert result['protocol_sha256']==sha(PROTOCOL)
    return {k:v for k,v in result.items()if k!='frames'}

def embed(clip,payload,volume):
    import numpy as np,torch,subprocess
    from huggingface_hub import snapshot_download
    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
    from nbhl_tracking_fourhour import tracklet_evidence,link_candidates
    from nbhl_tracking_global_links import global_links
    import nbhl_vlm_reconnect as reconnect
    p,_=guard();torch.set_num_threads(4)
    meta=read(OUT/'features'/clip/'metadata.json');features=dict(np.load(OUT/'features'/clip/'features.npz'))
    assert sha(OUT/'features'/clip/'features.npz')==meta['summary']['feature_sha256']
    raw_path=OUT/'tracks/existing-sam2-local'/(clip+'.json');raw=read(raw_path)
    target=OUT/'tracks/existing-sam2-osnet-qwen2b'/(clip+'.json')
    if target.exists():
        result=read(target);assert result['protocol_sha256']==sha(PROTOCOL)
        return {k:v for k,v in result.items()if k not in ['frames','links']}
    reconnect.OUT=OUT/'reconnect'
    reconnect.datasets=lambda phase:iter([(meta,features,raw,folder(clip),sha(raw_path))])
    reconnect.prepare(clip,sha(PROTOCOL))
    manifest=read(reconnect.OUT/clip/'evidence.json');evidence_row=manifest['clips'][0]
    evidence=tracklet_evidence(meta,features,raw);edges=link_candidates(meta,features,evidence)
    selected={(e['a'],e['b'])for e in evidence_row['selected_edges']}
    dest=OUT/'embeddings'/clip;dest.mkdir(parents=True,exist_ok=True)
    partial=dest/'features.npz';vectors=dict(np.load(partial))if partial.exists()else {}
    jobs=[(tid,row)for tid,row in evidence_row['tracks'].items()if tid not in vectors]
    start=time.monotonic()
    if jobs:
        assert subprocess.check_output(['git','-C','/opt/qwen-embedding','rev-parse','HEAD'],text=True).strip()==p['qwen']['code_revision']
        path=snapshot_download(p['qwen']['model'],revision=p['qwen']['revision'],local_files_only=True)
        encoder=Qwen3VLEmbedder(path,max_length=8192,min_pixels=65536,max_pixels=262144,torch_dtype=torch.bfloat16,attn_implementation='sdpa')
        instruction='Represent the identity of the central ball-hockey player for matching the same player across camera views. Focus on visible jersey digits, individual clothing and equipment details. Ignore background, pose and camera position.'
        for offset in range(0,len(jobs),4):
            batch=jobs[offset:offset+4];inputs=[]
            for tid,row in batch:
                for im in row['images']:assert sha(im['path'])==im['sha256']
                inputs.append(encoder.format_model_input(image=[im['path']for im in row['images']],instruction=instruction))
            processed=encoder._preprocess_inputs(inputs)
            assert 'pixel_values'in processed and len(processed['image_grid_thw'])==sum(len(r['images'])for _,r in batch)
            processed={k:v.to(encoder.model.device)for k,v in processed.items()}
            with torch.inference_mode():
                output=encoder.forward(processed)
                pooled=encoder._pooling_last(output['last_hidden_state'],output['attention_mask']).float()
                values=torch.nn.functional.normalize(pooled,p=2,dim=-1).cpu().numpy()
            assert np.isfinite(values).all()
            for (tid,_),value in zip(batch,values):vectors[tid]=value
            np.savez_compressed(partial,**vectors)
            if offset%64==0:volume.commit();print(json.dumps({'task':'embed','clip':clip,'completed':len(vectors),'total':len(evidence_row['tracks'])}),flush=True)
    scored=[]
    for edge in edges:
        if (edge['a'],edge['b'])not in selected:continue
        cosine=float(np.dot(vectors[edge['a']],vectors[edge['b']]))
        scored.append(edge|{'cosine_embedding':.5*(cosine+edge['cosine_mean'])})
    result=global_links(raw,evidence,scored,reconnect.SETTING|{'pooling':'embedding','cosine':.75})
    result.update(method='existing-sam2-osnet-qwen2b',protocol_sha256=sha(PROTOCOL),embedding_seconds=time.monotonic()-start,embedding_tracks=len(vectors),candidate_edges=len(edges),selected_edges=len(selected),qwen=p['qwen'])
    dump(target,result)
    return {k:v for k,v in result.items()if k not in ['frames','links']}

def export(clip,payload,volume):
    import tarfile,shutil
    from PIL import Image,ImageDraw
    from nbhl_tracking_fourhour import evaluate
    p,reference=guard();dest=OUT/'public';dest.mkdir(parents=True,exist_ok=True)
    data=dest/'data';media=dest/'media';data.mkdir(exist_ok=True);media.mkdir(exist_ok=True)
    reports=[];reservations=[];runtimes=[]
    for source in p['sources']:
        reservations.append(read(OUT/'review'/source['id']/'reservation.json'))
    for spec in p['clips']:
        name=spec['clip'];meta=metadata(name);h,w=meta['shape']
        meta['observations']=[o for o in reference['observations']if o['clip']==name]
        meta['pairs']=[o for o in reference['pairs']if o['clip']==name]
        for method in p['methods']:
            path=OUT/'tracks'/method/(name+'.json');raw=read(path)
            assert raw['protocol_sha256']==sha(PROTOCOL) and len(raw['frames'])==spec['duration_s']*15
            report=evaluate(meta,raw)|{'method':method,'source_id':spec['source_id'],'tracks_sha256':sha(path)}
            # Long separation is not proof that a player was occluded in between.
            obs={o['id']:o for o in meta['observations']}
            for category in ['up_to_2s','over_2s']:
                pairs=[q for q in report['pairs']if (abs(obs[q['a']]['frame']-obs[q['b']]['frame'])<=30)==(category=='up_to_2s')]
                report[category]={'same_total':sum(q['truth']=='same'for q in pairs),'correct':sum(q['truth']=='same'and q['predicted_same']is True for q in pairs),'different_total':sum(q['truth']=='different'for q in pairs),'wrong':sum(q['truth']=='different'and q['predicted_same']is True for q in pairs),'unknown_same':sum(q['truth']=='uncertain'and q['predicted_same']is True for q in pairs)}
            reports.append(report);dump(data/'reports'/(name+'-'+method+'.json'),report)
            selected={o['frame']for o in meta['observations']}
            dump(data/'sampled-predictions'/(name+'-'+method+'.json'),{k:v for k,v in raw.items()if k in ['clip','method']}|{'method':method,'frames':[r for r in raw['frames']if r['frame']in selected]})
            ids=sorted({o['track_id']for r in raw['frames']for o in r['objects']});mapping={tid:i+1 for i,tid in enumerate(ids)}
            frames=[]
            for row in raw['frames']:
                assert len({o['track_id']for o in row['objects']})==len(row['objects'])
                frames.append([[mapping[o['track_id']],*[round(min(1.,max(0.,v/(h if k%2 else w)))*10000)for k,v in enumerate(o['bbox'])]]for o in row['objects']])
            replay={'format':'compact_observed_boxes_v1','coordinate_scale':10000,'clip':name,'source_id':spec['source_id'],'method':method,'fps':15,'start_s':spec['start_s'],'frames':frames,'fragment_ids_only':True,'manually_corrected':False,'interpolated_positions':False,'track_id_map':mapping,'source_tracks_sha256':sha(path)}
            (media/(name+'-'+method+'.json')).write_text(json.dumps(replay,separators=(',',':'))+'\n')
            runtimes.append({'clip':name,'method':method,**{k:raw[k]for k in ['seconds','embedding_seconds','embedding_tracks','candidate_edges','selected_edges']if k in raw}})
        shutil.copy2(OUT/'media'/(name+'.mp4'),media/(name+'.mp4'))
        im=Image.open(folder(name)/'000001.jpg').convert('RGB');im.thumbnail((1280,720));im.save(media/(name+'.jpg'),quality=90)
        # Separate reference-only images allow review without model boxes.
        for frame in spec['reference_frames']:
            im=Image.open(folder(name)/f'{frame+1:06d}.jpg').convert('RGB');draw=ImageDraw.Draw(im)
            for o in [o for o in meta['observations']if o['frame']==frame]:
                draw.rectangle(o['bbox'],outline='#16bf48',width=3)
                label=o['id'].rsplit('/',1)[-1]+' '+(o.get('identity')or'?')
                x,y=o['bbox'][:2];draw.rectangle((x,max(0,y-19),x+max(55,len(label)*7),max(0,y-19)+19),fill='white');draw.text((x+2,max(0,y-18)),label,fill='black')
            im.thumbnail((1280,720));im.save(media/(name+f'-reference-{frame:04d}.jpg'),quality=94)
    totals=[]
    for method in p['methods']:
        rows=[r for r in reports if r['method']==method]
        totals.append({'method':method,'counts':{k:sum(r['counts'][k]for r in rows)for k in rows[0]['counts']},**{cat:{k:sum(r[cat][k]for r in rows)for k in rows[0][cat]}for cat in ['up_to_2s','over_2s']}})
    result={'protocol_sha256':sha(PROTOCOL),'reference_sha256':sha('/work/reference.json'),'source_review_sha256':sha('/work/source-review.json'),'sources':p['sources'],'clips':p['clips'],'frames':sum(c['duration_s']*15 for c in p['clips']),'methods':totals,'per_clip':[{k:v for k,v in r.items()if k!='pairs'}for r in reports],'reservations':reservations,'runtimes':runtimes,'execution':[read(f)for f in sorted((OUT/'execution').glob('*/*.json'))],'limits':p['evaluation']}
    dump(data/'results.json',result)
    for path,name in [(PROTOCOL,'protocol.json'),(Path('/work/reference.json'),'reference.json'),(Path('/work/source-review.json'),'source-review.json')]:shutil.copy2(path,data/name)
    manifest={str(f.relative_to(dest)):sha(f)for f in sorted(dest.rglob('*'))if f.is_file()}
    dump(dest/'manifest.json',manifest)
    archive=OUT/'public.tar.gz'
    with tarfile.open(archive,'w:gz')as t:
        for path in sorted(dest.rglob('*')):
            if path.is_file():t.add(path,arcname=str(path.relative_to(dest)))
    volume.commit()
    return {'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'totals':totals}
