"""Export verified SAM3 experiment evidence and compact browser replays."""
from pathlib import Path
import json
import modal
from modal_campaign import ROOT, volume
from modal_nbhl_sam3_analysis import image, native_path, adapted_control_variant, BASE, OUT
image = image.add_local_file(ROOT/'scripts/modal_nbhl_sam3_analysis.py','/work/modal_nbhl_sam3_analysis.py')
image = image.add_local_file(ROOT/'notes/ball-hockey/sam3-20261024/memory-score-cache-cpu-verification.json','/work/memory-score-cache-cpu-verification.json')
for source in ['1f0Y5BIr5Ps','5Qw_g9E7EqE']:
    image = image.add_local_file(ROOT/'data/ball-hockey/annotations/heldout'/(source+'.json'),'/work/finetune-annotations/'+source+'.json')

app = modal.App('sport-scans-sam3-publish-20261024')
FT = BASE/'sam3-finetune-20261024-v1'


def compact(raw, shape, clip, method, source_hash, detector_supported=False):
    import math
    height,width = shape; ids = {}; frames = []
    for i,row in enumerate(raw['frames']):
        assert row['frame']==i
        objects = []
        for obj in row['objects']:
            box = obj['bbox']; assert all(math.isfinite(v) for v in box)
            assert box[0]<box[2] and box[1]<box[3]
            objects.append([ids.setdefault(obj['track_id'],len(ids)+1),
                            *[max(0,min(10000,round(v/scale*10000)))
                              for v,scale in zip(box,[width,height,width,height])]])
        assert len({o[0] for o in objects})==len(objects)
        frames.append(objects)
    return {'format':'compact_observed_boxes_v1','coordinate_scale':10000,'clip':clip,
            'method':method,'fps':15,'start_s':raw['frames'][0]['source_time_s'],
            'frames':frames,'fragment_ids_only':True,'manually_corrected':False,
            'source_tracks_sha256':source_hash,'track_id_map':ids,
            'observation_definition':('Current detector boxes supported by tracking, not verified player identities.'
                                      if detector_supported else
                                      'Model-emitted mask boxes, not verified visibility. Masks may drift or persist behind occlusion.'),
            'interpolated_positions':False}


@app.function(image=image,cpu=4,memory=8192,timeout=600,volumes={'/data':volume})
def reports(partial=False):
    from nbhl_vlm_reconnect import sha
    volume.reload(); files = []
    def add(key,path):
        if partial and not path.exists():return
        value=json.loads(path.read_text())
        files.append({'name':key,'value':value,'sha256':sha(path)})
    for variant in ['heads-lr1e4','decoder-lr1e5','decoder-lr5e5','transformer-lr1e5','transformer-lr3e5','vision4-lr1e6']:
        add('finetune/'+variant+'.json',FT/variant/'completed.json')
    for name in ['selection','regression','features']:add('finetune/'+name+'.json',FT/(name+'.json'))
    selected=json.loads((FT/'selection.json').read_text())['best']['variant']
    add('finetune/'+selected+'-validation-predictions.json',FT/selected/'best-validation-predictions.json')
    for variant in ['original',selected]:
        add('finetune/'+variant+'-predictions.json',FT/variant/'regression-predictions.json')
    for name in ['development','selection','regression','runtime-verification','memory-score-cache-verification','supplementary-audit','fresh-audit','fresh-audit-predictions']:
        add(name+'.json',OUT/(name+'.json'))
    add('memory-score-cache-cpu-verification.json',Path('/work/memory-score-cache-cpu-verification.json'))
    for family in ['native','visual','visual-memory','adapted']:
        for name in ['development','selection','regression']:
            add('video-'+family+'/'+name+'.json',OUT/'video-selection'/family/(name+'.json'))
    add('long-run.json',OUT/'stitched/default-offload/completed.json')
    add('long-video.json',OUT/'public/video.json')
    add('long-baselines.json',OUT/'public/long-baselines.json')
    add('detection-review.json',OUT/'public/detection-review.json')
    for name in ['report','predictions','masks','review','baseline-control']:
        add('fresh-detection/'+name+'.json',OUT/'fresh-detection'/(name+'.json'))
    rink_selection=OUT/'rink-filter/selection.json'
    if rink_selection.exists():
        for name in ['development','selection']:add('rink-filter/'+name+'.json',OUT/'rink-filter'/(name+'.json'))
        if json.loads(rink_selection.read_text())['improves_validation_f1']:
            for name in ['development-masks','regression','regression-filtered-predictions','regression-masks','review']:
                add('rink-filter/'+name+'.json',OUT/'rink-filter'/(name+'.json'))
    return files


@app.function(image=image,cpu=4,memory=16384,timeout=900,volumes={'/data':volume})
def replays(long_only=False):
    from nbhl_vlm_reconnect import datasets,sha
    volume.reload(); files=[]; catalog=[]; reference_frames={}
    def samples(clip,method,raw,shape,source_hash):
        if clip not in reference_frames:return
        value={'source_tracks_sha256':source_hash,'shape':shape,
               'frames':[raw['frames'][f] for f in reference_frames[clip]]}
        files.append({'path':'data/reliability/sam3-20261024/sampled-predictions/'+clip+'-'+method+'.json',
                      'content':json.dumps(value,separators=(',',':'))+'\n'})
    def emit(clip,method,source,shape,folder,supported=False):
        raw=json.loads(source.read_text())
        value=compact(raw,shape,clip,method,sha(source),supported)
        samples(clip,method,raw,shape,sha(source))
        files.append({'path':folder+'/'+clip+'-'+method+'.json','content':json.dumps(value,separators=(',',':'))+'\n'})
        if source.with_name('metrics.json').exists():
            files.append({'path':'data/reliability/sam3-20261024/pairs/'+clip+'-'+method+'.json',
                          'content':source.with_name('metrics.json').read_text()})
        return value
    if not long_only:
        hybrid=json.loads((OUT/'selection.json').read_text())
        selected={r['family']:r['key'] for r in hybrid['winners']}
        videos={f:json.loads((OUT/'video-selection'/f/'selection.json').read_text())['best'] for f in ['native','visual','visual-memory','adapted']}
        for meta,_,_,_,_ in datasets('regression',labels=True):
            clip=meta['summary']['clip']; methods=[]
            reference_frames[clip]=sorted({r['frame'] for r in meta['observations']})
            for family,key in selected.items():
                method='sam3-'+family
                emit(clip,method,OUT/'regression'/key/clip/'tracks.json',meta['shape'],'media/tracking-fourhour',True)
                methods.append({'id':method,'family':family,'setting':key})
            emit(clip,'sam3-native',native_path('regression',clip)/'tracks.json',meta['shape'],'media/tracking-fourhour')
            methods.append({'id':'sam3-native','family':'baseline','setting':'default'})
            for family,spec in videos.items():
                if family!='adapted' and spec['variant']=='default':continue
                method='sam3-'+family+'-tuned'
                emit(clip,method,native_path('regression',clip,spec['variant'],spec['delta'])/'tracks.json',meta['shape'],'media/tracking-fourhour')
                methods.append({'id':method,'family':family,'setting':spec['variant'],'delta':spec['delta']})
            control=adapted_control_variant(videos['adapted']['variant'])
            emit(clip,'sam3-adapted-control',native_path('regression',clip,control)/'tracks.json',meta['shape'],'media/tracking-fourhour')
            methods.append({'id':'sam3-adapted-control','family':'adapted-control','setting':control})
            source=BASE/'vlm-reconnect-20261019-v1/embedding-evaluation/2b/regression/joint'/(clip+'.json')
            emit(clip,'vlm-embedding-2b-joint',source,meta['shape'],'media/tracking-fourhour',True)
            for baseline,method in [('sam2-qwen-joint','sam2-qwen-joint'),('sam2-raw','sam2-raw')]:
                path=OUT/'regression'/baseline/clip/'tracks.json'
                samples(clip,method,json.loads(path.read_text()),meta['shape'],sha(path))
                files.append({'path':'data/reliability/sam3-20261024/pairs/'+clip+'-'+method+'.json',
                              'content':(OUT/'regression'/baseline/clip/'metrics.json').read_text()})
            methods.append({'id':'vlm-embedding-2b-joint','family':'baseline','setting':'fixed prior Qwen 2B joint-image blend'})
            catalog.append({'clip':clip,'methods':methods,'frames':len(meta['frames'])})
    clip='pittsburgh-sam3-20min'; method='sam3-long'
    value=emit(clip,method,OUT/'stitched/default-offload/tracks.json',[1080,1920],'media/sam3-20261024')
    assert len(value['frames'])==18225
    methods=[{'id':method,'family':'long','setting':'default with CPU memory offload'}]
    baselines=[]
    for name,variant in [('sam2-long-generic','osnet_parts_generic'),('sam2-long-trained','osnet_parts_head')]:
        folder=BASE/'tracking-four-hour-v1/aligned-stitch-trial/recovery'/variant
        report=json.loads((folder/'completed.json').read_text())
        assert report['exact_overlap_frames']==675 and sha(folder/'tracks.json')==report['tracks_sha256']
        old=json.loads((folder/'tracks.json').read_text())
        assert len(old['frames'])==18225
        for i,row in enumerate(old['frames']):
            assert abs(row['source_time_s']-(1785+i/15))<1e-4
            assert all(o['observed'] is True for o in row['objects'])
        emit(clip,name,folder/'tracks.json',[1080,1920],'media/sam3-20261024',True)
        methods.append({'id':name,'family':'long-baseline','setting':variant})
        baselines.append({'method':name,'report':report})
    from nbhl_vlm_reconnect import dump
    dump(OUT/'public/long-baselines.json',{'results':baselines,'new_inference':False,
         'scope':'Existing automatic 20m15s runs, newly exported in full for qualitative comparison. No new full-game identity accuracy claim.'})
    volume.commit()
    catalog.append({'clip':clip,'methods':methods,'frames':18225})
    return {'files':files,'catalog':catalog}


@app.function(image=image,cpu=2,memory=4096,timeout=300,volumes={'/data':volume})
def poster():
    from PIL import Image
    from nbhl_vlm_reconnect import sha
    volume.reload()
    source=BASE/'tracking-four-hour-v1/long-cache-trial/pittsburgh-runtime-1785-315-absolute-gpu-bounds/frames/000001.jpg'
    target=OUT/'public/pittsburgh-sam3-20min.jpg'
    im=Image.open(source).convert('RGB'); im.resize((960,540)).save(target,quality=88)
    volume.commit();return {'sha256':sha(target),'source_sha256':sha(source)}


@app.function(image=image,cpu=2,memory=4096,timeout=300,volumes={'/data':volume})
def fresh_audit_frames():
    from PIL import Image,ImageDraw,ImageFont
    from nbhl_vlm_reconnect import sha,dump
    import shutil
    volume.reload()
    root=BASE/'tracking-four-hour-v1/long-cache-trial/pittsburgh-runtime-2385-315-absolute-gpu-bounds'
    meta=json.loads((root/'metadata.json').read_text())
    dest=OUT/'fresh-audit';dest.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    rows=[]
    for second in range(31):
        f=(135+second)*15;source=root/'frames'/f'{f+1:06d}.jpg'
        raw=Image.open(source).convert('RGB');raw.save(dest/f'{second:02d}-raw.jpg',quality=95)
        im=raw.copy();draw=ImageDraw.Draw(im);boxes=[]
        for i,box in enumerate(meta['frames'][f]['detections']):
            if box[4]<.1:continue
            x,y,u,v=box[:4];draw.rectangle([x,y,u,v],outline='#00e5ff',width=2)
            text=str(i);draw.rectangle([x,y,x+40,y+25],fill='white');draw.text((x+2,y+1),text,font=font,fill='black')
            boxes.append({'proposal':i,'bbox':box[:4],'confidence':box[4]})
        im.save(dest/f'{second:02d}-proposals.jpg',quality=95)
        rows.append({'second':second,'frame':f,'source_time_s':2520+second,'source_sha256':sha(source),'proposals':boxes})
    dump(dest/'frames.json',rows);volume.commit();return rows


@app.function(image=image,cpu=4,memory=8192,timeout=600,volumes={'/data':volume})
def detection_review():
    """Every regression frame, in fixed source/frame order; no example selection."""
    from PIL import Image,ImageDraw,ImageFont
    from nbhl_detection_metrics import match
    from nbhl_vlm_reconnect import sha,dump
    volume.reload()
    report=json.loads((FT/'regression.json').read_text())
    assert len(report['results'])==2
    original,adapted=report['results']
    predictions={r['variant']:{f['id']:f for f in json.loads((FT/r['variant']/'regression-predictions.json').read_text())} for r in report['results']}
    annotations={}
    for p in Path('/work/finetune-annotations').glob('*.json'):
        for f in json.loads(p.read_text())['frames']:annotations[f['id']]=f
    keys=sorted(predictions['original']);assert len(keys)==36 and set(keys)==set(predictions[adapted['variant']])
    dest=OUT/'public';dest.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    files=[];details=[]
    for sheet in range(6):
        canvas=Image.new('RGB',(1920,6*392),'white');draw=ImageDraw.Draw(canvas)
        for row,key in enumerate(keys[sheet*6:(sheet+1)*6]):
            annotation=annotations[key];source=Path(annotation['image']);assert sha(source)==annotation['sha256']
            picture=Image.open(source).convert('RGB');w,h=picture.size
            truth=predictions['original'][key]['players'];assert truth==predictions[adapted['variant']][key]['players']
            frame_detail={'frame_id':key,'image_sha256':annotation['sha256'],'models':{}}
            for col,spec in enumerate([None,original,adapted]):
                im=picture.resize((640,360));dr=ImageDraw.Draw(im)
                def box(coords,color,width=2):dr.rectangle([v/s*size for v,s,size in zip(coords,[w,h,w,h],[640,360,640,360])],outline=color,width=width)
                if spec is None:
                    for obj in truth:box(obj['bbox'],'#008cff')
                    title=f'{key} | {len(truth)} reference players'
                else:
                    ranked,_=match([predictions[spec['variant']][key]],.5)
                    pred=[p for p in ranked if p['confidence']>=spec['threshold']]
                    for obj in pred:box(obj['bbox'],'#00df55' if obj['hit'] else '#ff3a35')
                    tp=sum(p['hit'] for p in pred);fp=len(pred)-tp;fn=len(truth)-tp
                    title=f'{"Pretrained" if col==1 else "Fine-tuned"} | {tp} found, {fp} false, {fn} missed'
                    frame_detail['models'][spec['variant']]={'tp':tp,'fp':fp,'fn':fn,'threshold':spec['threshold']}
                x=col*640;y=row*392;draw.text((x+5,y+7),title,font=font,fill='black');canvas.paste(im,(x,y+32))
            details.append(frame_detail)
        target=dest/f'detection-review-{sheet+1}.jpg';canvas.save(target,quality=88)
        files.append({'name':target.name,'sha256':sha(target),'bytes':target.stat().st_size})
    for spec in report['results']:
        for metric in ['tp','fp','fn']:
            assert sum(r['models'][spec['variant']][metric] for r in details)==spec['metrics'][metric]
    value={'files':files,'frames':details,'selection':'All 36 reused regression frames, sorted by frame ID, six per sheet.',
           'legend':'Blue: reviewed reference. Green: predicted box matched at IoU>=0.5. Red: unmatched predicted box.',
           'adapted_variant':adapted['variant'],'regression_sha256':sha(FT/'regression.json')}
    dump(dest/'detection-review.json',value);volume.commit();return value


@app.local_entrypoint()
def main(task='reports',long_only:bool=False,partial:bool=False):
    if task=='poster':print(json.dumps(poster.remote()));return
    if task=='fresh-audit':
        result=fresh_audit_frames.remote()
        (ROOT/'notes/ball-hockey/sam3-20261024/fresh-audit-frames.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'frames':len(result)}));return
    if task=='detection-review':
        result=detection_review.remote()
        (ROOT/'notes/ball-hockey/sam3-20261024/detection-review.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='frames'}));return
    if task=='reports':
        import hashlib
        result=reports.remote(partial);root=ROOT/'notes/ball-hockey/sam3-20261024/export'
        for r in result:
            target=root/r['name'];target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(json.dumps(r['value'],indent=2)+'\n')
            assert hashlib.sha256(target.read_bytes()).hexdigest()==r['sha256']
        (root/'export-manifest.json').write_text(json.dumps({r['name']:r['sha256'] for r in result},indent=2)+'\n')
    elif task=='replays':
        result=replays.remote(long_only);root=ROOT.parent/'hockey-tracking/web'
        asset_paths=set(json.loads((root.parent/'site-assets.json').read_text())['files'])
        for r in result['files']:
            # Existing release-backed baseline replays are already available.
            # Do not create a duplicate source file at an archive-owned path.
            if r['path'] in asset_paths:continue
            target=root/r['path']
            if target.exists() and r['path'].startswith('media/') and r['path'].endswith('-sam3-native.json'):
                old=json.loads(target.read_text());new=json.loads(r['content'])
                assert old['source_tracks_sha256']==new['source_tracks_sha256']
                assert old['frames']==new['frames'] and old['track_id_map']==new['track_id_map']
                continue
            target.parent.mkdir(parents=True,exist_ok=True);target.write_text(r['content'])
        (ROOT/'notes/ball-hockey/sam3-20261024/replay-catalog.json').write_text(json.dumps(result['catalog'],indent=2)+'\n')
    else:raise ValueError(task)
    print(json.dumps({'task':task,'exported':len(result) if isinstance(result,list) else len(result['files'])}))
