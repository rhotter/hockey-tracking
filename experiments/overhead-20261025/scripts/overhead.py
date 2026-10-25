"""Development experiment; manually calibrated, not an independent accuracy benchmark."""
from pathlib import Path
import json
import hashlib

BASE=Path('/data/ball-hockey/reliability-20261018')
OUT=BASE/'overhead-20261025-v1'
SOURCE='OLpK8enSjJs'
START=900
DURATION=300
FPS=15
def read(p): return json.loads(Path(p).read_text())
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):
    with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()

def prepare(variant,volume):
    import subprocess,tarfile,shutil
    from PIL import Image,ImageDraw
    raw=Path('/data/raw')/(SOURCE+'.mp4');assert raw.exists()
    frames=OUT/'frames';frames.mkdir(parents=True,exist_ok=True)
    if len(list(frames.glob('*.jpg')))!=DURATION*FPS:
        subprocess.run(['ffmpeg','-v','error','-threads','4','-ss',str(START),'-i',str(raw),
            '-t',str(DURATION),'-an','-vf',f'setpts=PTS-STARTPTS,fps={FPS}:start_time=0',
            '-frames:v',str(DURATION*FPS),'-q:v','2','-y',str(frames/'%06d.jpg')],check=True)
    count=len(list(frames.glob('*.jpg')));assert count==DURATION*FPS
    review=OUT/'review';review.mkdir(exist_ok=True)
    selected=list(range(0,DURATION*FPS,15*FPS))+[DURATION*FPS-1]
    for page in range((len(selected)+5)//6):
        sheet=Image.new('RGB',(1280,3*384),'white');d=ImageDraw.Draw(sheet)
        for i,f in enumerate(selected[page*6:page*6+6]):
            p=frames/f'{f+1:06d}.jpg';im=Image.open(p);im.thumbnail((640,360))
            x,y=(i%2)*640,(i//2)*384;sheet.paste(im,(x,y+24))
            d.text((x+6,y+5),f'frame {f} / video {START+f/FPS:.2f}s',fill='black')
            shutil.copyfile(p,review/f'raw-{f:04d}.jpg')
        sheet.save(review/f'sheet-{page}.jpg',quality=94)
    bank=read(BASE/'camera-bank-v1'/SOURCE/'bank.json')
    dump(review/'bank-summary.json',{'anchor':bank['anchor'],'frames':[{k:r[k]for k in ['id','image']}for r in bank['frames']]})
    anchor=next(r for r in bank['frames']if r['id']==bank['anchor'])
    shutil.copyfile(anchor['image'],review/'anchor.jpg')
    with tarfile.open(OUT/'review.tar.gz','w:gz')as tar:
        for p in review.iterdir():tar.add(p,arcname=p.name)
    media=OUT/'media';media.mkdir(exist_ok=True)
    if not (media/'nj-0900-5min.mp4').exists():
        subprocess.run(['ffmpeg','-v','error','-threads','4','-framerate',str(FPS),'-i',str(frames/'%06d.jpg'),
            '-an','-vf','scale=-2:720','-c:v','libx264','-threads','4','-preset','fast','-crf','25',
            '-pix_fmt','yuv420p','-movflags','+faststart','-y',str(media/'nj-0900-5min.mp4')],check=True)
    return {'source':SOURCE,'start_s':START,'duration_s':DURATION,'fps':FPS,'frames':count,
            'raw_sha256':sha(raw),'review_archive':str(OUT/'review.tar.gz'),
            'video':str(media/'nj-0900-5min.mp4'),'video_sha256':sha(media/'nj-0900-5min.mp4'),
            'camera_bank_frames':len(bank['frames'])}

def detect(variant,volume):
    import cv2,numpy as np,torch,time
    from ultralytics import YOLO
    from scipy.optimize import linear_sum_assignment
    from nbhl_detection_metrics import iou
    cv2.setNumThreads(2);torch.set_num_threads(4)
    weights=BASE/'training/yolo11m-multi-s17-f10-e80-remap-v2-6f459b18/weights/best.pt'
    assert sha(weights)=='f6a5da194807cfe78ae46be40571a9da3bca9d9e94e9ba2311ea857f3ff75ac3'
    model=YOLO(str(weights));pose=YOLO('/data/models/yolo11m-pose.pt')
    rows=[];started=time.monotonic();previous=None;scene=0
    for f,path in enumerate(sorted((OUT/'frames').glob('*.jpg'))):
        frame=cv2.imread(str(path));h,w=frame.shape[:2]
        gray=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(160,90))
        # Candidate cuts are recorded for review, never triggered by MAD alone.
        mad=float(np.mean(cv2.absdiff(gray,previous)))if previous is not None else 0.
        hist=cv2.calcHist([gray],[0],None,[32],[0,256]);cv2.normalize(hist,hist)
        oldhist=cv2.calcHist([previous],[0],None,[32],[0,256])if previous is not None else hist.copy()
        cv2.normalize(oldhist,oldhist)
        hd=float(cv2.compareHist(hist,oldhist,cv2.HISTCMP_BHATTACHARYYA))
        reset=mad>55 and hd>.5
        if reset:scene+=1
        previous=gray
        boxes=model.predict(frame,imgsz=1280,conf=.025,iou=.65,classes=[0],max_det=100,device=0,verbose=False)[0].boxes.data.cpu().numpy().reshape(-1,6)
        result=pose.predict(frame,imgsz=1280,conf=.15,iou=.65,device=0,verbose=False)[0]
        poses={};pb=result.boxes.xyxy.cpu().numpy()
        if len(boxes) and len(pb):
            mat=np.array([[iou(a[:4],b)for b in pb]for a in boxes]);aa,bb=linear_sum_assignment(-mat)
            for a,b in zip(aa,bb):
                if mat[a,b]>=.5:
                    pts=result.keypoints.xy[b].cpu().numpy();scores=result.keypoints.conf[b].cpu().numpy()
                    poses[int(a)]={'ankles':pts[15:17].tolist(),'scores':scores[15:17].tolist(),'match_iou':float(mat[a,b])}
        colors=[]
        for x,y,u,v,confidence,cl in boxes:
            crop=frame[max(0,round(y+.15*(v-y))):min(h,round(y+.55*(v-y))),max(0,round(x+.2*(u-x))):min(w,round(x+.8*(u-x)))]
            red=white=0.
            if crop.size:
                hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
                red=float(((((hsv[:,:,0]<12)|(hsv[:,:,0]>170)))&(hsv[:,:,1]>70)&(hsv[:,:,2]>50)).mean())
                white=float(((hsv[:,:,1]<60)&(hsv[:,:,2]>120)).mean())
            colors.append({'red':red,'white':white})
        rows.append({'frame':f,'source_time_s':START+f/FPS,'detections':boxes.tolist(),
                     'poses':poses,'colors':colors,'scene':scene,'reset':reset,'mad':mad,'hist_distance':hd})
        if f%300==0:print(json.dumps({'stage':'detect','frame':f,'seconds':time.monotonic()-started}),flush=True)
    report={'summary':{'clip':'nj-0900-5min'},'shape':[h,w],'frames':rows,'weights_sha256':sha(weights),
            'pose_weights_sha256':sha('/data/models/yolo11m-pose.pt'),'seconds':time.monotonic()-started}
    dump(OUT/'detections.json',report)
    return {k:v for k,v in report.items()if k!='frames'}|{'frames':len(rows),'scene_resets':[r['frame']for r in rows if r['reset']]}

def sam(variant,volume):
    import numpy as np,torch
    from sam_tracker import track
    from nbhl_sam2_lazy_frames import install
    torch.set_num_threads(4);meta=read(OUT/'detections.json');meta['summary']['input_sha256']=sha(OUT/'detections.json')
    weights=Path('/data/models/sam2.1_hiera_large.pt')
    assert sha(weights)=='2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318'
    restore,caches=install(capacity=4)
    try:result=track(meta,{'resets':np.array([r['reset']for r in meta['frames']])},OUT/'frames',weights,'large',.1)
    finally:restore()
    result['frame_cache']=[c.statistics()for c in caches]
    dump(OUT/'tracks-sam2.json',result)
    return {k:v for k,v in result.items()if k!='frames'}

def register(variant,volume):
    from calibration import register as run
    return run(variant,volume)

def ball(variant,volume):
    import cv2,torch,time
    from ultralytics import YOLO
    torch.set_num_threads(4);cv2.setNumThreads(2)
    weights=BASE/'ball/training/yolo11m-crops640-s17-f10-e80-c6ab3e44/weights/best.pt'
    assert weights.exists()
    model=YOLO(str(weights));ball_class=next(k for k,v in model.names.items()if v=='active_ball')
    rows=[];started=time.monotonic()
    for f,path in enumerate(sorted((OUT/'frames').glob('*.jpg'))):
        im=cv2.imread(str(path))
        result=model.predict(im,imgsz=1280,conf=.25,iou=.5,classes=[ball_class],device=0,max_det=20,verbose=False)[0]
        rows.append({'frame':f,'detections':result.boxes.data.cpu().numpy().tolist()})
        if f%450==0:print({'stage':'ball','frame':f,'seconds':round(time.monotonic()-started)},flush=True)
    dump(OUT/'ball.json',{'weights_sha256':sha(weights),'frames':rows,'seconds':time.monotonic()-started})
    return {'frames':len(rows),'candidate_frames':sum(bool(r['detections'])for r in rows),'seconds':time.monotonic()-started,'weights_sha256':sha(weights)}

def roles(variant,volume):
    import numpy as np,torch,time
    from PIL import Image
    from transformers import AutoModel,AutoImageProcessor
    torch.set_num_threads(4)
    tracks=read(OUT/'tracks-sam2.json');byid={}
    for row in tracks['frames']:
        for o in row['objects']:byid.setdefault(o['track_id'],[]).append((row['frame'],o['bbox']))
    role_dir=BASE/'role-classifier/dinov2-parts-6f459b18'
    meta=read(role_dir/'frozen-selection.json');classifier=np.load(role_dir/'classifier.npz')
    model=AutoModel.from_pretrained(meta['model'],revision=meta['revision'],local_files_only=True).cuda().eval()
    processor=AutoImageProcessor.from_pretrained(meta['model'],revision=meta['revision'],local_files_only=True)
    result={};started=time.monotonic()
    for i,(tid,rows)in enumerate(byid.items()):
        indices=np.unique(np.linspace(0,len(rows)-1,min(15,len(rows))).round().astype(int));scores=[];crops=[]
        for index in indices:
            f,box=rows[index];im=Image.open(OUT/'frames'/f'{f+1:06d}.jpg').convert('RGB');crop=im.crop(box);w,h=crop.size
            for region in [(0,0,w,h),(.1*w,.15*h,.9*w,.65*h),(.1*w,.55*h,.9*w,h)]:
                crops.append(crop.crop(region).resize((224,224),Image.Resampling.BICUBIC))
        with torch.inference_mode():
            inputs=processor(images=crops,do_resize=False,do_center_crop=False,return_tensors='pt').to('cuda')
            z=model(**inputs).last_hidden_state[:,0].float().cpu().numpy()
        z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-8)
        x=z.reshape(len(indices),-1)/np.sqrt(3.)
        scores=1/(1+np.exp(-(x@classifier['coef'].T+classifier['intercept']).ravel()))
        result[tid]={'median_player_probability':float(np.median(scores)),'keep':bool(np.median(scores)>=float(classifier['threshold'])),
                     'samples':[{'frame':rows[int(j)][0],'probability':float(s)}for j,s in zip(indices,scores)]}
        if i%30==0:print({'stage':'roles','tracks':i,'seconds':round(time.monotonic()-started)},flush=True)
    report={'classifier_sha256':sha(role_dir/'classifier.npz'),'threshold':float(classifier['threshold']),
            'method':'Frozen DINOv2 body/torso/legs linear role classifier; median of up to 15 equally spaced observations per automatic fragment.',
            'tracks':result,'seconds':time.monotonic()-started}
    dump(OUT/'roles.json',report)
    return {k:v for k,v in report.items()if k!='tracks'}|{'tracks':len(result),'kept':sum(r['keep']for r in result.values())}

def review_tracks(variant,volume):
    import tarfile,numpy as np
    from PIL import Image,ImageDraw,ImageFont
    tracks=read(OUT/'tracks-sam2.json');det=read(OUT/'detections.json')['frames'];byid={}
    for row in tracks['frames']:
        for o in row['objects']:
            byid.setdefault(o['track_id'],[]).append({'frame':row['frame'],**o,'color':det[row['frame']]['colors'][o['detection_index']]})
    out=OUT/'track-review';out.mkdir(exist_ok=True)
    selected=set(variant.split(','))if variant else set(byid)
    records=[]
    for tid,rows in byid.items():
        if tid not in selected:continue
        if not variant and len(rows)<15:continue
        indices=np.unique(np.linspace(0,len(rows)-1,min(7,len(rows))).round().astype(int))
        sheet=Image.new('RGB',(7*180,330),'white');d=ImageDraw.Draw(sheet)
        for j,index in enumerate(indices):
            r=rows[index];im=Image.open(OUT/'frames'/f'{r["frame"]+1:06d}.jpg').convert('RGB')
            a,b,c,e=r['bbox'];w,h=c-a,e-b;crop=im.crop((max(0,a-.1*w),max(0,b-.05*h),min(im.width,c+.1*w),min(im.height,e+.05*h)))
            crop.thumbnail((175,285));sheet.paste(crop,(j*180+(180-crop.width)//2,40))
            d.text((j*180+4,4),f'T{tid} f{r["frame"]}',fill='black');d.text((j*180+4,20),f'{START+r["frame"]/FPS:.2f}s',fill='black')
        sheet.save(out/f'track-{int(tid):04d}.jpg',quality=95)
        records.append({'track_id':tid,'count':len(rows),'first':rows[0]['frame'],'last':rows[-1]['frame'],
            'red':float(np.median([r['color']['red']for r in rows])),'white':float(np.median([r['color']['white']for r in rows])),
            'samples':[rows[int(i)]for i in indices]})
    for f in range(0,DURATION*FPS,5*FPS):
        im=Image.open(OUT/'frames'/f'{f+1:06d}.jpg').convert('RGB');d=ImageDraw.Draw(im)
        for o in tracks['frames'][f]['objects']:
            a,b,c,e=o['bbox'];d.rectangle((a,b,c,e),outline='yellow',width=2);d.text((a,max(0,b-13)),o['track_id'],fill='yellow',stroke_width=1,stroke_fill='black')
        im.thumbnail((1280,720));im.save(out/f'frame-{f:04d}.jpg',quality=94)
    dump(out/'tracks.json',records)
    with tarfile.open(OUT/'track-review.tar.gz','w:gz')as tar:
        for p in out.iterdir():tar.add(p,arcname=p.name)
        tar.add(OUT/'tracks-sam2.json',arcname='tracks-sam2.json')
        tar.add(OUT/'detections.json',arcname='detections.json')
    return {'tracks':len(records),'archive':str(OUT/'track-review.tar.gz')}

def export(variant,volume):
    from export_replay import export as run
    return run(variant,volume)

def identity_review(variant,volume):
    """Dense chronological identity review, including track gaps and endpoints."""
    import tarfile
    from PIL import Image,ImageDraw
    wanted=set(variant.split(','));tracks=read(OUT/'tracks-sam2.json')['frames'];byid={}
    for row in tracks:
        for o in row['objects']:
            if o['track_id']in wanted:byid.setdefault(o['track_id'],[]).append({'frame':row['frame'],**o})
    out=OUT/'identity-review';out.mkdir(exist_ok=True);manifest={}
    for tid,rows in byid.items():
        frames={r['frame']:r for r in rows};sample={rows[0]['frame'],rows[-1]['frame']}|{r['frame']for r in rows if r['frame']%15==0}
        for a,b in zip(rows,rows[1:]):
            if b['frame']-a['frame']>1:sample.update([a['frame'],b['frame']])
        samples=sorted(sample);manifest[tid]=samples
        for offset in range(0,len(samples),32):
            sheet=Image.new('RGB',(8*170,4*250),'white');d=ImageDraw.Draw(sheet)
            for j,f in enumerate(samples[offset:offset+32]):
                o=frames[f];im=Image.open(OUT/'frames'/f'{f+1:06d}.jpg');a,b,c,e=o['bbox'];w,h=c-a,e-b
                crop=im.crop((max(0,a-.1*w),max(0,b-.05*h),min(1920,c+.1*w),min(1080,e+.05*h)));crop.thumbnail((165,220))
                x,y=(j%8)*170,(j//8)*250;sheet.paste(crop,(x+(170-crop.width)//2,y+28));d.text((x+4,y+5),f'T{tid} f{f} / {f/FPS:.2f}s',fill='black')
            sheet.save(out/f'T{tid}-{offset//32:02d}.jpg',quality=96)
    dump(out/'manifest.json',manifest)
    archive=OUT/'identity-review.tar.gz'
    with tarfile.open(archive,'w:gz')as tar:
        for p in out.iterdir():tar.add(p,arcname=p.name)
    return {'samples':manifest,'archive':str(archive)}

def refine(variant,volume):
    from refine_calibration import refine as run
    return run(variant,volume)

def frames(variant,volume):
    import tarfile,shutil
    from PIL import Image,ImageDraw
    numbers=[int(x)for x in variant.split(',')];assert all(0<=f<DURATION*FPS for f in numbers)
    dest=OUT/'additional-review';dest.mkdir(exist_ok=True)
    sheet=Image.new('RGB',(1280,384*((len(numbers)+1)//2)),'white');d=ImageDraw.Draw(sheet)
    for i,f in enumerate(numbers):
        p=OUT/'frames'/f'{f+1:06d}.jpg';shutil.copyfile(p,dest/f'raw-{f:04d}.jpg')
        im=Image.open(p);im.thumbnail((640,360));x,y=(i%2)*640,(i//2)*384;sheet.paste(im,(x,y+24));d.text((x+5,y+5),f'frame {f} / {START+f/FPS:.3f}s',fill='black')
    sheet.save(dest/'additional-sheet.jpg',quality=94)
    with tarfile.open(OUT/'additional-review.tar.gz','w:gz')as tar:
        for p in dest.iterdir():tar.add(p,arcname=p.name)
    return {'frames':numbers,'archive':str(OUT/'additional-review.tar.gz')}
