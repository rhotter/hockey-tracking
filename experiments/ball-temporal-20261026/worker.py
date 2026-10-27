"""Training and inference workers; all source frames remain on Modal."""
from pathlib import Path
import json,hashlib,time

BASE=Path('/data/ball-hockey/reliability-20261018')
OUT=BASE/'ball-temporal-20261026-v1'
SOURCE=BASE/'overhead-20261025-v1/frames'
WARMSTART=BASE/'ball/heatmap-v1/resnet18-rgb-heatmap-s17-e120-c6ab3e44/best.pt'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,separators=(',',':'))+'\n');tmp.replace(p)

def protocol_file(variant):
    if variant.endswith('-hard'): return '/work/protocol-hard.json'
    return '/work/protocol-center.json' if '-center' in variant else '/work/protocol.json'

def neighbor(path,offset):
    path=Path(path);stem=path.stem;prefix='frame-'if stem.startswith('frame-')else''
    number=int(stem.removeprefix(prefix))+offset
    candidate=path.parent/f'{prefix}{max(1,number):06d}.jpg'
    return candidate if candidate.exists()else path

def registration(previous,current):
    import cv2,numpy as np
    # RANSAC background affine removes camera movement. Players may move freely.
    scale=.25
    a=cv2.resize(cv2.cvtColor(previous,cv2.COLOR_RGB2GRAY),None,fx=scale,fy=scale)
    b=cv2.resize(cv2.cvtColor(current,cv2.COLOR_RGB2GRAY),None,fx=scale,fy=scale)
    p=cv2.goodFeaturesToTrack(a,600,.015,7)
    if p is None or len(p)<12:return np.eye(2,3,dtype=np.float32),0.
    q,status,_=cv2.calcOpticalFlowPyrLK(a,b,p,None,winSize=(21,21),maxLevel=3)
    if q is None:return np.eye(2,3,dtype=np.float32),0.
    valid=status.ravel().astype(bool)
    if valid.sum()<12:return np.eye(2,3,dtype=np.float32),0.
    m,inliers=cv2.estimateAffinePartial2D(p[valid],q[valid],method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=1000)
    quality=float(inliers.mean())if inliers is not None else 0.
    if m is None or quality<.35 or not .8<float(np.linalg.det(m[:,:2]))<1.25:return np.eye(2,3,dtype=np.float32),0.
    m[:,2]/=scale
    if np.linalg.norm(m[:,2])>180:return np.eye(2,3,dtype=np.float32),0.
    return m.astype(np.float32),quality

def prepare_data(commit):
    import cv2,numpy as np,tarfile
    from PIL import Image,ImageDraw
    cv2.setNumThreads(2);OUT.mkdir(parents=True,exist_ok=True)
    labels=read('/work/training-labels.json')['rows']
    additional=read('/work/regression-labels.json')['rows']
    rows=[]
    for row in labels+additional:
        p=Path(row['image']);assert p.exists(),p
        fingerprint=sha(p)
        if row.get('sha256'):assert fingerprint==row['sha256'],p
        rows.append({'path':str(p),'sha256':fingerprint,'neighbor_sha256':{str(o):sha(neighbor(p,o))for o in [-2,2]}})
    assert WARMSTART.exists()
    # Prediction-free review sheets: one raw central frame, two temporal neighbors.
    protocol=read('/work/protocol.json');dest=OUT/'raw-review';dest.mkdir(exist_ok=True);manifest=[]
    for clip in protocol['fresh_ball_test']['clips']:
        folder=BASE/'new-games-20261025-v1/frames'/clip
        assert len(list(folder.glob('*.jpg')))==1350
        for f in protocol['fresh_ball_test']['sample_frames']:
            p=folder/f'{f+1:06d}.jpg';im=Image.open(p).convert('RGB');w,h=im.size
            sheet=Image.new('RGB',(1920,1660),'white');d=ImageDraw.Draw(sheet)
            center=im.resize((1920,1080));sheet.paste(center,(0,30));d.text((8,8),f'{clip} | frame {f} | native {w} x {h} | central raw frame; no predictions',fill='black')
            for i,delta in enumerate([-3,3]):
                p2=folder/f'{f+delta+1:06d}.jpg';context=Image.open(p2).convert('RGB');context.thumbnail((960,540));sheet.paste(context,(i*960,1120));d.text((i*960+8,1105),f'Frame {f+delta}',fill='black')
            filename=f'{clip}-{f:04d}.jpg';sheet.save(dest/filename,quality=95)
            manifest.append({'clip':clip,'frame':f,'image':filename,'source_sha256':sha(p),'width':w,'height':h,'neighbors':[f-3,f+3]})
    write(dest/'manifest.json',{'protocol_sha256':sha('/work/protocol.json'),'samples':manifest})
    with tarfile.open(OUT/'raw-review.tar.gz','w:gz')as tar:tar.add(dest,arcname='raw-review')
    report={'training_rows':len(labels),'source_images':rows,'warmstart_sha256':sha(WARMSTART),'raw_review_samples':len(manifest),'protocol_sha256':sha('/work/protocol.json'),'archive':str(OUT/'raw-review.tar.gz')}
    write(OUT/'preparation.json',report);commit();return {k:v for k,v in report.items()if k!='source_images'}

def train_model(variant,commit):
    import random,math,cv2,numpy as np,torch
    from torch.utils.data import Dataset,DataLoader
    from temporal_model import TemporalBall,inputs,focal_heatmap
    torch.set_num_threads(4);cv2.setNumThreads(1)
    protocol_path=protocol_file(variant)
    protocol=read(protocol_path);assert variant in protocol['training_variants']
    random.seed(protocol['seed']);np.random.seed(protocol['seed']);torch.manual_seed(protocol['seed'])
    dest=OUT/variant;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'completed.json').exists():return read(dest/'completed.json')
    labels=read('/work/training-labels.json')['rows'];cache={};registrations=[]
    for i,row in enumerate(labels):
        p=Path(row['image']);center=cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB)
        if row.get('sha256'):assert sha(p)==row['sha256']
        sequence=[];quality=[]
        for offset in [-2,0,2]:
            if offset==0:sequence.append(center);quality.append(1.);continue
            im=cv2.cvtColor(cv2.imread(str(neighbor(p,offset))),cv2.COLOR_BGR2RGB)
            matrix,q=registration(im,center)
            sequence.append(cv2.warpAffine(im,matrix,(center.shape[1],center.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101));quality.append(q)
        cache[i]=np.stack(sequence);registrations.append({'image_sha256':sha(p),'quality':quality})
    train_indices=[i for i,r in enumerate(labels)if r['split']=='train']
    valid_indices=[i for i,r in enumerate(labels)if r['split']=='validation']
    size=protocol['crop_size'];yy,xx=np.mgrid[:size//2,:size//2]
    hard_examples=[]
    if variant.endswith('-hard'):
        mined=read(OUT/'rgb-expanded/train.json');assert mined['complete']
        by_key={(r['clip'],r['frame']):r for r in mined['frames']}
        for i in train_indices:
            label=labels[i]
            # Only previously reviewed labels can define the absence region.
            # Model pseudo labels never create mined negative labels.
            if label['label_kind'].startswith('published'):continue
            pred=by_key[(label['clip'],label['frame'])]
            if label.get('sha256'):assert pred['source_sha256']==label['sha256']
            target=np.array(label['ball']['center'])
            for candidate in pred['candidates']:
                if candidate['score']<.015:continue
                if np.max(np.abs(np.array(candidate['pixel'])-target))<size/2+85:continue
                hard_examples.append((i,candidate['pixel'],candidate['score']))
        assert hard_examples,'No valid mined negative crops'
        write(dest/'mined-negatives.json',{'source_sha256':sha(OUT/'rgb-expanded/train.json'),'examples':[
            {'training_index':i,'image_sha256':sha(labels[i]['image']),'pixel':p,'score':s}for i,p,s in hard_examples]})

    def crop(sequence,origin):
        x,y=map(int,origin);h,w=sequence.shape[1:3]
        pad=((-min(0,y),max(0,y+size-h)),(-min(0,x),max(0,x+size-w)))
        if any(v for pair in pad for v in pair):
            sequence=np.pad(sequence,((0,0),pad[0],pad[1],(0,0)),mode='reflect');x+=pad[1][0];y+=pad[0][0]
        return sequence[:,y:y+size,x:x+size].copy()

    def sample(index,positive,rng):
        sequence=cache[index];point=np.array(labels[index]['ball']['center'],float);h,w=sequence.shape[1:3]
        if positive:origin=np.round(point-rng.uniform(40,size-40,2)).astype(int)
        else:
            for _ in range(100):
                origin=np.array([rng.randint(0,max(1,w-size)),rng.randint(0,max(1,h-size))])
                local=point-origin
                if not np.all((local>=-100)&(local<=size+100)):break
            else:raise ValueError('No negative crop away from the active ball')
        return crop(sequence,origin),point-origin if positive else None

    def synthetic(sequence,rng):
        donor_index=rng.choice(train_indices);donor=cache[donor_index][1];p=labels[donor_index]['ball']['center']
        patch=cv2.getRectSubPix(donor,(13,13),tuple(map(float,p))).astype(np.float32)
        radius=rng.uniform(2.5,5.5);gy,gx=np.mgrid[-6:7,-6:7]
        alpha=np.clip(radius+.75-np.hypot(gx,gy),0,1).astype(np.float32)
        center=rng.uniform(64,size-64,2);velocity=rng.uniform(-28,28,2);acceleration=rng.uniform(-5,5,2)
        if rng.random()<.2:velocity*=.05
        # Motion blur and a hidden middle-frame example teach temporal evidence.
        hidden=rng.random()<.2
        for j,t in enumerate([-1,0,1]):
            if hidden and j==1:continue
            pos=center+t*velocity+.5*t*t*acceleration
            canvas=np.zeros((size,size,3),np.float32);mask=np.zeros((size,size),np.float32)
            steps=5 if np.linalg.norm(velocity)>12 else 1
            for blur in np.linspace(-.18,.18,steps):
                q=pos+velocity*blur;matrix=np.array([[1,0,q[0]-6],[0,1,q[1]-6]],np.float32)
                canvas+=cv2.warpAffine(patch*alpha[:,:,None],matrix,(size,size))/steps
                mask+=cv2.warpAffine(alpha,matrix,(size,size))/steps
            sequence[j]=np.clip(sequence[j]*(1-mask[:,:,None])+canvas,0,255).astype(np.uint8)
        return sequence,center

    class Training(Dataset):
        def __len__(self):return len(train_indices)*16
        def __getitem__(self,index):
            rng=np.random;item=int(rng.choice(train_indices));positive=rng.random()<.7
            synth=variant.endswith('synthetic')and rng.random()<.35
            sequence,target=sample(item,positive and not synth,rng)
            if hard_examples and not positive and rng.random()<.75:
                mined_index,center,_=hard_examples[int(rng.randint(len(hard_examples)))]
                origin=np.round(np.array(center)-size/2+rng.uniform(-30,30,2)).astype(int)
                assert not np.all((np.array(labels[mined_index]['ball']['center'])-origin>=-50)&(np.array(labels[mined_index]['ball']['center'])-origin<=size+50))
                sequence,target=crop(cache[mined_index],origin),None
            if synth:sequence,target=synthetic(sequence,rng)
            if rng.random()<.5:
                sequence=sequence[:,:,::-1].copy()
                if target is not None:target[0]=size-1-target[0]
            if rng.random()<.5:sequence=sequence[::-1].copy()
            if rng.random()<.15:
                sequence=np.stack([cv2.GaussianBlur(im,(3,3),rng.uniform(.2,.7))for im in sequence])
            sequence=np.clip(sequence.astype(np.float32)*rng.uniform(.8,1.2)+rng.uniform(-12,12),0,255)/255
            heat=np.zeros((size//2,size//2),np.float32)
            if target is not None:
                x,y=np.round(target/2).astype(int)
                heat=np.exp(-((xx-x)**2+(yy-y)**2)/(2*1.5**2)).astype(np.float32)
            return torch.from_numpy(sequence.transpose(0,3,1,2).copy()),torch.from_numpy(heat[None])

    validation=[];valid_meta=[]
    for i in valid_indices:
        for n in range(8):
            positive=n<4;sequence,target=sample(i,positive,np.random.RandomState(29000+i*10+n))
            validation.append(torch.from_numpy(sequence.transpose(0,3,1,2).copy()))
            valid_meta.append({'image_sha256':sha(labels[i]['image']),'positive':positive,'target':target.tolist()if target is not None else None})
    validation=torch.stack(validation);total_positive=sum(r['positive']for r in valid_meta)
    loader=DataLoader(Training(),batch_size=protocol['batch_size'],shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True,prefetch_factor=2)
    warm_path=OUT/'rgb-expanded/best.pt' if variant.endswith('-hard') else WARMSTART
    warm=torch.load(warm_path,map_location='cpu',weights_only=False)
    model=TemporalBall(variant,warm).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,protocol['epochs'],eta_min=1e-6)
    best=(-1.,-1);history=[];started=time.monotonic()
    for epoch in range(protocol['epochs']):
        model.train();losses=[]
        for sequence,target in loader:
            sequence=sequence.cuda(non_blocking=True);target=target.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=focal_heatmap(model(inputs(sequence,variant)),target)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();losses.append(float(loss.detach()))
        scheduler.step();model.eval();predictions=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for sequence in validation.split(32):
                heat=model(inputs(sequence.cuda().float()/255,variant)).float().sigmoid()
                scores,indices=heat.flatten(1).max(1)
                for confidence,idx in zip(scores.cpu().tolist(),indices.cpu().tolist()):predictions.append({'score':confidence,'pixel':[2*(idx%(size//2)),2*(idx//(size//2))]})
        options=[]
        for threshold in [.05,.1,.2,.3,.5,.7,.9]:
            tp=fp=fn=0;errors=[]
            for pred,meta in zip(predictions,valid_meta):
                output=pred['score']>=threshold;error=math.dist(pred['pixel'],meta['target'])if meta['positive']else None
                hit=output and error is not None and error<=12
                tp+=hit;fp+=output and not hit;fn+=meta['positive']and not hit
                if hit:errors.append(error)
            options.append({'threshold':threshold,'tp':tp,'fp':fp,'fn':fn,'f1':2*tp/max(1,2*tp+fp+fn),'median_hit_error_px':float(np.median(errors))if errors else None})
        selected=max(options,key=lambda r:(r['f1'],-r['fp'],r['threshold']))
        row={'epoch':epoch+1,'loss':float(np.mean(losses)),'validation':selected,'seconds':time.monotonic()-started};history.append(row)
        if (selected['f1'],-selected['fp'])>best:
            best=(selected['f1'],-selected['fp']);torch.save({'variant':variant,'state_dict':{k:v.detach().cpu()for k,v in model.state_dict().items()},'epoch':epoch+1,'validation':selected,'protocol_sha256':sha(protocol_path)},dest/'best.pt')
            write(dest/'validation-predictions.json',{'samples':valid_meta,'predictions':predictions})
        write(dest/'history.json',history)
        print({'variant':variant,**row},flush=True)
        if epoch%5==4:commit()
    ckpt=torch.load(dest/'best.pt',map_location='cpu',weights_only=False)
    result={'variant':variant,'epochs':protocol['epochs'],'selected_epoch':ckpt['epoch'],'validation':ckpt['validation'],'positive_training_frames':len(train_indices),'pseudo_label_training_frames':sum(labels[i]['label_kind'].startswith('published')for i in train_indices),'validation_positive_frames':len(valid_indices),'validation_positive_crops':total_positive,'validation_negative_crops':len(valid_meta)-total_positive,'warmstart_sha256':sha(warm_path),'weights_sha256':sha(dest/'best.pt'),'training_labels_sha256':sha('/work/training-labels.json'),'protocol_sha256':sha(protocol_path),'seconds':time.monotonic()-started,'history':history,'mined_negative_crops':len(hard_examples),'limits':'Correlated crop-level development selection; not whole-frame or independent accuracy. Expanded model-reviewed anchors are pseudo labels.'}
    write(dest/'registrations.json',registrations);write(dest/'completed.json',result);commit();return {k:v for k,v in result.items()if k!='history'}

def infer_model(variant,scope,commit):
    import math,cv2,numpy as np,torch
    from functools import lru_cache
    from torch.nn import functional as F
    from temporal_model import TemporalBall,inputs
    from crop_verifier import SpatialBallVerifier,patch_inputs
    torch.set_num_threads(4);cv2.setNumThreads(1)
    protocol_path=protocol_file(variant)
    protocol=read(protocol_path);size=protocol['crop_size']
    weights=WARMSTART if variant=='rgb-original' else OUT/variant/'best.pt';ckpt=torch.load(weights,map_location='cpu',weights_only=False)
    model=TemporalBall(variant).cuda().eval();model.load_state_dict(ckpt['state_dict'])
    verifier_path=BASE/'ball-rebuild-20261026-v1/verifier-v4/best.pt'
    verifier=SpatialBallVerifier().cuda().eval();verifier.load_state_dict(torch.load(verifier_path,map_location='cpu',weights_only=False)['state_dict'])
    context=read('/work/context.json')['frames']
    if scope=='train':
        samples=[{**r,'key':r['clip']+'-'+str(r['frame']),'current_game':r['clip']=='nj-overhead-development'}for r in read('/work/training-labels.json')['rows']if r['split']=='train']
    elif scope=='regression':
        samples=[{**r,'key':r['clip']+'-'+str(r['frame']),'current_game':False}for r in read('/work/regression-labels.json')['rows']]
        samples += [{'key':'current-'+str(r['frame']),'frame':r['frame'],'clip':'nj-overhead','image':str(SOURCE/f"{r['frame']+1:06d}.jpg"),'current_game':True}for r in read('/work/reused-reference.json')['rows']]
    elif scope=='replay':
        samples=[{'key':'current-'+str(f),'frame':f,'clip':'nj-overhead','image':str(SOURCE/f'{f+1:06d}.jpg'),'current_game':True}for f in range(4500)]
    elif scope in ('fresh','fresh-review'):
        samples=[{'key':clip+'-'+str(f),'clip':clip,'frame':f,'image':str(BASE/'new-games-20261025-v1/frames'/clip/f'{f+1:06d}.jpg'),'current_game':False}for clip in protocol['fresh_ball_test']['clips']for f in range(1350)]
    elif scope in ('extra','extra-review'):
        samples=[{'key':clip+'-'+str(f),'clip':clip,'frame':f,'image':str(OUT/'extra-frames'/clip/f'{f+1:06d}.jpg'),'current_game':False}for clip in ['pittsburgh-finals-1200','southjersey-finals-1200']for f in range(1350)]
    elif scope=='final-check':
        samples=[{'key':clip+'-'+str(f),'clip':clip,'frame':f,'image':str(OUT/'extra-frames'/clip/f'{f+1:06d}.jpg'),'current_game':False}for clip in ['pittsburgh-finals-1800','southjersey-finals-1800']for f in range(23,1350,90)]
    else:raise ValueError(scope)
    if scope.endswith('-review'):
        samples=[r for r in samples if r['frame'] in protocol['fresh_ball_test']['sample_frames']]
    dest=OUT/variant/(scope+'.json');result=read(dest)if dest.exists()else{'variant':variant,'scope':scope,'weights_sha256':sha(weights),'frames':[]}
    assert result['weights_sha256']==sha(weights)
    done={r['key']for r in result['frames']};started=time.monotonic()
    @lru_cache(maxsize=12)
    def load(path):
        im=cv2.imread(path)
        if im is None:raise FileNotFoundError(path)
        return cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    def starts(n):return sorted(set(range(0,max(1,n-size),size-64))|{max(0,n-size)})
    for index,sample in enumerate(samples):
        if sample['key']in done:continue
        p=Path(sample['image']);center=load(str(p));h,w=center.shape[:2]
        sequence=[];quality=[]
        for offset in [-2,0,2]:
            if offset==0:sequence.append(center);quality.append(1.);continue
            im=load(str(neighbor(p,offset)));matrix,q=registration(im,center)
            sequence.append(cv2.warpAffine(im,matrix,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101));quality.append(q)
        sequence=np.stack(sequence)
        origins=[(x,y)for y in starts(h)for x in starts(w)]
        hint=context[sample['frame']].get('pixel')if sample['current_game']else None
        if hint:
            origin=(int(np.clip(round(hint[0]*w-size/2),0,w-size)),int(np.clip(round(hint[1]*h-size/2),0,h-size)))
            if origin not in origins:origins.append(origin)
        candidates=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for start in range(0,len(origins),32):
                batch=origins[start:start+32]
                tensor=torch.from_numpy(np.stack([sequence[:,y:y+size,x:x+size].transpose(0,3,1,2)for x,y in batch])).cuda().float()/255
                maps=model(inputs(tensor,variant)).float().sigmoid();peaks=(maps==F.max_pool2d(maps,5,1,2))&(maps>=.01)
                scores,indices=(maps*peaks).flatten(1).topk(3,dim=1)
                for (x,y),ss,ii in zip(batch,scores.cpu().tolist(),indices.cpu().tolist()):
                    for score,idx in zip(ss,ii):
                        if score<.01:continue
                        candidates.append({'pixel':[2*(idx%(size//2))+x,2*(idx//(size//2))+y],'score':score})
            kept=[]
            for c in sorted(candidates,key=lambda r:-r['score']):
                if not any(math.dist(c['pixel'],r['pixel'])<10 for r in kept):kept.append(c)
            # Retain global proposals and local-context proposals separately.
            retained=kept[:24]
            if hint:
                for c in [c for c in kept if math.dist(c['pixel'],[hint[0]*w,hint[1]*h])<=160][:6]:
                    if c not in retained:retained.append(c)
            if retained:
                crops=torch.from_numpy(np.stack([patch_inputs(center,r['pixel'])for r in retained])).cuda()
                scores=verifier(crops).float().sigmoid().cpu().tolist()
                for r,score in zip(retained,scores):r['appearance_score']=score
        result['frames'].append({'key':sample['key'],'clip':sample['clip'],'frame':sample['frame'],'width':w,'height':h,'source_sha256':sha(p),'registration_quality':quality,'candidates':retained})
        if index%100==0:
            result['seconds']=time.monotonic()-started;write(dest,result);commit()
            print({'variant':variant,'scope':scope,'sample':index,'total':len(samples),'seconds':round(result['seconds'])},flush=True)
    result.update(seconds=time.monotonic()-started,complete=True,verifier_sha256=sha(verifier_path),protocol_sha256=sha(protocol_path),code_sha256=sha(__file__))
    write(dest,result);commit();return {k:v for k,v in result.items()if k!='frames'}|{'frames':len(result['frames'])}


def infer_baseline(clip,commit):
    import cv2,numpy as np,torch
    from ultralytics import YOLO
    from crop_verifier import SpatialBallVerifier,patch_inputs
    torch.set_num_threads(4);cv2.setNumThreads(1)
    protocol=read('/work/protocol.json')
    extra_clips=['pittsburgh-finals-'+s for s in ['1200','1800']]+['southjersey-finals-'+s for s in ['1200','1800']]
    assert clip in protocol['fresh_ball_test']['clips']+extra_clips+['regression']
    source=(OUT/'extra-frames'/clip) if clip in extra_clips else (BASE/'new-games-20261025-v1/frames'/clip)
    dest=OUT/'yolo-full1920'/(clip+'.json')
    if dest.exists()and read(dest).get('complete'):return {'clip':clip,'cached':True}
    weights=BASE/'ball/training/yolo11m-crops640-s17-f10-e80-c6ab3e44/weights/best.pt';model=YOLO(str(weights))
    verifier_path=BASE/'ball-rebuild-20261026-v1/verifier-v4/best.pt';verifier=SpatialBallVerifier().cuda().eval()
    verifier.load_state_dict(torch.load(verifier_path,map_location='cpu',weights_only=False)['state_dict'])
    result={'variant':'yolo-full1920','clip':clip,'weights_sha256':sha(weights),'verifier_sha256':sha(verifier_path),'frames':[]};started=time.monotonic()
    if clip=='regression':
        samples=read('/work/regression-labels.json')['rows']
    else:
        samples=[{'frame':f,'clip':clip,'image':str(source/f'{f+1:06d}.jpg')} for f in (range(23,1350,90) if clip.endswith('1800') else range(1350))]
    for index,sample in enumerate(samples):
        f=sample['frame'];p=Path(sample['image']);im=cv2.imread(str(p));h,w=im.shape[:2]
        pred=model.predict(im,imgsz=1920,conf=.015,iou=.4,device=0,max_det=25,verbose=False)[0]
        candidates=[{'pixel':((box[:2]+box[2:4])/2).tolist(),'score':float(box[4])}for box in pred.boxes.data.cpu().numpy()]
        if candidates:
            x=torch.from_numpy(np.stack([patch_inputs(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),r['pixel'])for r in candidates])).cuda()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):scores=verifier(x).float().sigmoid().cpu().tolist()
            for r,s in zip(candidates,scores):r['appearance_score']=s
        result['frames'].append({'key':sample['clip']+'-'+str(f),'clip':sample['clip'],'frame':f,'width':w,'height':h,'source_sha256':sha(p),'candidates':candidates})
        if index%200==0:print({'baseline':clip,'frame':f,'seconds':round(time.monotonic()-started)},flush=True)
    result.update(complete=True,seconds=time.monotonic()-started,protocol_sha256=sha('/work/protocol.json'))
    write(dest,result);commit();return {k:v for k,v in result.items()if k!='frames'}|{'frames':len(result['frames'])}
