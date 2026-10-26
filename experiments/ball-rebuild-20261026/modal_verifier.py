"""Hard-negative ball verification trained on reviewed NBHL frames only."""
from pathlib import Path
import modal
from modal_experiment import image,BASE,SOURCE,OUT,volume
HERE=Path(__file__).parent;ROOT=HERE.parents[1] if modal.is_local() else Path('/work')
app=modal.App('sport-scans-ball-verifier-20261026')
runtime=(image.env({'TORCH_HOME':'/data/cache/torch'})
    .add_local_file(HERE/'modal_experiment.py','/root/modal_experiment.py')
    .add_local_file(HERE/'crop_verifier.py','/work/crop_verifier.py')
    .add_local_file(HERE/'training-labels.json','/work/training-labels.json')
    .add_local_file(HERE/'verifier-adaptation-labels.json','/work/adaptation.json')
    .add_local_file(HERE/'regression-labels.json','/work/regression-labels.json')
    .add_local_file(HERE/'development-audit.json','/work/audit.json')
    .add_local_file(ROOT/'hockey_tracking/ball_path.py','/work/ball_path.py'))


@app.function(image=runtime,gpu='H100',cpu=8,memory=32768,timeout=3600,scaledown_window=10,volumes={'/data':volume})
def train(adapted:bool=False,hard_mining:bool=False,spatial:bool=False):
    import sys,json,hashlib,random,time,math,cv2,numpy as np,torch
    from ultralytics import YOLO
    from torch.utils.data import Dataset,DataLoader
    from torch.nn import functional as F
    sys.path.insert(0,'/work');from crop_verifier import BallVerifier,SpatialBallVerifier,patch_inputs
    volume.reload();torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(17);random.seed(17);np.random.seed(17)
    version=4 if spatial else 3 if hard_mining else 2 if adapted else 1
    out=OUT/f'verifier-v{version}';out.mkdir(exist_ok=True)
    if (out/'completed.json').exists():return json.loads((out/'completed.json').read_text())
    labels=json.loads(Path('/work/training-labels.json').read_text())['rows']
    if adapted:labels+=json.loads(Path('/work/adaptation.json').read_text())['rows']
    weights=BASE/'ball/training/yolo11m-crops640-s17-f10-e80-c6ab3e44/weights/best.pt';detector=YOLO(str(weights))
    cache={};samples={'train':[],'validation':[]};mining=[]
    for row in labels:
        path=row['image'];fingerprint=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if row.get('sha256'):assert fingerprint==row['sha256']
        bgr=cv2.imread(path);im=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);cache[path]=im;split=row['split'];point=row['ball']['center']
        samples[split].append({'image':path,'point':point,'label':1,'kind':'reviewed_ball'})
        origins=[(x,y)for y in [0,440]for x in [0,480,960,1280]]
        preds=detector.predict([bgr[y:y+640,x:x+640]for x,y in origins],imgsz=640,conf=.02,device=0,half=True,max_det=20,verbose=False)
        negatives=[]
        for pred,(x,y) in zip(preds,origins):
            for box in pred.boxes.data.cpu().numpy():
                p=((box[:2]+box[2:4])/2+np.array([x,y])).tolist()
                if math.dist(p,point)>48 and not any(math.dist(p,q)<15 for q in negatives):negatives.append(p)
        rng=random.Random(int(fingerprint[:12],16))
        for _ in range(5):
            p=[rng.randint(16,im.shape[1]-17),rng.randint(16,im.shape[0]-17)]
            if math.dist(p,point)>150:negatives.append(p)
        samples[split].extend({'image':path,'point':p,'label':0,'kind':'known_ball_elsewhere'}for p in negatives)
        if hard_mining and split=='train':
            # Mine lookalikes independently of the already overfit detector:
            # warm-colored components include tape, shoes, jerseys and markings.
            rgb=im.astype(np.int16);warm=((rgb[:,:,0]-rgb[:,:,1]>8)&(rgb[:,:,0]-rgb[:,:,2]>-12)).astype(np.uint8)
            _,_,stats,centroids=cv2.connectedComponentsWithStats(warm,8)
            hard=[]
            for stat,point2 in zip(stats[1:],centroids[1:]):
                left,top,w,h,area=stat
                if 3<=area<=12000 and .1<=w/max(h,1)<=10 and math.dist(point2,point)>96:
                    hard.append(point2.tolist())
            rng.shuffle(hard)
            samples[split].extend({'image':path,'point':p,'label':0,'kind':'appearance_hard_negative'}for p in hard[:48])
        mining.append({'image_sha256':fingerprint,'split':split,'negative_centers':negatives})
    # Only unequivocal development false objects; no reserved audit frames.
    hard_frames={49,71,98,111,356,1047,1222,1265,1288,1299,1322}
    for r in json.loads(Path('/work/audit.json').read_text())['rows']:
        if r['frame'] not in hard_frames:continue
        path=str(SOURCE/'frames'/f"{r['frame']+1:06d}.jpg")
        cache[path]=cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
        samples['train'].append({'image':path,'point':r['pixel'],'label':0,'kind':'reviewed_hard_negative'})
    del detector;torch.cuda.empty_cache()
    train_pos=[r for r in samples['train'] if r['label']];train_neg=[r for r in samples['train'] if not r['label']]
    reviewed_neg=[r for r in train_neg if r['kind']=='reviewed_hard_negative']
    appearance_neg=[r for r in train_neg if r['kind']=='appearance_hard_negative']
    class Training(Dataset):
        def __len__(self):return len(train_pos)*24
        def __getitem__(self,index):
            pool=train_pos if index%2 else train_neg
            if hard_mining and not index%2:
                draw=random.random();pool=reviewed_neg if draw<.2 else appearance_neg if draw<.8 and appearance_neg else train_neg
            row=random.choice(pool)
            a=patch_inputs(cache[row['image']],row['point'],np.random.uniform(-3,3,2))
            if random.random()<.5:a=a[:,:,::-1].copy()
            a=a*random.uniform(.9,1.1)+random.uniform(-.1,.1)
            return torch.from_numpy(np.ascontiguousarray(a)),torch.tensor(float(row['label']))
    loader=DataLoader(Training(),batch_size=64,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True)
    validation=torch.from_numpy(np.stack([patch_inputs(cache[r['image']],r['point'])for r in samples['validation']]))
    targets=np.array([r['label'] for r in samples['validation']]);model=(SpatialBallVerifier(True) if spatial else BallVerifier(True)).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.001);scaler=torch.amp.GradScaler('cuda')
    best=-1;history=[];started=time.monotonic()
    for epoch in range(20):
        model.train();losses=[]
        for x,y in loader:
            x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):loss=F.binary_cross_entropy_with_logits(model(x),y)
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();losses.append(float(loss.detach()))
        model.eval();prob=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            for x in validation.split(128):prob.extend(model(x.cuda()).float().sigmoid().cpu().tolist())
        prob=np.array(prob);options=[]
        for threshold in [.1,.2,.3,.4,.5,.6,.7,.8,.9]:
            pred=prob>=threshold;tp=int((pred&(targets==1)).sum());fp=int((pred&(targets==0)).sum());fn=int((~pred&(targets==1)).sum())
            options.append({'threshold':threshold,'tp':tp,'fp':fp,'fn':fn,'f1':2*tp/(2*tp+fp+fn)})
        operating=max(options,key=lambda r:(r['f1'],-r['fp'],r['threshold']))
        row={'epoch':epoch+1,'loss':sum(losses)/len(losses),'validation':operating,'seconds':time.monotonic()-started};history.append(row);print(row,flush=True)
        if operating['f1']>best:
            best=operating['f1'];torch.save({'state_dict':model.state_dict(),'epoch':epoch+1,'operating_point':operating,'spatial':spatial},out/'best.pt')
    checkpoint=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    report={'adapted':adapted,'hard_mining':hard_mining,'spatial':spatial,'adaptation_labels_sha256':hashlib.sha256(Path('/work/adaptation.json').read_bytes()).hexdigest() if adapted else None,'training_positive_frames':len(train_pos),'training_negative_crops':len(train_neg),'validation_positive_frames':int(targets.sum()),
            'validation_negative_crops':int((targets==0).sum()),'selected_epoch':checkpoint['epoch'],'operating_point':checkpoint['operating_point'],
            'weights_sha256':hashlib.sha256((out/'best.pt').read_bytes()).hexdigest(),'history':history,
            'labels_sha256':hashlib.sha256(Path('/work/training-labels.json').read_bytes()).hexdigest(),
            'limits':'Correlated pilot. Crop-verification validation is not full-frame detection accuracy. Hard negatives are mined only where a visible active ball has a known different location, plus 11 reviewed development false objects.'}
    (out/'training-samples.json').write_text(json.dumps(samples,separators=(',',':'))+'\n');(out/'mining.json').write_text(json.dumps(mining,separators=(',',':'))+'\n')
    (out/'completed.json').write_text(json.dumps(report,indent=2)+'\n');volume.commit();return report


@app.function(image=runtime,gpu='H100',cpu=8,memory=32768,timeout=3600,scaledown_window=10,volumes={'/data':volume})
def infer(dense:bool=True,adapted:bool=False,hard_mining:bool=False,spatial:bool=False):
    import sys,json,time,cv2,numpy as np,torch,hashlib
    sys.path.insert(0,'/work');from crop_verifier import BallVerifier,SpatialBallVerifier,patch_inputs
    from ball_path import fuse_candidates
    volume.reload();torch.set_num_threads(4);cv2.setNumThreads(1)
    version=4 if spatial else 3 if hard_mining else 2 if adapted else 1
    trained=OUT/f'verifier-v{version}';checkpoint=torch.load(trained/'best.pt',map_location='cpu',weights_only=False)
    model=(SpatialBallVerifier() if checkpoint.get('spatial') else BallVerifier()).cuda().eval();model.load_state_dict(checkpoint['state_dict'])
    suffix='' if dense else '-regression'
    inputs=[json.loads((OUT/f'{s}{suffix}.json').read_text())['frames']for s in ['full1280','full1920','tiles640']]
    fused=fuse_candidates(inputs);labels=json.loads(Path('/work/regression-labels.json').read_text())['rows'] if not dense else None
    rows=[];started=time.monotonic()
    for f,row in enumerate(fused):
        path=SOURCE/'frames'/f'{f+1:06d}.jpg' if dense else Path(labels[f]['image'])
        im=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB);boxes=row['detections'];prob=[]
        if boxes:
            x=torch.from_numpy(np.stack([patch_inputs(im,(np.array(b[:2])+b[2:4])/2)for b in boxes])).cuda()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):prob=model(x).float().sigmoid().cpu().tolist()
        rows.append({'frame':f,'probabilities':prob})
        if f%500==0:print({'dense':dense,'frame':f,'seconds':round(time.monotonic()-started)},flush=True)
    result={'frames':rows,'weights_sha256':hashlib.sha256((trained/'best.pt').read_bytes()).hexdigest(),'seconds':time.monotonic()-started,
            'operating_point':checkpoint['operating_point']}
    (OUT/f'verifier{f"-v{version}" if version>1 else ""}{suffix}.json').write_text(json.dumps(result,separators=(',',':'))+'\n');volume.commit()
    return {k:v for k,v in result.items() if k!='frames'}


@app.local_entrypoint()
def main(task:str='train',adapted:bool=False,hard_mining:bool=False,spatial:bool=False):
    if hard_mining and not adapted:raise ValueError('Hard-mining versions require --adapted')
    if spatial and not hard_mining:raise ValueError('The spatial version requires --adapted --hard-mining')
    if task=='train':print(train.remote(adapted,hard_mining,spatial))
    else:
        for result in infer.starmap([(False,adapted,hard_mining,spatial),(True,adapted,hard_mining,spatial)],order_outputs=False):print(result)
