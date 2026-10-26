"""Re-run ball models on fixed cross-game labels and on the five-minute replay."""
from pathlib import Path
import modal
from modal_experiment import image, BASE, SOURCE, OUT, volume

HERE = Path(__file__).parent
app = modal.App('sport-scans-ball-alternatives-20261026')
runtime = (image.add_local_file(HERE/'modal_experiment.py', '/root/modal_experiment.py')
           .add_local_file(HERE/'heatmap_model.py', '/work/heatmap_model.py')
           .add_local_file(HERE/'regression-labels.json', '/work/labels.json'))


@app.function(image=runtime, gpu='H100', cpu=8, memory=32768, timeout=7200,
              max_containers=3, scaledown_window=10, volumes={'/data':volume})
def run(variant: str, dense: bool):
    import sys, json, hashlib, time, math
    import numpy as np, cv2, torch
    from torchvision.ops import nms
    from torch.nn import functional as F
    sys.path.insert(0, '/work')
    torch.set_num_threads(4); cv2.setNumThreads(1); volume.reload()
    dest=OUT/f'{variant}-{"dense" if dense else "regression"}.json'
    if dest.exists(): return {'variant':variant,'dense':dense,'cached':True}
    temporal=variant=='heatmap-temporal3'; heatmap=variant.startswith('heatmap')
    if heatmap:
        from heatmap_model import BallHeatmap
        weights=BASE/f'ball/heatmap-v1/resnet18-{"temporal3" if temporal else "rgb"}-heatmap-s17-e120-c6ab3e44/best.pt'
        ckpt=torch.load(weights,map_location='cpu',weights_only=False)
        model=BallHeatmap(temporal).cuda().eval(); model.load_state_dict(ckpt['state_dict'])
        mean=np.array([.485,.456,.406]*(3 if temporal else 1),np.float32)[:,None,None]
        std=np.array([.229,.224,.225]*(3 if temporal else 1),np.float32)[:,None,None]
    else:
        from ultralytics import YOLO
        weights=BASE/'ball/training/yolo11m-crops640-s17-f10-e80-c6ab3e44/weights/best.pt'
        model=YOLO(str(weights))
    if dense:
        samples=[{'frame':f,'image':str(SOURCE/'frames'/f'{f+1:06d}.jpg')} for f in range(4500)]
    else: samples=json.loads(Path('/work/labels.json').read_text())['rows']
    rows=[]; started=time.monotonic(); cache={}
    for idx,row in enumerate(samples):
        path=Path(row['image']); f=row['frame']
        if not dense: assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        def starts(n):return sorted(set(list(range(0,max(1,n-640),480))+[max(0,n-640)]))
        if heatmap:
            indices=[max(0,f-2),f,min(4499 if dense else 199,f+2)] if temporal else [f]
            frames=[]
            for j in indices:
                p=path.parent/(f'{j+1:06d}.jpg' if dense else f'frame-{j+1:06d}.jpg')
                if str(p) not in cache: cache[str(p)]=cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB)
                frames.append(cache[str(p)])
            h,w=frames[0].shape[:2];origins=[(x,y) for y in starts(h) for x in starts(w)]
            inputs=[]
            for x,y in origins:
                a=np.concatenate([im[y:y+640,x:x+640] for im in frames],axis=2)
                a=(a.transpose(2,0,1).astype(np.float32)/255-mean)/std
                inputs.append(torch.from_numpy(np.ascontiguousarray(a)))
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                prob=model(torch.stack(inputs).cuda()).float().sigmoid()
            peaks=(prob==F.max_pool2d(prob,7,1,3)) & (prob>=.015)
            candidates=[]
            for i,(x,y) in enumerate(origins):
                yy,xx=torch.where(peaks[i,0]); ss=prob[i,0,yy,xx]
                for j in torch.argsort(ss,descending=True)[:20].tolist():
                    cx=float(xx[j]*2+x);cy=float(yy[j]*2+y)
                    candidates.append([cx-5,cy-5,cx+5,cy+5,float(ss[j]),0])
            # A center-radius merge avoids multiple peaks from overlapping tiles.
            kept=[]
            for p in sorted(candidates,key=lambda p:-p[4]):
                if not any(math.dist(p[:2],q[:2])<10 for q in kept):kept.append(p)
            candidates=kept[:30]
            cache={str(path.parent/(f'{j+1:06d}.jpg' if dense else f'frame-{j+1:06d}.jpg')):im for j,im in zip(indices,frames)}
        else:
            im=cv2.imread(str(path))
            h,w=im.shape[:2];origins=[(x,y) for y in starts(h) for x in starts(w)]
            tiled=variant.startswith('tiles');size=int(variant[5:] if tiled else variant[4:])
            if not tiled:origins=[(0,0)]
            inputs=[im[y:y+640,x:x+640] for x,y in origins] if tiled else [im]
            preds=model.predict(inputs,imgsz=size,conf=.015,iou=.4,device=0,max_det=25,verbose=False)
            candidates=[]
            for p,(x,y) in zip(preds,origins):
                a=p.boxes.data.cpu().numpy()
                if len(a):a[:,:4]+=np.array([x,y,x,y]);candidates.extend(a.tolist())
            if candidates:
                a=torch.tensor(candidates);candidates=a[nms(a[:,:4],a[:,4],.35)[:30]].numpy().tolist()
        rows.append({'frame':f,'clip':row.get('clip'),'detections':candidates})
        if idx%300==0: print({'variant':variant,'dense':dense,'frame':idx,'seconds':round(time.monotonic()-started)},flush=True)
    report={'variant':variant,'dense':dense,'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
            'seconds':time.monotonic()-started,'frames':rows,'frame_offsets':[-2,0,2] if temporal else [0]}
    dest.write_text(json.dumps(report,separators=(',',':'))+'\n');volume.commit()
    return {k:v for k,v in report.items() if k!='frames'}


@app.local_entrypoint()
def main(dense:bool=False):
    jobs=[(v,dense) for v in (['heatmap-rgb','heatmap-temporal3'] if dense else
                              ['full1280','full1920','tiles640','tiles960','heatmap-rgb','heatmap-temporal3'])]
    for result in run.starmap(jobs,order_outputs=False):print(result,flush=True)
