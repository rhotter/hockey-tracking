"""Native-resolution NBHL ball detector ablations on the existing Modal volume."""
from pathlib import Path
import modal

ROOT=Path(__file__).resolve().parents[2] if modal.is_local() else Path('/work')
BASE=Path('/data/ball-hockey/reliability-20261018')
SOURCE=BASE/'overhead-20261025-v1'
OUT=BASE/'ball-rebuild-20261026-v1'
app=modal.App('sport-scans-ball-rebuild-20261026')
volume=modal.Volume.from_name('sport-scans-experiments')
image = (modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.9.1", "torchvision==0.24.1", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("ultralytics==8.4.181", "numpy==1.26.4", "opencv-python==4.11.0.86", "scipy", "pillow")
    .run_commands("python -m pip uninstall -y opencv-python", "python -m pip install --no-deps --force-reinstall opencv-python-headless==4.11.0.86")
    .env({"YOLO_CONFIG_DIR": "/data/cache/ultralytics", "OMP_NUM_THREADS": "4"}))


@app.function(image=image,cpu=8,memory=32768,gpu='H100',timeout=7200,max_containers=3,
              scaledown_window=10,volumes={'/data':volume})
def detect(mode: str):
    import cv2,json,time,hashlib,torch,numpy as np
    from ultralytics import YOLO
    from torchvision.ops import nms
    assert mode in ['full1280','full1920','tiles640','tiles960']
    volume.reload();OUT.mkdir(exist_ok=True);torch.set_num_threads(4);cv2.setNumThreads(2)
    weights=BASE/'ball/training/yolo11m-crops640-s17-f10-e80-c6ab3e44/weights/best.pt'
    model=YOLO(str(weights));cls=next(k for k,v in model.names.items() if v=='active_ball')
    dest=OUT/(mode+'.json')
    if dest.exists():return {'mode':mode,'cached':True}
    rows=[];started=time.monotonic();files=sorted((SOURCE/'frames').glob('*.jpg'))
    assert len(files)==4500
    for f,path in enumerate(files):
        im=cv2.imread(str(path));h,w=im.shape[:2]
        if mode.startswith('tiles'):
            origins=[(x,y)for y in [0,440]for x in [0,480,960,1280]]
            inputs=[im[y:y+640,x:x+640] for x,y in origins];size=int(mode[5:])
        else:origins=[(0,0)];inputs=[im];size=int(mode[4:])
        preds=model.predict(inputs,imgsz=size,conf=.015,iou=.4,classes=[cls],device=0,max_det=25,verbose=False)
        candidates=[]
        for pred,(x,y) in zip(preds,origins):
            a=pred.boxes.data.cpu().numpy()
            if len(a):a[:,:4]+=np.array([x,y,x,y]);candidates.extend(a.tolist())
        if candidates:
            a=torch.tensor(candidates);keep=nms(a[:,:4],a[:,4],.35);candidates=a[keep[:30]].numpy().tolist()
        rows.append({'frame':f,'detections':candidates})
        if f%300==0:print({'mode':mode,'frame':f,'seconds':round(time.monotonic()-started)},flush=True)
    report={'mode':mode,'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
            'native_size':[1920,1080],'fps':15,'frames':rows,'seconds':time.monotonic()-started}
    dest.write_text(json.dumps(report,separators=(',',':'))+'\n');volume.commit()
    return {'mode':mode,'seconds':report['seconds'],'candidate_frames':sum(bool(r['detections'])for r in rows),
            'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}


@app.function(image=modal.Image.debian_slim(python_version="3.12").pip_install("pillow==11.3.0"),cpu=2,memory=4096,timeout=300,volumes={'/data':volume})
def review():
    import json,hashlib,tarfile
    from PIL import Image,ImageDraw
    volume.reload();dest=OUT/'review';dest.mkdir(parents=True,exist_ok=True)
    frames=list(range(37,4500,75));manifest=[]
    for f in frames:
        src=SOURCE/'frames'/f'{f+1:06d}.jpg';im=Image.open(src).convert('RGB')
        im.save(dest/f'frame-{f:04d}.jpg',quality=95)
        # Full-width floor view for first-pass inspection, not a ball proposal.
        crop=im.crop((0,270,1920,1080)).resize((1280,540))
        d=ImageDraw.Draw(crop);d.rectangle((0,0,260,24),fill='white')
        d.text((5,4),f'Frame {f} / native origin 0,270 / scale 2/3',fill='black')
        crop.save(dest/f'floor-{f:04d}.jpg',quality=95)
        manifest.append({'frame':f,'sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
                         'split':'development' if f<1500 else 'test'})
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with tarfile.open(OUT/'review.tar.gz','w:gz')as tar:tar.add(dest,arcname='review')
    volume.commit();return {'frames':len(frames),'archive':str(OUT/'review.tar.gz')}


@app.function(image=image,cpu=8,memory=32768,gpu='H100',timeout=3600,
              scaledown_window=10,volumes={'/data':volume})
def detect_batched():
    import cv2,json,time,hashlib,torch
    from ultralytics import YOLO
    volume.reload();torch.set_num_threads(4);cv2.setNumThreads(2)
    dest=OUT/'full1920-fast.json'
    if dest.exists():return {'cached':True}
    weights=BASE/'ball/training/yolo11m-crops640-s17-f10-e80-c6ab3e44/weights/best.pt'
    model=YOLO(str(weights));files=sorted((SOURCE/'frames').glob('*.jpg'));rows=[];started=time.monotonic()
    for offset in range(0,len(files),8):
        inputs=[cv2.imread(str(p)) for p in files[offset:offset+8]]
        predictions=model.predict(inputs,imgsz=1920,conf=.015,iou=.4,device=0,max_det=25,verbose=False,half=True)
        rows.extend({'frame':offset+i,'detections':p.boxes.data.cpu().numpy().tolist()} for i,p in enumerate(predictions))
        if offset%400==0:print({'mode':'full1920-fast','frame':offset,'seconds':round(time.monotonic()-started)},flush=True)
    result={'mode':'full1920-fast','weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
            'native_size':[1920,1080],'fps':15,'half':True,'batch_size':8,'frames':rows,'seconds':time.monotonic()-started}
    dest.write_text(json.dumps(result,separators=(',',':'))+'\n');volume.commit()
    return {k:v for k,v in result.items() if k!='frames'}


@app.local_entrypoint()
def main(task:str='detect',modes:str='full1280,full1920,tiles640'):
    import json
    local=ROOT/'build/ball-rebuild-20261026';local.mkdir(parents=True,exist_ok=True)
    if task=='batched':results=[detect_batched.remote()]
    elif task=='review':results=[review.remote()]
    else:results=list(detect.map(modes.split(','),order_outputs=False))
    (local/(task+'-result.json')).write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
