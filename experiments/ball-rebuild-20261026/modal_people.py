"""All-person pose context, including officials and spectators excluded by player tracking."""
from pathlib import Path
import modal
from modal_experiment import image, BASE, SOURCE, OUT, volume

app=modal.App('sport-scans-ball-person-context-20261026')
runtime=image.add_local_file(Path(__file__).parent/'modal_experiment.py','/root/modal_experiment.py')


@app.function(image=runtime,gpu='H100',cpu=8,memory=32768,timeout=3600,
              scaledown_window=10,volumes={'/data':volume})
def run(tiled:bool=False):
    import cv2,json,time,torch,hashlib,os,numpy as np
    from torchvision.ops import nms
    from ultralytics import YOLO
    volume.reload();torch.set_num_threads(4);cv2.setNumThreads(2)
    dest=OUT/('people-pose-tiled.json' if tiled else 'people-pose.json')
    if dest.exists():return {'cached':True}
    cache=Path('/data/cache/ball-pose');cache.mkdir(parents=True,exist_ok=True);os.chdir(cache)
    model=YOLO('yolo11m-pose.pt');started=time.monotonic();rows=[]
    files=sorted((SOURCE/'frames').glob('*.jpg'))
    step=1 if tiled else 8
    prior=json.loads((OUT/'people-pose.json').read_text())['frames'] if tiled else None
    for offset in range(0,len(files),step):
        if tiled:
            im=cv2.imread(str(files[offset]));origins=[(x,y) for y in [0,440] for x in [0,480,960,1280]]
            preds=model.predict([im[y:y+640,x:x+640] for x,y in origins],imgsz=640,conf=.1,iou=.5,device=0,half=True,max_det=30,verbose=False)
            boxes=list(prior[offset]['boxes']);joints=list(prior[offset]['keypoints'])
            for pred,(x,y) in zip(preds,origins):
                b=pred.boxes.data.cpu().numpy();k=pred.keypoints.data.cpu().numpy()
                if len(b):
                    b[:,:4]+=np.array([x,y,x,y]);k[:,:,:2]+=np.array([x,y]);boxes.extend(b.tolist());joints.extend(k.tolist())
            keep=nms(torch.tensor(boxes)[:,:4],torch.tensor(boxes)[:,4],.5).tolist() if boxes else []
            rows.append({'frame':offset,'boxes':[boxes[j] for j in keep],'keypoints':[joints[j] for j in keep]})
            if offset%400==0:print({'frame':offset,'seconds':round(time.monotonic()-started),'people':len(keep)},flush=True)
            continue
        images=[cv2.imread(str(p)) for p in files[offset:offset+8]]
        preds=model.predict(images,imgsz=960,conf=.15,iou=.5,device=0,half=True,max_det=60,verbose=False)
        for i,p in enumerate(preds):
            rows.append({'frame':offset+i,'boxes':p.boxes.data.cpu().numpy().tolist(),
                         'keypoints':p.keypoints.data.cpu().numpy().tolist()})
        if offset%400==0:print({'frame':offset,'seconds':round(time.monotonic()-started)},flush=True)
    result={'model':'yolo11m-pose, generic COCO person/pose weights','weights_sha256':hashlib.sha256(Path('yolo11m-pose.pt').read_bytes()).hexdigest(),
            'seconds':time.monotonic()-started,'frames':rows}
    dest.write_text(json.dumps(result,separators=(',',':'))+'\n');volume.commit()
    return {k:v for k,v in result.items() if k!='frames'}


@app.local_entrypoint()
def main(tiled:bool=False):print(run.remote(tiled))
