"""Inspect candidate neighborhoods for extended red painted markings."""
from pathlib import Path
import modal
HERE=Path(__file__).parent
ROOT=HERE.parents[1] if modal.is_local() else Path('/work')
app=modal.App('sport-scans-ball-appearance-20261026')
volume=modal.Volume.from_name('sport-scans-experiments')
image=(modal.Image.debian_slim(python_version='3.12').pip_install('numpy==1.26.4','opencv-python-headless==4.11.0.86')
       .add_local_file(ROOT/'hockey_tracking/ball_path.py','/work/ball_path.py'))


@app.function(image=image,cpu=8,memory=16384,timeout=900,volumes={'/data':volume})
def run(shape:bool=False):
    import sys,json,cv2,numpy as np,time
    sys.path.insert(0,'/work');from ball_path import fuse_candidates
    base=Path('/data/ball-hockey/reliability-20261018');out=base/'ball-rebuild-20261026-v1';volume.reload();cv2.setNumThreads(2)
    streams=[json.loads((out/f'{s}.json').read_text())['frames'] for s in ['full1280','full1920','tiles640']]
    candidates=fuse_candidates(streams);rows=[];started=time.monotonic()
    previous=json.loads((out/'appearance.json').read_text())['frames'] if shape else None
    verification=json.loads((out/'verifier-v4.json').read_text())['frames'] if shape else None
    for f,row in enumerate(candidates):
        im=cv2.imread(str(base/'overhead-20261025-v1/frames'/f'{f+1:06d}.jpg'));features=[]
        for j,box in enumerate(row['detections']):
            if shape and verification[f]['probabilities'][j]<.01:
                features.append({**previous[f]['features'][j],'elongated_structure':False,'components':[]});continue
            x,y=np.round((np.array(box[:2])+box[2:4])/2).astype(int)
            patch=im[max(0,y-32):min(1080,y+33),max(0,x-32):min(1920,x+33)].astype(np.float32)
            if not patch.size:features.append({'extended_marking':False});continue
            b,g,r=patch.transpose(2,0,1);warm=((r-g>8)&(r-b>-12)).astype(np.uint8)
            n,labels,stats,_=cv2.connectedComponentsWithStats(warm,8)
            cy=y-max(0,y-32);cx=x-max(0,x-32)
            near=labels[max(0,cy-4):cy+5,max(0,cx-4):cx+5];ids=set(near.ravel())-{0}
            extended=False;components=[]
            for k in ids:
                left,top,w,h,area=stats[k]
                vertical=top<=1 and top+h>=patch.shape[0]-1
                horizontal=left<=1 and left+w>=patch.shape[1]-1
                if area>=200 and (vertical or horizontal):extended=True
                if shape and 12<=area<=2000:
                    points=np.argwhere(labels==k)[:,::-1].astype(float)
                    centered=points-points.mean(axis=0);values,vectors=np.linalg.eigh(centered.T@centered/len(points))
                    distance=float(np.linalg.norm(points-[cx,cy],axis=1).min())
                    components.append({'area':int(area),'major_sigma':float(np.sqrt(values[1])),
                                       'axis_ratio':float(np.sqrt(values[1]/max(.1,values[0]))),'distance':distance})
            feature={'extended_marking':bool(extended)}
            if shape:
                feature.update(components=components,elongated_structure=any(c['major_sigma']>=5 and c['axis_ratio']>=3 and c['distance']<=3 for c in components))
            features.append(feature)
        rows.append({'frame':f,'features':features})
    result={'sources':['full1280','full1920','tiles640'],'frames':rows,'seconds':time.monotonic()-started,
            'limits':'Red-marking prior for this rink; can withhold a ball directly on a red line. Not a universal ball-color classifier.'}
    name='appearance-shape.json' if shape else 'appearance.json'
    if shape:result['limits']+=' Elongated-color rejection is an uncalibrated shape prior and may withhold a severely motion-blurred ball.'
    (out/name).write_text(json.dumps(result,separators=(',',':'))+'\n');volume.commit()
    return {'seconds':result['seconds'],'flagged':sum(f['extended_marking'] for r in rows for f in r['features']),
            'elongated':sum(f.get('elongated_structure',False) for r in rows for f in r['features'])}


@app.local_entrypoint()
def main(shape:bool=False):print(run.remote(shape))
