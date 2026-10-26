"""Learned point tracking across native NBHL sequences, forward and backward."""
from pathlib import Path
import modal

HERE=Path(__file__).parent;ROOT=HERE.parents[1] if modal.is_local() else Path('/work')
app=modal.App('sport-scans-ball-point-sequence-20261026')
volume=modal.Volume.from_name('sport-scans-experiments')
image=(modal.Image.debian_slim(python_version='3.12').apt_install('git')
       .pip_install('torch==2.9.1','torchvision==0.24.1',index_url='https://download.pytorch.org/whl/cu128')
       .pip_install('einops==0.8.1','numpy==1.26.4','opencv-python-headless==4.11.0.86','pillow==11.3.0','tqdm','omegaconf')
       .pip_install('git+https://github.com/facebookresearch/co-tracker.git@82e02e8029753ad4ef13cf06be7f4fc5facdda4d','git+https://github.com/google-deepmind/tapnet.git@730cda1c730877cfedbe01bf87fb1cadb78a565d',extra_options='--no-deps')
       .add_local_file(HERE/'point-protocol.json','/work/point-protocol.json')
       .add_local_file(ROOT/'web/data/overhead-20261025/replay.json','/work/players.json')
       .add_local_file(ROOT/'web/data/ball-rebuild-20261026/replay.json','/work/anchors.json'))


@app.function(image=image,cpu=8,memory=32768,gpu='H100',timeout=7200,max_containers=2,scaledown_window=10,volumes={'/data':volume})
def run(model_name:str,all_seeds:bool=False):
    import json,time,hashlib,urllib.request
    import numpy as np,torch,cv2
    volume.reload();torch.set_num_threads(4);cv2.setNumThreads(2)
    protocol=json.loads(Path('/work/point-protocol.json').read_text());replay=json.loads(Path('/work/players.json').read_text())
    anchors=json.loads(Path('/work/anchors.json').read_text())
    assert hashlib.sha256(Path('/work/anchors.json').read_bytes()).hexdigest()==protocol['baseline_sha256']
    base=Path('/data/ball-hockey/reliability-20261018');source=base/'overhead-20261025-v1/frames'
    out=base/'ball-sequence-20261026-v1/points';out.mkdir(parents=True,exist_ok=True)
    checkpoints=Path('/data/cache/ball-point-trackers');checkpoints.mkdir(parents=True,exist_ok=True)
    if model_name=='cotracker3':
        ckpt=checkpoints/'scaled_offline.pth';url='https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth'
    elif model_name=='tapnextpp512':
        ckpt=checkpoints/'tapnextpp_512.ckpt';url='https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt'
    else:raise ValueError(model_name)
    if not ckpt.exists():
        temp=ckpt.with_suffix('.download');urllib.request.urlretrieve(url,temp);temp.replace(ckpt);volume.commit()
    if model_name=='cotracker3':
        from cotracker.predictor import CoTrackerPredictor
        model=CoTrackerPredictor(checkpoint=str(ckpt),offline=True).cuda().eval()
    else:
        from tapnet.tapnextpp.votsp2026.model import TAPNextPP
        model=TAPNextPP.from_checkpoint(ckpt,device='cuda',input_resolution=512)
    seeds=protocol['pilot_seeds']
    if all_seeds:
        seeds=[]
        for row in anchors['frames']:
            if row['state']=='detected' and (not seeds or row['frame']-seeds[-1]['frame']>=30):
                seeds.append({'frame':row['frame'],'pixel':row['pixel'],'evidence_score':row['evidence_score']})
    dest=out/(model_name+('-all'if all_seeds else'-pilot')+'.json')
    result=json.loads(dest.read_text())if dest.exists()else{'model':model_name,'protocol':protocol,'weights_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),'tracks':[]}
    done={r['seed_frame']for r in result['tracks']};started=time.monotonic()
    for seed in seeds:
        sf=seed['frame']
        if sf in done:continue
        start=max(0,sf-90);stop=min(4500,sf+91)
        if sf<466:stop=min(stop,466)
        elif sf>=516:start=max(start,516)
        else:continue
        point=np.array(seed['pixel'])*[1920,1080];H0=replay['frames'][sf].get('image_to_floor')
        origins=[];crops=[];previous=point
        for f in range(start,stop):
            H=replay['frames'][f].get('image_to_floor');center=previous
            if H0 is not None and H is not None:
                try:
                    q=np.linalg.inv(np.array(H))@np.array(H0)@np.r_[point,1];candidate=q[:2]/q[2]
                    if np.isfinite(candidate).all() and np.linalg.norm(candidate-point)<2200:center=candidate
                except np.linalg.LinAlgError:pass
            previous=center;origin=center-[384,288];origins.append(origin)
            frame=cv2.imread(str(source/f'{f+1:06d}.jpg'))
            crop=cv2.warpAffine(frame,np.array([[1,0,-origin[0]],[0,1,-origin[1]]],np.float32),(768,576),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(96,96,96))
            crops.append(crop)
        index=sf-start;qxy=point-origins[index];positions=np.zeros((len(crops),2));visible=np.zeros(len(crops),bool)
        inference_start=time.monotonic()
        if model_name=='cotracker3':
            video=torch.from_numpy(np.stack(crops)[...,::-1].copy()).permute(0,3,1,2)[None].float().cuda()
            query=torch.tensor([[[index,*qxy]]],dtype=torch.float32,device='cuda')
            with torch.inference_mode():pred,vis=model(video,queries=query,backward_tracking=True)
            positions=pred[0,:,0].cpu().numpy();visible=vis[0,:,0].cpu().numpy();del video
        else:
            for direction in [1,-1]:
                state=None
                for i in range(index,len(crops)if direction==1 else-1,direction):
                    pos,vis,state=model.track_frame(crops[i],query_points_xy=qxy[None]if state is None else None,state=state)
                    positions[i]=pos[0];visible[i]=vis[0]
        rows=[]
        for i,(p,v,origin)in enumerate(zip(positions,visible,origins)):
            native=p+origin
            rows.append({'frame':start+i,'pixel':native.tolist(),'visible':bool(v),'inside_crop':bool(np.all(p>[12,12])and np.all(p<[756,564])),'inside_image':bool(np.all(native>=0)and np.all(native<[1920,1080]))})
        result['tracks'].append({'seed_frame':sf,'seed_pixel':point.tolist(),'interval':[start,stop],'inference_seconds':time.monotonic()-inference_start,'rows':rows})
        result['elapsed_seconds']=time.monotonic()-started
        temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(result,separators=(',',':'))+'\n');temp.replace(dest);volume.commit()
        print({'model':model_name,'seed':sf,'seconds':round(result['tracks'][-1]['inference_seconds']),'visible':sum(r['visible']and r['inside_crop']for r in rows)},flush=True)
        torch.cuda.empty_cache()
    return {'model':model_name,'seeds':len(result['tracks']),'seconds':result.get('elapsed_seconds',0)}


@app.function(image=image.add_local_file(ROOT/'experiments/ball-rebuild-20261026/crop_verifier.py','/work/crop_verifier.py'),
              cpu=8,memory=16384,gpu='H100',timeout=3600,scaledown_window=10,volumes={'/data':volume})
def verify_points():
    import sys,json,time,hashlib
    import numpy as np,torch,cv2
    sys.path.insert(0,'/work');from crop_verifier import SpatialBallVerifier,patch_inputs
    volume.reload();torch.set_num_threads(4);cv2.setNumThreads(2)
    base=Path('/data/ball-hockey/reliability-20261018');out=base/'ball-sequence-20261026-v1/points'
    source=base/'overhead-20261025-v1/frames';path=out/'tapnextpp512-all.json'
    tracks=json.loads(path.read_text());weights=base/'ball-rebuild-20261026-v1/verifier-v4/best.pt'
    checkpoint=torch.load(weights,map_location='cpu',weights_only=False)
    model=SpatialBallVerifier().cuda().eval();model.load_state_dict(checkpoint['state_dict'])
    rows=[{'frame':f,'proposals':[]}for f in range(4500)]
    for track in tracks['tracks']:
        for row in track['rows']:
            if row['inside_crop']and row['inside_image']:
                rows[row['frame']]['proposals'].append({**row,'seed_frame':track['seed_frame']})
    started=time.monotonic()
    for row in rows:
        if not row['proposals']:continue
        im=cv2.cvtColor(cv2.imread(str(source/f"{row['frame']+1:06d}.jpg")),cv2.COLOR_BGR2RGB)
        x=torch.from_numpy(np.stack([patch_inputs(im,p['pixel'])for p in row['proposals']])).cuda()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):prob=model(x).float().sigmoid().cpu().tolist()
        for proposal,score in zip(row['proposals'],prob):proposal['appearance_score']=score
    result={'frames':rows,'tracker_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'verifier_weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'seconds':time.monotonic()-started}
    (out/'tapnextpp512-verified.json').write_text(json.dumps(result,separators=(',',':'))+'\n');volume.commit()
    return {'seconds':result['seconds'],'proposals':sum(len(r['proposals'])for r in rows)}


@app.local_entrypoint()
def main(all_seeds:bool=False,models:str='cotracker3,tapnextpp512',verify:bool=False):
    if verify:print(verify_points.remote())
    else:print(list(run.starmap([(name,all_seeds)for name in models.split(',')],order_outputs=False)))
