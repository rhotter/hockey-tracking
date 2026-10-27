"""Native-resolution short tracks; visual evidence and propagation stay distinct.

Each tracker sees the same camera-compensated crops and sequence-model seeds.
Neither a high visibility score nor agreement is proof that a point is the ball.
"""
from pathlib import Path
import hashlib
import json
import time
import tempfile
import shutil
import gc

BASE = Path('/data/ball-hockey/reliability-20261018')
OUT = BASE/'ball-temporal-20261026-v1'
SOURCE = BASE/'overhead-20261025-v1/frames'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, separators=(',', ':'))+'\n')
    temp.replace(path)


def verifier():
    import torch
    from crop_verifier import SpatialBallVerifier
    weights = BASE/'ball-rebuild-20261026-v1/verifier-v4/best.pt'
    model = SpatialBallVerifier().cuda().eval()
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=False)['state_dict'])
    return model, sha(weights)


def appearance(model, frame, points, source=SOURCE):
    import numpy as np
    import torch
    import cv2
    from crop_verifier import patch_inputs
    im = cv2.cvtColor(cv2.imread(str(source/f'{frame+1:06d}.jpg')), cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(np.stack([patch_inputs(im, p) for p in points])).cuda()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        return model(x).float().sigmoid().cpu().tolist()


def prepare_seeds(batch, pilot, commit):
    import numpy as np
    import torch
    import cv2
    torch.set_num_threads(4); cv2.setNumThreads(2)
    dest = OUT/f'tracking-seeds-{batch}.json'
    if dest.exists(): return {'cached': True, 'batch': batch, 'sha256': sha(dest)}
    ledger_path = OUT/('refinement-discovery' if 'discovery' in batch else 'refinement-v2')/'ledger.json'
    ledger = json.loads(ledger_path.read_text())
    audit = json.loads(Path('/work/pilot-audit.json').read_text())
    model, verifier_sha = verifier()
    candidates = []
    pilot_starts = {'0000', '0900', '1800', '2700', '3900'}
    for window, request in sorted(ledger['requests'].items()):
        if pilot and window not in pilot_starts: continue
        for sample in (request.get('answer') or {}).get('samples', []):
            if sample['status'] != 'visible': continue
            f = sample['frame']
            if 466 <= f < 516: continue
            origin = request['origins'][str(f)]
            point = [origin[i]+sample['xy_1000'][i]*[256,144][i]/1000 for i in range(2)]
            score = appearance(model, f, [point])[0]
            candidates.append({**sample, 'pixel': point, 'appearance_score': score,
                               'source_sha256': sha(SOURCE/f'{f+1:06d}.jpg'), 'window': window})
    # Seed selection is sparse and fixed before looking at tracker output.
    # The explicit pilot audit excludes uncertain visual proposals, not tracker errors.
    excluded = {r['frame'] for r in audit['rows'] if r['verdict'] != 'visible_ball'} if pilot else set()
    dense = batch.startswith('dense')
    spacing,half = (3,6) if dense else (12,18)
    eligible = [r for r in candidates if r['frame'] not in excluded and (pilot or dense or r['appearance_score'] >= .2)]
    seeds = []
    for row in sorted(eligible, key=lambda r: (-r['appearance_score'], r['frame'])):
        if all(abs(row['frame']-s['frame']) >= spacing for s in seeds): seeds.append(row)
    seeds.sort(key=lambda r: r['frame'])
    result = {'batch': batch, 'pilot': pilot, 'ledger_sha256': sha(ledger_path),
              'verifier_weights_sha256': verifier_sha, 'pilot_audit_sha256': sha(Path('/work/pilot-audit.json')) if pilot else None,
              'half_window_frames': half, 'crop_size': [512,384], 'seed_minimum_spacing_frames': spacing,
              'seed_appearance_threshold': None if pilot or dense else .2,
              'offload_state_to_cpu': not batch.endswith('-gpu'),
              'cpu_transfer_synchronization': batch.endswith('-sync'),
              'limits': 'Sequence-model proposals are not ground truth. Pilot audit is post-prediction diagnostic.',
              'samples': candidates, 'seeds': seeds}
    save(dest, result); commit()
    return {'batch': batch, 'samples': len(candidates), 'seeds': len(seeds), 'sha256': sha(dest)}


def prepare_transfer(clip, commit):
    """Predict target frames from neighboring automatic detections only."""
    data = json.loads((OUT/'yolo-full1920'/f'{clip}.json').read_text())
    assert data['complete']
    source = (OUT/'extra-frames'/clip if clip.endswith('1200') else
              BASE/'new-games-20261025-v1/frames'/clip)
    batch = f'transfer-{clip}-gpu'
    seeds = []
    for target in range(23,1350,45):
        for f in [target-3,target+3]:
            row = data['frames'][f]
            candidates = row['candidates']
            if not candidates: continue
            best = max(candidates,key=lambda p:p['score']*p['appearance_score'])
            if best['score']*best['appearance_score'] < .01: continue
            seeds.append({'frame':f,**best,'support':'unclear',
                          'evidence':'Automatic YOLO detection in a neighboring frame.',
                          'source_sha256':row['source_sha256']})
    result = {'batch':batch,'pilot':False,'seeds':seeds,'target_frames':list(range(23,1350,45)),
              'source_directory':str(source),'source_frame_count':1350,
              'source_width':data['frames'][0]['width'],'source_height':data['frames'][0]['height'],
              'camera_compensation':False,'half_window_frames':6,'crop_size':[512,384],
              'offload_state_to_cpu':False,'detector_sha256':sha(OUT/'yolo-full1920'/f'{clip}.json'),
              'limits':'Exploratory transfer test on previously scored clips. No reference positions seed tracking. No target-frame detector output enters the tracker.'}
    path = OUT/f'tracking-seeds-{batch}.json'
    if path.exists():
        assert json.loads(path.read_text()) == result
    else:
        save(path,result);commit()
    return batch


def prepare_crop_variant(batch, commit):
    parent={'dense-wide-full-gpu':'dense-full-gpu',
            'dense-wide-discovery-gpu':'dense-discovery-gpu'}.get(batch,'dense-gpu')
    source = OUT/f'tracking-seeds-{parent}.json'
    data = json.loads(source.read_text())
    data['batch'] = batch
    data['parent_seed_file_sha256'] = sha(source)
    if batch.startswith('dense-wide-'): data['crop_size']=[768,576]
    elif batch == 'dense-close-gpu': data['crop_size']=[384,288]
    elif batch == 'dense-motion-gpu': data['motion_guided_crop']=True
    else: raise ValueError(batch)
    path = OUT/f'tracking-seeds-{batch}.json'
    if path.exists(): assert json.loads(path.read_text()) == data
    else: save(path,data);commit()


def run_tracking(kind, batch, commit):
    import numpy as np
    import torch
    import cv2
    torch.set_num_threads(4); cv2.setNumThreads(2)
    seed_path = OUT/f'tracking-seeds-{batch}.json'
    protocol = json.loads(seed_path.read_text())
    source = Path(protocol.get('source_directory',str(SOURCE)))
    source_count = protocol.get('source_frame_count',4500)
    source_size = [protocol.get('source_width',1920),protocol.get('source_height',1080)]
    current_game = protocol.get('camera_compensation',True)
    replay = json.loads(Path('/work/players.json').read_text())
    dest = OUT/f'short-{kind}-{batch}.json'
    result = json.loads(dest.read_text()) if dest.exists() else {
        'tracker': kind, 'seed_file_sha256': sha(seed_path), 'tracks': [],
        'protocol': {k:v for k,v in protocol.items() if k not in ['samples','seeds']}}
    assert result['seed_file_sha256'] == sha(seed_path)
    done = {t['seed_frame'] for t in result['tracks']}
    if len(done) == len(protocol['seeds']): return {'cached': True, 'tracks': len(done)}
    if kind == 'tapnextpp':
        from tapnet.tapnextpp.votsp2026.model import TAPNextPP
        weights = Path('/data/cache/ball-point-trackers/tapnextpp_512.ckpt')
        model = TAPNextPP.from_checkpoint(weights, device='cuda', input_resolution=512)
    elif kind == 'sam2-box10':
        from sam2.build_sam import build_sam2_video_predictor
        weights = Path('/data/models/sam2.1_hiera_large.pt')
        assert sha(weights) == '2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318'
        model = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml',str(weights),
                                         device='cuda',apply_postprocessing=False).eval()
        result['evaluation_mode'] = True
    elif kind in ('sam3','sam3-point','sam3-point-negative','sam3-box6','sam3-box14','sam3-mask3','sam3-mask5'):
        from huggingface_hub import hf_hub_download
        from sam3.model_builder import build_sam3_video_model
        from sam3.model.utils.sam2_utils import _load_img_as_tensor
        weights = Path(hf_hub_download('facebook/sam3', 'sam3.pt', revision='3c879f39826c281e95690f02c7821c4de09afae7', local_files_only=True))
        full = build_sam3_video_model(checkpoint_path=str(weights), load_from_HF=False)
        result['training_mode_before_eval'] = {'full':full.training,'tracker':full.tracker.training,
                                              'modules':sum(m.training for m in full.modules())}
        # Keep inference mode explicit. The measured constructor was already in
        # eval mode; this is not the fix for asynchronous CPU-state corruption.
        full.eval()
        model = full.tracker; model.backbone = full.detector.backbone
        model.eval()
        assert not any(m.training for m in model.modules())
        result['evaluation_mode'] = True
        if protocol.get('cpu_transfer_synchronization'):
            # Diagnostic control only: ensure async device-to-host copies finish
            # before the predictor combines their CPU tensors. Production keeps
            # this small tracking state on the GPU and needs no vendor patch.
            from types import MethodType
            def synchronized(method):
                def wrapped(_self,*args,**kwargs):
                    value=method(*args,**kwargs)
                    torch.cuda.synchronize()
                    return value
                return MethodType(wrapped,model)
            for method in ['_run_single_frame_inference','_run_memory_encoder']:
                setattr(model,method,synchronized(getattr(model,method)))
            result['synchronized_cpu_transfers'] = True
    else: raise ValueError(kind)
    result['weights_sha256'] = sha(weights)
    verify, result['verifier_weights_sha256'] = verifier()
    width,height = protocol['crop_size']; half = protocol['half_window_frames']
    started = time.monotonic()
    for seed in protocol['seeds']:
        sf = seed['frame']
        if sf in done: continue
        start,stop = max(0,sf-half),min(source_count,sf+half+1)
        if current_game:
            if sf < 466: stop = min(stop,466)
            elif sf >= 516: start = max(start,516)
            else: continue
        point = np.array(seed['pixel']); h0 = replay['frames'][sf].get('image_to_floor') if current_game else None
        velocities = {}
        motion_guides = {}
        if protocol.get('motion_guided_crop') and current_game:
            for direction in [-1,1]:
                # Forward crops use a prior sighting; backward crops use a later
                # one. The target side never supplies a guiding position.
                candidates=[s for s in protocol['seeds'] if 0 < direction*(sf-s['frame']) <= 12]
                if not candidates: continue
                guide=min(candidates,key=lambda s:abs(sf-s['frame']))
                hp=replay['frames'][guide['frame']].get('image_to_floor')
                gp=np.array(guide['pixel'],dtype=float)
                if hp is not None and h0 is not None:
                    try:
                        q=np.linalg.inv(np.array(h0))@np.array(hp)@np.r_[gp,1]
                        gp=q[:2]/q[2]
                    except np.linalg.LinAlgError: continue
                velocity=(point-gp)/(sf-guide['frame'])
                if np.isfinite(velocity).all() and np.linalg.norm(velocity)<=140:
                    velocities[direction]=velocity;motion_guides[direction]=guide['frame']
        origins,crops = [],[]; sequence_hash = hashlib.sha256()
        for f in range(start,stop):
            center = point.copy(); h = replay['frames'][f].get('image_to_floor') if current_game else None
            if h0 is not None and h is not None:
                try:
                    q = np.linalg.inv(np.array(h))@np.array(h0)@np.r_[point,1]
                    p = q[:2]/q[2]
                    if np.isfinite(p).all() and np.linalg.norm(p-point)<2200: center=p
                except np.linalg.LinAlgError: pass
            direction=1 if f>=sf else -1
            if direction in velocities: center=center+(f-sf)*velocities[direction]
            origin = center-[width/2,height/2]; origins.append(origin)
            path = source/f'{f+1:06d}.jpg'; sequence_hash.update(hashlib.sha256(path.read_bytes()).digest())
            image = cv2.imread(str(path))
            crops.append(cv2.warpAffine(image, np.array([[1,0,-origin[0]],[0,1,-origin[1]]],np.float32),
                         (width,height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(96,96,96)))
        index=sf-start; qxy=point-origins[index]; rows={}; inference_start=time.monotonic()
        if kind == 'tapnextpp':
            for direction in [1,-1]:
                state=None
                for i in range(index,len(crops) if direction==1 else -1,direction):
                    pos,vis,state=model.track_frame(crops[i],query_points_xy=qxy[None] if state is None else None,state=state)
                    rows[start+i]={'local_pixel':pos[0].tolist(),'visible':bool(vis[0]),'shape_valid':True}
        elif kind == 'sam2-box10':
            folder = Path(tempfile.mkdtemp(prefix='ball-sam2-'))
            for i,crop in enumerate(crops):
                cv2.imwrite(str(folder/f'{i:06d}.jpg'),crop,[cv2.IMWRITE_JPEG_QUALITY,100])
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                state = model.init_state(video_path=str(folder),offload_video_to_cpu=True,
                                         offload_state_to_cpu=False)
                model.add_new_points_or_box(state,frame_idx=index,obj_id=1,
                                            box=np.r_[qxy-5,qxy+5].astype(np.float32))
                for reverse in [False,True]:
                    for i,ids,masks in model.propagate_in_video(state,start_frame_idx=index,
                            max_frame_num_to_track=len(crops),reverse=reverse):
                        count,components,stats,centers = cv2.connectedComponentsWithStats(
                            (masks[0,0]>0).cpu().numpy().astype(np.uint8),8)
                        p=qxy;area=w=h=0
                        if count>1:
                            selected=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
                            x,y,w,h,area=stats[selected].tolist();p=centers[selected]
                        rows[start+i]={'local_pixel':p.tolist(),'visible':count>1,
                            'shape_valid':bool(2<=area<=500 and max(w,h)<=40),
                            'mask_area':area,'mask_extent':[w,h],'mask_components':count-1}
                del state,masks
            shutil.rmtree(folder)
        else:
            folder=Path(tempfile.mkdtemp(prefix='ball-short-'))
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                images=torch.empty(len(crops),3,model.image_size,model.image_size,dtype=torch.float32)
                for i,crop in enumerate(crops):
                    path=folder/f'{i:06d}.png';cv2.imwrite(str(path),crop)
                    tensor,vh,vw=_load_img_as_tensor(str(path),model.image_size)
                    assert (vw,vh)==(width,height);images[i]=tensor
                images.sub_(.5).div_(.5)
                state=model.init_state(video_height=height,video_width=width,num_frames=len(crops),offload_video_to_cpu=True,
                                       offload_state_to_cpu=protocol.get('offload_state_to_cpu',True))
                state['images']=images
                if kind == 'sam3-point-negative':
                    points = qxy + np.array([[0,0],[-15,-15],[-15,15],[15,-15],[15,15]])
                    prompt = {'points':(points/[width,height]).astype(np.float32),
                              'labels':np.array([1,0,0,0,0],np.int32)}
                elif kind == 'sam3-point':
                    prompt = {'points':(qxy/[width,height])[None].astype(np.float32),
                              'labels':np.array([1],np.int32)}
                else:
                    radius = {'sam3-box6':3,'sam3-box14':7}.get(kind,5)
                    prompt = {'box':(np.r_[qxy-radius,qxy+radius]/[width,height,width,height]).astype(np.float32)}
                if kind in ('sam3-mask3','sam3-mask5'):
                    yy,xx=np.mgrid[:height,:width]
                    radius=int(kind[-1])
                    mask=torch.from_numpy((xx-qxy[0])**2+(yy-qxy[1])**2 <= radius**2)
                    model.add_new_mask(state,frame_idx=index,obj_id=1,mask=mask)
                else:
                    model.add_new_points_or_box(state,frame_idx=index,obj_id=1,rel_coordinates=True,**prompt)
                for reverse in [False,True]:
                    for i,ids,low,masks,scores in model.propagate_in_video(state,start_frame_idx=index,max_frame_num_to_track=len(crops),reverse=reverse,propagate_preflight=True,tqdm_disable=True):
                        count,components,stats,centers=cv2.connectedComponentsWithStats((masks[0,0]>0).cpu().numpy().astype(np.uint8),8)
                        presence=float(scores.float().sigmoid().flatten()[0].cpu())
                        p=qxy;area=w=h=0
                        if count>1:
                            selected=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));x,y,w,h,area=stats[selected].tolist();p=centers[selected]
                        rows[start+i]={'local_pixel':p.tolist(),'visible':bool(count>1 and presence>=.5),
                                      'shape_valid':bool(2<=area<=500 and max(w,h)<=40),'mask_area':area,'mask_extent':[w,h],'presence_score':presence,'mask_components':count-1}
                del state,images,masks,low
            shutil.rmtree(folder)
        for f,row in rows.items():
            p=np.array(row.pop('local_pixel'));native=p+origins[f-start]
            row.update(frame=f,pixel=native.tolist(),inside_crop=bool(np.all(p>[8,8])and np.all(p<[width-8,height-8])),
                       inside_image=bool(np.all(native>=0)and np.all(native<source_size)),appearance_score=0.)
            if row['visible']and row['inside_crop']and row['inside_image']:
                row['appearance_score']=appearance(verify,f,[native],source)[0]
        seed_error=float(np.linalg.norm(np.array(rows[sf]['pixel'])-point))
        result['tracks'].append({'seed_frame':sf,'seed_pixel':point.tolist(),'seed_support':seed['support'],
            'motion_guides':motion_guides,
            'seed_error_px':seed_error,'seed_mask_valid':bool(rows[sf]['visible']and rows[sf]['shape_valid']and seed_error<=12),
            'interval':[start,stop],'input_frame_sequence_sha256':sequence_hash.hexdigest(),
            'inference_seconds':time.monotonic()-inference_start,'rows':[rows[f]for f in sorted(rows)]})
        result['seconds']=time.monotonic()-started;result['complete']=len(result['tracks'])==len(protocol['seeds'])
        save(dest,result)
        # Persist every seed locally, but avoid a remote volume barrier for each
        # one-second GPU window. A restart loses at most nineteen seed windows.
        if len(result['tracks']) % 20 == 0 or result['complete']: commit()
        print({'tracker':kind,'batch':batch,'seed':sf,'done':len(result['tracks']),'total':len(protocol['seeds']),
               'seconds':round(result['tracks'][-1]['inference_seconds'],1)},flush=True)
        gc.collect();torch.cuda.empty_cache()
    return {'tracker':kind,'batch':batch,'tracks':len(result['tracks']),'seconds':result['seconds']}
