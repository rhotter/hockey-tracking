"""SAM3 temporal mask propagation; masks are converted to one ball candidate."""
from pathlib import Path
import json,hashlib,time,shutil,tempfile,gc

BASE=Path('/data/ball-hockey/reliability-20261018')
OUT=BASE/'ball-sam-20261026-v1'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,separators=(',',':'))+'\n');temp.replace(path)

def run_tracking(variant,phase,commit):
    import numpy as np,torch,cv2
    from huggingface_hub import hf_hub_download
    from sam3.model_builder import build_sam3_video_model
    from sam3.model.utils.sam2_utils import _load_img_as_tensor
    from crop_verifier import SpatialBallVerifier,patch_inputs
    torch.set_num_threads(4);cv2.setNumThreads(2)
    protocol=json.loads(Path('/work/protocol.json').read_text());replay=json.loads(Path('/work/players.json').read_text())
    anchors=json.loads(Path('/work/anchors.json').read_text())
    assert sha(Path('/work/anchors.json'))==protocol['baseline_sha256']
    assert variant in protocol['models']
    width=int(variant.rsplit('-',1)[1]);height=width*3//4;prompt='point'if'-point-'in variant else'box'
    seeds=protocol['pilot_seeds']
    if phase=='all':
        seeds=[]
        for row in anchors['frames']:
            if row['state']=='detected'and(not seeds or row['frame']-seeds[-1]['frame']>=30):
                seeds.append({'frame':row['frame'],'pixel':row['pixel']})
    if phase=='smoke':seeds=seeds[:1]
    dest=OUT/phase/(variant+'.json')
    result=json.loads(dest.read_text())if dest.exists()else{'model':variant,'phase':phase,'protocol':protocol,'tracks':[]}
    done={r['seed_frame']for r in result['tracks']}
    if len(done)==len(seeds):return {'model':variant,'phase':phase,'seeds':len(done),'cached':True}
    checkpoint=Path(hf_hub_download('facebook/sam3','sam3.pt',revision=protocol['sam3_weights_revision'],local_files_only=True))
    result['weights_sha256']=sha(checkpoint);start_time=time.monotonic()
    full=build_sam3_video_model(checkpoint_path=str(checkpoint),load_from_HF=False)
    core=full.tracker;core.backbone=full.detector.backbone
    verifier_path=BASE/'ball-rebuild-20261026-v1/verifier-v4/best.pt'
    verifier=SpatialBallVerifier().cuda().eval();verifier.load_state_dict(torch.load(verifier_path,map_location='cpu',weights_only=False)['state_dict'])
    result['verifier_weights_sha256']=sha(verifier_path)
    source=BASE/'overhead-20261025-v1/frames'
    if phase=='smoke':
        tap=json.loads(Path('/work/tap-pilot.json').read_text())
        for track in tap['tracks']:
            for row in track['rows']:
                row['appearance_score']=0.
                if row['visible']and row['inside_crop']and row['inside_image']:
                    im=cv2.cvtColor(cv2.imread(str(source/f"{row['frame']+1:06d}.jpg")),cv2.COLOR_BGR2RGB)
                    x=torch.from_numpy(patch_inputs(im,row['pixel']))[None].cuda()
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):row['appearance_score']=float(verifier(x).float().sigmoid().item())
        tap['verifier_weights_sha256']=sha(verifier_path)
        save(OUT/'tapnextpp512-pilot-verified.json',tap);commit()
    for seed in seeds:
        sf=seed['frame']
        if sf in done:continue
        half=8 if phase=='smoke'else protocol['half_window_frames'];start=max(0,sf-half);stop=min(4500,sf+half+1)
        if sf<466:stop=min(stop,466)
        elif sf>=516:start=max(start,516)
        else:continue
        point=np.array(seed['pixel'])*[1920,1080];H0=replay['frames'][sf].get('image_to_floor')
        origins=[];previous=point;folder=Path(tempfile.mkdtemp(prefix='ball-sam-'));input_hash=hashlib.sha256()
        for f in range(start,stop):
            H=replay['frames'][f].get('image_to_floor');center=previous
            if H0 is not None and H is not None:
                try:
                    q=np.linalg.inv(np.array(H))@np.array(H0)@np.r_[point,1];candidate=q[:2]/q[2]
                    if np.isfinite(candidate).all()and np.linalg.norm(candidate-point)<2200:center=candidate
                except np.linalg.LinAlgError:pass
            previous=center;origin=center-[width/2,height/2];origins.append(origin)
            src=source/f'{f+1:06d}.jpg';input_hash.update(hashlib.sha256(src.read_bytes()).digest());frame=cv2.imread(str(src))
            crop=cv2.warpAffine(frame,np.array([[1,0,-origin[0]],[0,1,-origin[1]]],np.float32),(width,height),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(96,96,96))
            # PNG preserves exactly the same native crop pixels used by point trackers.
            cv2.imwrite(str(folder/f'{f-start:06d}.png'),crop)
        index=sf-start;qxy=point-origins[index];rows={};inference_start=time.monotonic()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            # The pinned folder scanner accepts JPEG suffixes only. Use its exact
            # decoder/resizer on lossless PNGs and the same float32 normalization.
            images=torch.empty(len(origins),3,core.image_size,core.image_size,dtype=torch.float32)
            for i in range(len(origins)):
                tensor,vh,vw=_load_img_as_tensor(str(folder/f'{i:06d}.png'),core.image_size)
                assert (vw,vh)==(width,height)
                images[i]=tensor
            images.sub_(.5).div_(.5)
            state=core.init_state(video_height=height,video_width=width,num_frames=len(origins),offload_video_to_cpu=True,offload_state_to_cpu=True)
            state['images']=images
            radius=protocol['box_native_half_size']
            kwargs={'points':(qxy/[width,height])[None].astype(np.float32),'labels':np.array([1],np.int32)}if prompt=='point'else{'box':(np.r_[qxy-radius,qxy+radius]/[width,height,width,height]).astype(np.float32)}
            core.add_new_points_or_box(state,frame_idx=index,obj_id=1,rel_coordinates=True,**kwargs)
            for reverse in [False,True]:
                for i,ids,low,masks,scores in core.propagate_in_video(state,start_frame_idx=index,max_frame_num_to_track=len(origins),reverse=reverse,propagate_preflight=True,tqdm_disable=True):
                    assert ids==[1]
                    mask=(masks[0,0]>0).cpu().numpy().astype(np.uint8)
                    count,components,stats,centers=cv2.connectedComponentsWithStats(mask,8)
                    object_score=float(scores.float().sigmoid().flatten()[0].cpu())
                    if count>1:
                        chosen=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));x,y,w,h,area=stats[chosen].tolist();p=centers[chosen];native=p+origins[i]
                        valid=protocol['minimum_mask_area']<=area<=protocol['maximum_mask_area']and max(w,h)<=protocol['maximum_mask_extent']
                        visible=object_score>=.5
                    else:
                        x=y=w=h=area=0;p=qxy;native=p+origins[i];valid=visible=False
                    rows[start+i]={'frame':start+i,'pixel':native.tolist(),'visible':visible,'inside_crop':bool(np.all(p>[12,12])and np.all(p<[width-12,height-12])),'inside_image':bool(np.all(native>=0)and np.all(native<[1920,1080])),
                        'mask_area':area,'mask_extent':[w,h],'mask_box':[x,y,x+w,y+h],'components':count-1,'shape_valid':bool(valid),'presence_score':object_score}
            del state,images,masks,low
        # Appearance verifier is independent of the SAM mask prediction.
        for f,row in rows.items():
            row['appearance_score']=0.
            if row['visible']and row['inside_crop']and row['inside_image']:
                im=cv2.cvtColor(cv2.imread(str(source/f'{f+1:06d}.jpg')),cv2.COLOR_BGR2RGB)
                x=torch.from_numpy(patch_inputs(im,row['pixel']))[None].cuda()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):row['appearance_score']=float(verifier(x).float().sigmoid().item())
        result['tracks'].append({'seed_frame':sf,'seed_pixel':point.tolist(),'interval':[start,stop],'input_frame_sequence_sha256':input_hash.hexdigest(),'inference_seconds':time.monotonic()-inference_start,'rows':[rows[f]for f in sorted(rows)]})
        result['elapsed_seconds']=time.monotonic()-start_time;result['gpu']=torch.cuda.get_device_name(0)
        save(dest,result);commit();print({'model':variant,'phase':phase,'seed':sf,'seconds':round(result['tracks'][-1]['inference_seconds']),'mask_visible':sum(r['visible']for r in rows.values()),'shape_valid':sum(r['visible']and r['shape_valid']for r in rows.values())},flush=True)
        shutil.rmtree(folder);gc.collect();torch.cuda.empty_cache()
    return {'model':variant,'phase':phase,'seeds':len(result['tracks']),'seconds':result['elapsed_seconds']}
