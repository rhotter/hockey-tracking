"""Prepare SAM 3D Body and run a fixed NBHL foot audit only after HF approval.

No GPU is allocated for preparation or an unsuccessful checkpoint preflight.
The 48 existing automatic boxes are prompts, not manually repaired identities.
"""
from pathlib import Path
import modal
from modal_campaign import ROOT,base_image,volume

BASE=Path('/data/ball-hockey/reliability-20261018')
CODE_REVISION='b5c765a0d89d789985e186d396315e7590887b94'
REPOSITORY='facebook/sam-3d-body-dinov3'
WEIGHTS_REVISION='11aaa346c7204874a1cbafe3d39a979080b2c55a'
app=modal.App('sport-scans-nbhl-sam3d-body')
image=(base_image
    .pip_install('pytorch-lightning','pyrender','scikit-image','dill','rich','hydra-submitit-launcher',
                 'hydra-colorlog','pyrootutils','webdataset','chump','networkx==3.2.1','roma','joblib',
                 'seaborn','wandb','appdirs','appnope','ffmpeg','jsonlines','optree','fvcore')
    .run_commands('pip install --no-build-isolation xtcocotools==1.14.3')
    .run_commands('git clone https://github.com/facebookresearch/sam-3d-body.git /opt/sam-3d-body',
                  f'git -C /opt/sam-3d-body checkout {CODE_REVISION}')
    .env({'PYTHONPATH':'/opt/sam-3d-body:/opt/smp:/work','HF_HUB_DISABLE_XET':'1','HF_XET_CACHE':'/tmp/hf-xet','HF_HOME':'/data/cache/huggingface'})
    .add_local_file(ROOT/'scripts/modal_campaign.py','/work/modal_campaign.py')
    .add_local_file(ROOT/'data/ball-hockey/map-feet-visibility-v1.json','/work/foot-reference.json'))


@app.function(image=image,cpu=2,memory=8192,timeout=600,scaledown_window=10)
def prepare():
    import importlib.metadata,json,subprocess
    import sam_3d_body
    from sam_3d_body.metadata.mhr70 import mhr_names
    assert mhr_names[13:15]==['left-ankle','right-ankle']
    assert subprocess.check_output(['git','-C','/opt/sam-3d-body','rev-parse','HEAD'],text=True).strip()==CODE_REVISION
    return {'id':'nbhl-sam3d-runtime-v1','code_revision':CODE_REVISION,'import_succeeded':True,
            'checkpoint_loaded':False,'gpu_allocated':False,'inference_performed':False,
            'versions':{p:importlib.metadata.version(p)for p in ['torch','torchvision','pytorch-lightning','timm','numpy','roma','huggingface_hub']},
            'scope':'Runtime preparation only. Supplied NBHL boxes bypass optional external person detector and FOV models.'}


@app.function(image=image,cpu=2,memory=8192,timeout=3600,scaledown_window=10,
              secrets=[modal.Secret.from_name('huggingface',required_keys=['HF_TOKEN'])],volumes={'/data':volume})
def approved_checkpoint():
    import os,datetime
    from huggingface_hub import get_hf_file_metadata,hf_hub_url,snapshot_download
    try:
        metadata=get_hf_file_metadata(hf_hub_url(REPOSITORY,'model.ckpt',revision=WEIGHTS_REVISION),token=os.environ['HF_TOKEN'])
    except Exception as exc:
        return {'accessible':False,'error_type':type(exc).__name__,'http_status':getattr(getattr(exc,'response',None),'status_code',None),
                'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu_allocated':False}
    path=snapshot_download(REPOSITORY,revision=WEIGHTS_REVISION,token=os.environ['HF_TOKEN'],cache_dir='/data/cache/huggingface/hub');volume.commit()
    return {'accessible':True,'path':path,'checkpoint_bytes':metadata.size,'revision':WEIGHTS_REVISION}


@app.function(image=image,cpu=8,memory=49152,gpu=['H100','A100-80GB'],timeout=3600,
              scaledown_window=10,volumes={'/data':volume})
def audit(checkpoint_directory:str):
    import hashlib,json,time
    import cv2,numpy as np,torch
    from sam_3d_body import SAM3DBodyEstimator,load_sam_3d_body
    from sam_3d_body.metadata.mhr70 import mhr_names
    volume.reload();torch.set_num_threads(4);cv2.setNumThreads(2)
    folder=Path(checkpoint_directory);assert folder.resolve().is_relative_to(Path('/data/cache/huggingface').resolve()) and folder.name==WEIGHTS_REVISION, str(folder)
    def sha(path):
        with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
    model,cfg=load_sam_3d_body(checkpoint_path=str(folder/'model.ckpt'),mhr_path=str(folder/'assets/mhr_model.pt'),device='cuda')
    estimator=SAM3DBodyEstimator(model,cfg)
    path=Path('/work/foot-reference.json');reference=json.loads(path.read_text());assert reference['source_id']=='OLpK8enSjJs'
    assert len(reference['observations'])==48
    source=BASE/'relative-map-v1/nj-0900-20s';out=BASE/'sam3d-body-foot-audit-v1';out.mkdir(exist_ok=True)
    results=[];began=time.monotonic()
    for row in reference['observations']:
        frame_path=source/'frames'/f'{row["frame"]+1:06d}.jpg';assert sha(frame_path)==row['frame_sha256']
        raw=cv2.imread(str(frame_path));assert raw is not None
        predicted=estimator.process_one_image(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB),bboxes=np.asarray([row['bbox']],np.float32),inference_type='body')
        assert len(predicted)==1
        points=np.asarray(predicted[0]['pred_keypoints_2d'])[:70];assert points.shape==(70,2)
        values=points.tolist() if np.isfinite(points).all() else None
        result={k:row[k]for k in ['id','frame','track_id','bbox','frame_sha256']}
        result.update(keypoints_2d=values,ankle_pixels=points[13:15].tolist()if values is not None else None,
                      visibility_confidence=None,metric_position=None)
        results.append(result)
        canvas=raw.copy();x,y,u,v=map(int,row['bbox']);cv2.rectangle(canvas,(x,y),(u,v),(0,210,255),2)
        if values is not None:
            for j in range(9,21):cv2.circle(canvas,tuple(np.rint(points[j]).astype(int)),4,(0,210,255),-1)
        cv2.imwrite(str(out/f'{row["id"]:03d}.jpg'),cv2.resize(canvas,(1280,720)))
    report={'id':'nbhl-sam3d-body-foot-audit-v1','source_id':reference['source_id'],'reference_sha256':sha(path),
            'code_revision':CODE_REVISION,'weights_repository':REPOSITORY,'weights_revision':WEIGHTS_REVISION,
            'checkpoint_sha256':sha(folder/'model.ckpt'),'mhr_sha256':sha(folder/'assets/mhr_model.pt'),
            'inference_type':'body','camera':'Default model field of view; not calibrated.',
            'keypoint_names':mhr_names[:70],'seconds':time.monotonic()-began,'observations':results,
            'limitations':'Fixed 48 automatic NBHL track boxes. Predicted joints have no measured visibility confidence in this interface. Finite points are not proof of visible feet or floor contact. No metric position, distance, time on floor or identity is inferred. Requires separate visual review.'}
    (out/'predictions.json').write_text(json.dumps(report,indent=2));volume.commit()
    return {k:v for k,v in report.items()if k not in ['observations','keypoint_names']}


@app.local_entrypoint()
def main(task:str='prepare'):
    import json
    assert task in ('prepare','run')
    if task=='prepare':result=prepare.remote()
    else:
        result=approved_checkpoint.remote()
        if result['accessible']:result=audit.remote(result['path'])
    (ROOT/'notes/ball-hockey'/('sam3d-body-'+task+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
