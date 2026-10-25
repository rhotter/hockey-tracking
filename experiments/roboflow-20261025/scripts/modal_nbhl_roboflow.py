"""Run pinned Roboflow components on existing NBHL inputs in Modal."""
from pathlib import Path
import modal
import sys
if not modal.is_local():sys.path.insert(0,'/work')
from modal_campaign import ROOT, base_image, volume

app = modal.App('sport-scans-roboflow-20261025')
image = (base_image
    .pip_install('numpy==2.2.6','scipy==1.16.1','pandas==2.3.3','matplotlib==3.10.7',
                 'scikit-learn==1.7.2','umap-learn==0.5.9.post2','supervision==0.27.0',
                 'transformers==5.17.0','sentencepiece','protobuf',
                 'trackers @ git+https://github.com/roboflow/trackers.git@54078229c1bf381bcdc41c431ca11097abfec16c',
                 'sports @ git+https://github.com/roboflow/sports.git@42c80c06b6b65a7f89455b89fe31cdf4c38ba227')
    .env({'HF_HOME':'/data/cache/huggingface','RF_HOME':'/data/cache/rfdetr',
          'HF_HUB_DISABLE_XET':'1','HF_XET_CACHE':'/tmp/hf-xet','TOKENIZERS_PARALLELISM':'false',
          'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}))
detector_image=(base_image.pip_install(
    'rfdetr[train] @ git+https://github.com/roboflow/rf-detr.git@5b22dfc0940697ce1abdfa681a5f7392b3eddbfb',
    'transformers==5.17.0').env({'RF_HOME':'/data/cache/rfdetr','HF_HUB_DISABLE_XET':'1','HF_XET_CACHE':'/tmp/hf-xet'}))
for name in ['modal_campaign.py','nbhl_roboflow.py','nbhl_tracking_fourhour.py','nbhl_detection_metrics.py',
             'nbhl_tracking_reference_audit.py','evaluate_nbhl_jersey.py','roboflow_existing_linker.py']:
    image=image.add_local_file(ROOT/'scripts'/name,'/work/'+name)
    detector_image=detector_image.add_local_file(ROOT/'scripts'/name,'/work/'+name)
for name in ['roboflow-replacement-protocol-20261025.json','tracking-four-hour-reference-corrections-v1.json',
             'roboflow-team-torso-followup-20261025.json','roboflow-hybrid-protocol-20261025.json',
             'roboflow-existing-baseline-20261025.json','rink-relative-nj-v1.json','jersey-number-review-v2.json','sources.json']:
    image=image.add_local_file(ROOT/'data/ball-hockey'/name,'/work/'+name)
    detector_image=detector_image.add_local_file(ROOT/'data/ball-hockey'/name,'/work/'+name)
image=image.add_local_file(ROOT/'notes/ball-hockey/ball-neighbor-eval-yolo11m-crops640-s17-f10-e80-c6ab3e44-full1280-dense.json','/work/ball-neighbor-reference-report.json')


@app.function(image=image,cpu=4,memory=16384,timeout=7200,max_containers=2,scaledown_window=10,volumes={'/data':volume})
def cpu(task,phase='development',clip=''):
    import nbhl_roboflow as impl
    volume.reload()
    try:return getattr(impl,task)(phase,clip,volume)
    finally:volume.commit()


@app.function(image=image,cpu=8,memory=32768,gpu='L40S',timeout=7200,max_containers=2,scaledown_window=10,volumes={'/data':volume})
def gpu(task,phase='development',clip=''):
    import nbhl_roboflow as impl
    volume.reload()
    try:return getattr(impl,task)(phase,clip,volume)
    finally:volume.commit()


@app.function(image=detector_image,cpu=8,memory=32768,gpu='L40S',timeout=7200,max_containers=2,scaledown_window=10,volumes={'/data':volume})
def detector(task,phase='development',clip=''):
    import nbhl_roboflow as impl
    volume.reload()
    try:return getattr(impl,task)(phase,clip,volume)
    finally:volume.commit()


@app.local_entrypoint()
def main(task:str='preflight',phase:str='development',clips:str=''):
    import json
    protocol=json.loads((ROOT/'data/ball-hockey/roboflow-replacement-protocol-20261025.json').read_text())
    names=clips.split(',') if clips else protocol.get(phase,[])
    function=detector if task=='detect' else gpu if task=='teams' else cpu
    if task in ['preflight','select','utilities','export','teams','detector_audit','prefetch_check']:
        result=function.remote(task,phase,clips)
    elif phase=='all':
        assert task=='detect'
        result=list(function.starmap([(task,p,c) for p in ['development','regression'] for c in protocol[p]],order_outputs=False))
    else:
        result=list(function.starmap([(task,phase,c) for c in names],order_outputs=False))
    dest=ROOT/'notes/ball-hockey/roboflow-20261025';dest.mkdir(parents=True,exist_ok=True)
    (dest/(task+'-'+phase+('-'+clips.replace(',','_') if clips else '')+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
