"""NBHL-only temporal ball fine-tuning, native search and frozen-game review."""
from pathlib import Path
import modal

HERE=Path(__file__).parent
ROOT=HERE.parents[1] if modal.is_local() else Path('/work')
BASE=Path('/data/ball-hockey/reliability-20261018')
OUT=BASE/'ball-temporal-20261026-v1'
volume=modal.Volume.from_name('sport-scans-experiments')
app=modal.App('sport-scans-ball-temporal-20261026')
image=(modal.Image.from_registry('python:3.12-slim-bookworm')
       .apt_install('ffmpeg','libgl1','libglib2.0-0')
       .pip_install('torch==2.9.1','torchvision==0.24.1',index_url='https://download.pytorch.org/whl/cu128')
       .pip_install('numpy==1.26.4','opencv-python-headless==4.11.0.86','pillow==11.3.0','scipy==1.17.1','ultralytics==8.4.181')
       .env({'TORCH_HOME':'/data/cache/torch','YOLO_CONFIG_DIR':'/data/cache/ultralytics','OMP_NUM_THREADS':'4'})
       .add_local_file(HERE/'protocol.json','/work/protocol.json')
       .add_local_file(HERE/'protocol-center.json','/work/protocol-center.json')
       .add_local_file(HERE/'protocol-hard.json','/work/protocol-hard.json')
       .add_local_file(HERE/'training-labels.json','/work/training-labels.json')
       .add_local_file(HERE/'reused-reference.json','/work/reused-reference.json')
       .add_local_file(HERE/'worker.py','/work/worker.py')
       .add_local_file(HERE/'temporal_model.py','/work/temporal_model.py')
       .add_local_file(ROOT/'experiments/ball-rebuild-20261026/heatmap_model.py','/work/heatmap_model.py')
       .add_local_file(ROOT/'experiments/ball-rebuild-20261026/crop_verifier.py','/work/crop_verifier.py')
       .add_local_file(ROOT/'experiments/ball-rebuild-20261026/regression-labels.json','/work/regression-labels.json')
       .add_local_file(ROOT/'web/data/ball-sam-20261026/positions.json','/work/baseline.json')
       .add_local_file(ROOT/'web/data/ball-sequence-20261026/context-replay.json','/work/context.json')
       .add_local_file(ROOT/'web/data/overhead-20261025/replay.json','/work/players.json'))

@app.function(image=image,cpu=8,memory=32768,timeout=3600,volumes={'/data':volume})
def prepare():
    import sys
    sys.path.insert(0,'/work')
    from worker import prepare_data
    volume.reload()
    return prepare_data(volume.commit)

@app.function(image=image,cpu=12,memory=65536,gpu='H100',timeout=10800,max_containers=4,
              scaledown_window=10,volumes={'/data':volume})
def train(variant:str):
    import sys
    sys.path.insert(0,'/work')
    from worker import train_model
    volume.reload()
    return train_model(variant,volume.commit)

@app.function(image=image,cpu=8,memory=32768,gpu='H100',timeout=7200,max_containers=4,
              scaledown_window=10,volumes={'/data':volume})
def infer(variant:str,scope:str):
    import sys
    sys.path.insert(0,'/work')
    from worker import infer_model
    volume.reload()
    return infer_model(variant,scope,volume.commit)

@app.function(image=image,cpu=8,memory=32768,gpu='L40S',timeout=7200,max_containers=2,
              scaledown_window=10,volumes={'/data':volume})
def baseline(clip:str):
    import sys
    sys.path.insert(0,'/work')
    from worker import infer_baseline
    volume.reload()
    return infer_baseline(clip,volume.commit)

@app.local_entrypoint()
def main(task:str='prepare',variants:str='rgb-expanded,temporal3-expanded,motion3-expanded,motion3-synthetic',scope:str='regression'):
    import json
    if task=='prepare':results=[prepare.remote()]
    elif task=='train':results=list(train.map(variants.split(','),order_outputs=False))
    elif task=='infer':results=list(infer.starmap([(v,scope)for v in variants.split(',')],order_outputs=False))
    elif task=='baseline':
        suffix='1800' if scope=='final-check' else '1200' if scope=='extra' else '0600'
        clips=['regression'] if scope=='regression' else [f'pittsburgh-finals-{suffix}',f'southjersey-finals-{suffix}']
        results=list(baseline.map(clips,order_outputs=False))
    else:raise ValueError(task)
    dest=ROOT/'build/ball-temporal-20261026';dest.mkdir(parents=True,exist_ok=True)
    (dest/f'{task}-{scope}-summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
