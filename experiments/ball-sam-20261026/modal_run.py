"""Point/box-prompted SAM3 ball tracking on identical NBHL seed windows."""
from pathlib import Path
import modal

HERE=Path(__file__).parent;ROOT=HERE.parents[1] if modal.is_local() else Path('/work')
app=modal.App('sport-scans-ball-sam-20261026')
volume=modal.Volume.from_name('sport-scans-experiments')
base_image=(modal.Image.from_registry('python:3.12-slim-bookworm')
    .apt_install('git','ffmpeg','libgl1','libglib2.0-0','build-essential')
    .pip_install('torch==2.9.1','torchvision==0.24.1',index_url='https://download.pytorch.org/whl/cu128')
    .pip_install('ultralytics==8.4.181','numpy==1.26.4','opencv-python==4.11.0.86',
        'scipy','lap==0.5.13','cython-bbox','huggingface_hub','hydra-core','iopath','einops','tqdm','pillow','matplotlib','pandas','pycocotools',
        'loguru','filterpy','timm','yacs','gdown','tensorboard','tabulate')
    .run_commands('git clone https://github.com/holma91/selective-mask-propagation.git /opt/smp && cd /opt/smp && git checkout 6472b52e79c47a7793cd4c6cabd64e812a8e5852',
        'cd /opt/smp/vendor/sam2 && SAM2_BUILD_CUDA=0 pip install --no-build-isolation --no-deps -e .')
    .env({'PYTHONPATH':'/opt/smp:/work','HF_HOME':'/data/cache/huggingface','YOLO_CONFIG_DIR':'/data/cache/ultralytics'}))
image=(base_image.pip_install('ftfy==6.1.1','regex','decord','setuptools<81')
    .run_commands('git clone https://github.com/facebookresearch/sam3.git /opt/sam3',
        'git -C /opt/sam3 checkout 7f473cfd983d5eb43040ff608f66bff0d7bdcb53','pip install --no-deps -e /opt/sam3')
    .env({'PYTHONPATH':'/opt/sam3:/opt/smp:/work','HF_HOME':'/data/cache/huggingface','HF_HUB_DISABLE_XET':'1'})
    .add_local_file(HERE/'worker.py','/work/worker.py')
    .add_local_file(HERE/'protocol.json','/work/protocol.json')
    .add_local_file(ROOT/'web/data/overhead-20261025/replay.json','/work/players.json')
    .add_local_file(ROOT/'web/data/ball-rebuild-20261026/replay.json','/work/anchors.json')
    .add_local_file(ROOT/'experiments/ball-sequence-20261026/results/tapnextpp512-pilot.json','/work/tap-pilot.json')
    .add_local_file(ROOT/'experiments/ball-rebuild-20261026/crop_verifier.py','/work/crop_verifier.py'))

@app.function(image=image,cpu=4,memory=8192,timeout=600,volumes={'/data':volume})
def probe():
    import json,inspect
    from huggingface_hub import hf_hub_download
    from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor
    volume.reload()
    p=hf_hub_download('facebook/sam3','sam3.pt',revision='3c879f39826c281e95690f02c7821c4de09afae7',local_files_only=True)
    return {'checkpoint_cached':Path(p).exists(),'checkpoint_bytes':Path(p).stat().st_size,
        'frames':len(list(Path('/data/ball-hockey/reliability-20261018/overhead-20261025-v1/frames').glob('*.jpg'))),
        'add_prompt':str(inspect.signature(Sam3TrackerPredictor.add_new_points_or_box)),
        'propagate':str(inspect.signature(Sam3TrackerPredictor.propagate_in_video))}

@app.function(image=image,cpu=8,memory=98304,gpu='H100',timeout=7200,max_containers=3,scaledown_window=10,volumes={'/data':volume})
def run(variant:str,phase:str='pilot'):
    from worker import run_tracking
    volume.reload()
    return run_tracking(variant,phase,volume.commit)

@app.local_entrypoint()
def main(task:str='smoke',variants:str='sam3-point-768,sam3-box-768,sam3-box-384'):
    if task=='probe':print(probe.remote())
    elif task=='smoke':print(run.remote('sam3-box-768','smoke'))
    else:
        assert task in ['pilot','all']
        print(list(run.starmap([(v,task)for v in variants.split(',')],order_outputs=False)))
