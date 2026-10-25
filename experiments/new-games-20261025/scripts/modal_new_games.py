"""Frozen new-recording comparison. All video/model work runs on Modal."""
import json
from pathlib import Path
import sys
import modal
from modal_campaign import ROOT, base_image, volume

if not modal.is_local():
    sys.path.insert(0, '/work')
app = modal.App('sport-scans-new-games-20261025')
cpu_image = base_image
rf_image = base_image.pip_install('rfdetr[train] @ git+https://github.com/roboflow/rf-detr.git@5b22dfc0940697ce1abdfa681a5f7392b3eddbfb', 'transformers==5.17.0')
tracker_image = base_image.pip_install('numpy==2.2.6','scipy==1.16.1','supervision==0.27.0',
    'trackers @ git+https://github.com/roboflow/trackers.git@54078229c1bf381bcdc41c431ca11097abfec16c')
embedding_image = (base_image.pip_install('transformers==4.57.6','qwen-vl-utils==0.0.14','accelerate==1.11.0','av==16.0.1')
    .run_commands('git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git /opt/qwen-embedding',
                  'git -C /opt/qwen-embedding checkout 393e2978d27852b0d0230d6994f37f9c15bed73c')
    .env({'PYTHONPATH':'/opt/qwen-embedding:/opt/smp:/work','HF_HOME':'/data/cache/huggingface',
          'HF_HUB_DISABLE_XET':'1','HF_XET_CACHE':'/tmp/hf-xet'}))

def mount(image):
    image = image.env({'HF_HUB_DISABLE_XET':'1','HF_XET_CACHE':'/tmp/hf-xet','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1'})
    for p in sorted((ROOT/'scripts').glob('*.py')):
        if p.name != 'modal_new_games.py':
            image = image.add_local_file(p, '/work/'+p.name)
    image = image.add_local_file(ROOT/'data/protocol.json','/work/new-games-protocol.json')
    for name in ['reference.json','source-review.json']:
        if (ROOT/'data'/name).exists():
            image = image.add_local_file(ROOT/'data'/name,'/work/'+name)
    return image

cpu_image,rf_image,tracker_image,embedding_image = map(mount,[cpu_image,rf_image,tracker_image,embedding_image])

def run(task,clip='',payload=None):
    import new_games
    volume.reload()
    try:
        if task not in ['prepare','export']:
            new_games.record_start(task,clip)
            volume.commit()
        return getattr(new_games,task)(clip,payload,volume)
    finally:
        volume.commit()

@app.function(image=cpu_image,cpu=8,memory=16384,timeout=3600,max_containers=2,scaledown_window=10,volumes={'/data':volume})
def prepare(source,payload):
    return run('prepare',source,payload)

@app.function(image=rf_image,cpu=8,memory=24576,gpu='L40S',timeout=3600,max_containers=2,scaledown_window=10,volumes={'/data':volume})
def detect(clip):
    return run('detect',clip)

@app.function(image=tracker_image,cpu=8,memory=16384,timeout=3600,max_containers=2,scaledown_window=10,volumes={'/data':volume})
def track(clip):
    return run('track',clip)

@app.function(image=cpu_image,cpu=8,memory=49152,gpu='L40S',timeout=7200,max_containers=2,scaledown_window=10,volumes={'/data':volume})
def sam(clip):
    return run('sam',clip)

@app.function(image=embedding_image,cpu=8,memory=49152,gpu='L40S',timeout=7200,max_containers=2,scaledown_window=10,volumes={'/data':volume})
def embed(clip):
    return run('embed',clip)

@app.function(image=cpu_image,cpu=4,memory=16384,timeout=1800,volumes={'/data':volume})
def export():
    return run('export')

@app.local_entrypoint()
def main(task:str='prepare',metadata:str='',clips:str=''):
    protocol=json.loads((ROOT/'data/protocol.json').read_text())
    names=clips.split(',') if clips else [c['clip']for c in protocol['clips']]
    if task=='prepare':
        entries={e['id']:e for e in map(json.loads,Path(metadata).read_text().splitlines())}
        jobs=[]
        for source in protocol['sources']:
            if clips and source['id'] not in names:
                continue
            e=entries[source['id']]
            f=next(f for f in e['formats'] if f['format_id']=='137')
            jobs.append((e['id'],{'url':f['url'],'headers':f.get('http_headers',{}),'expected_bytes':f['filesize']}))
        result=list(prepare.starmap(jobs,order_outputs=False))
    elif task=='export':
        result=export.remote()
    else:
        function={'detect':detect,'track':track,'sam':sam,'embed':embed}[task]
        result=list(function.map(names,order_outputs=False))
    dest=ROOT.parents[1]/'build/new-games-20261025';dest.mkdir(parents=True,exist_ok=True)
    suffix='-'+clips.replace(',','_')if clips else ''
    (dest/(task+suffix+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
