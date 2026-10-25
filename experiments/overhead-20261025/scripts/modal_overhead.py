"""Five-minute NBHL camera calibration and trajectory experiment on Modal."""
from pathlib import Path
import json
import modal
from modal_campaign import ROOT, base_image, volume

app = modal.App('sport-scans-overhead-20261025')
def mount(image):
    image=image.env({'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1'})
    for path in sorted((ROOT/'scripts').glob('*.py')):
        if path.name != 'modal_overhead.py':image=image.add_local_file(path,'/work/'+path.name)
    for path in sorted((ROOT/'data').glob('*.json')):
        image=image.add_local_file(path,'/work/'+path.name)
    return image
image=mount(base_image)
role_image=mount(base_image.pip_install('transformers==4.57.6'))

def run(task, variant=''):
    import overhead
    volume.reload()
    try:
        return getattr(overhead, task)(variant, volume)
    finally:
        volume.commit()

@app.function(image=image, cpu=8, memory=24576, timeout=7200,
              max_containers=2, scaledown_window=10, volumes={'/data': volume})
def cpu(task: str, variant: str=''):
    return run(task, variant)

@app.function(image=image, cpu=8, memory=49152, gpu='L40S', timeout=14400,
              max_containers=2, scaledown_window=10, volumes={'/data': volume})
def gpu(task: str, variant: str=''):
    return run(task, variant)

@app.function(image=role_image, cpu=8, memory=32768,
              gpu='L40S',timeout=7200,max_containers=1,scaledown_window=10,volumes={'/data':volume})
def role(variant: str=''):
    return run('roles',variant)

@app.local_entrypoint()
def main(task: str='prepare', variant: str='', use_gpu: bool=False):
    if task=='roles':result=role.remote(variant)
    elif task=='sam':result=gpu.with_options(gpu='H100').remote(task,variant)
    else:result = (gpu if use_gpu else cpu).remote(task, variant)
    dest=ROOT.parents[1]/'build/overhead-20261025';dest.mkdir(parents=True, exist_ok=True)
    (dest/(task+('-'+variant if variant else '')+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
