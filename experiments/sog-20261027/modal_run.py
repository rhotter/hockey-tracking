"""Extract review sequences and run the bounded SOG detector on Modal."""
from pathlib import Path
import modal
HERE=Path(__file__).parent
ROOT=HERE.parents[1] if modal.is_local() else Path('/work')
app=modal.App('sport-scans-sog-20261027')
volume=modal.Volume.from_name('sport-scans-experiments')
image=(modal.Image.debian_slim(python_version='3.12').pip_install('pillow==11.3.0')
       .add_local_file(HERE/'protocol.json','/work/protocol.json')
       .add_local_file(HERE/'worker.py','/work/worker.py')
       .add_local_file(HERE/'context_worker.py','/work/context_worker.py')
       .add_local_file(HERE/'context-protocol.json','/work/context-protocol.json')
       .add_local_file(HERE/'recovery_worker.py','/work/recovery_worker.py')
       .add_local_file(HERE/'recovery-protocol.json','/work/recovery-protocol.json')
       .add_local_file(ROOT/'experiments/openai-20261024/run_nbhl_openai_pilot.py','/work/client.py'))
@app.function(image=image,cpu=4,memory=8192,timeout=7200,volumes={'/data':volume},max_containers=1)
def execute(task):
    import sys,json
    sys.path.insert(0,'/work');import worker
    volume.reload();protocol=json.loads(Path('/work/protocol.json').read_text())
    if task=='context':
        import context_worker
        result=context_worker.run(protocol,json.loads(Path('/work/context-protocol.json').read_text()),volume)
    elif task=='context-recovery':
        import recovery_worker
        result=recovery_worker.run(protocol,json.loads(Path('/work/recovery-protocol.json').read_text()),volume)
    elif task=='review':result=worker.review_sheets(protocol)
    elif task=='event-review':result=worker.review_sheets(protocol,json.loads((worker.OUT/'review-windows.json').read_text())['windows'])
    else:result=worker.run(protocol,task,volume)
    volume.commit();return result
@app.local_entrypoint()
def main(task:str='review',env_file:str=''):
    if task in ('scan','verify','context','context-recovery'):
        import sys
        sys.path.insert(0,str(ROOT/'experiments/openai-20261024'))
        from run_nbhl_openai_pilot import read_key
        secret=modal.Secret.from_dict({'OPENAI_API_KEY':read_key(Path(env_file))})
        print(execute.with_options(secrets=[secret]).remote(task))
    else:print(execute.remote(task))
