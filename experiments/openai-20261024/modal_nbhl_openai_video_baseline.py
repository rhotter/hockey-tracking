"""Prepare and run a full-task, full-frame OpenAI baseline on Modal."""
from pathlib import Path
import modal

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/data/ball-hockey/reliability-20261018/openai-video-baseline-20261024-v1')
app=modal.App('sport-scans-openai-video-baseline')
volume=modal.Volume.from_name('sport-scans-experiments')
image=(modal.Image.debian_slim(python_version='3.12').pip_install('pillow==11.3.0')
       .env({'PYTHONPATH':'/work'})
       .add_local_file(ROOT/'scripts/nbhl_openai_video_baseline.py','/work/nbhl_openai_video_baseline.py')
       .add_local_file(ROOT/'scripts/run_nbhl_openai_pilot.py','/work/run_nbhl_openai_pilot.py')
       .add_local_file(ROOT/'data/ball-hockey/openai-video-baseline-v1.json','/work/protocol.json'))


@app.function(image=image,cpu=2,memory=4096,timeout=7200,max_containers=1,volumes={'/data':volume})
def execute(task:str='run'):
    import json
    from PIL import Image
    from nbhl_openai_video_baseline import run,sample_indices,sha,atomic_json,SCHEMA,PROMPT
    volume.reload()
    protocol=json.loads(Path('/work/protocol.json').read_text())
    evidence=BASE/'evidence'; evidence.mkdir(parents=True,exist_ok=True)
    mp=evidence/'manifest.json'
    if mp.exists():
        assert sha(evidence/'protocol.json')==sha('/work/protocol.json')
        manifest=json.loads(mp.read_text())
    else:
        (evidence/'protocol.json').write_bytes(Path('/work/protocol.json').read_bytes())
        manifest={'protocol_sha256':sha('/work/protocol.json'), 'inference_inputs':'full frames and timestamps only',
                  'native_video':False, 'audio_included':False, 'reference_labels_supplied':False,
                  'prompt_sha256':__import__('hashlib').sha256(PROMPT.encode()).hexdigest(),
                  'schema':SCHEMA, 'clips':[]}
        for spec in protocol['clips']:
            inputs,outputs=sample_indices(spec)
            clip=spec|{'output_frames':outputs,'images':[]}
            for frame in inputs:
                source=Path('/data/sequences')/spec['clip']/'img1'/f'{frame+1:06d}.jpg'
                with Image.open(source) as im:
                    original=list(im.size)
                    im=im.convert('RGB')
                    if spec['max_width'] and im.width>spec['max_width']:
                        im=im.resize((spec['max_width'],round(im.height*spec['max_width']/im.width)),Image.Resampling.LANCZOS)
                    target=evidence/spec['clip']/f'{frame:06d}.jpg';target.parent.mkdir(exist_ok=True)
                    im.save(target,quality=92)
                    clip['images'].append({'frame':frame,'path':str(target.relative_to(evidence)),
                                           'sha256':sha(target),'source_sha256':sha(source),
                                           'source_dimensions':original,'width':im.width,'height':im.height})
            manifest['clips'].append(clip)
        atomic_json(mp,manifest); volume.commit()
    if task=='prepare':
        return {'protocol_sha256':sha(evidence/'protocol.json'),'manifest_sha256':sha(mp),
                'clips':[{'clip':c['clip'],'images':len(c['images']),'output_frames':c['output_frames']} for c in manifest['clips']]}
    assert task=='run'
    out=BASE/'run'
    try:
        ledger=run(evidence,out,protocol,manifest,commit=volume.commit)
        result={'stop_reason':ledger.get('stop_reason'),'requests':len(ledger['requests']),
                'valid':sum(r.get('parsed') is not None for r in ledger['requests'].values()),
                'cost_ceiling_usd':sum(r['cost_ceiling_usd'] for r in ledger['requests'].values()),
                'protocol_sha256':sha(evidence/'protocol.json'),'manifest_sha256':sha(mp)}
        atomic_json(out/'summary.json',result);volume.commit();return result
    except Exception as e:
        atomic_json(out/'error.json',{'type':type(e).__name__,'message':str(e)})
        volume.commit();raise


@app.local_entrypoint()
def main(task:str='run'):
    import json,sys
    sys.path.insert(0,str(ROOT/'scripts'))
    from run_nbhl_openai_pilot import read_key
    secret=modal.Secret.from_dict({'OPENAI_API_KEY':read_key(ROOT/'.env.local')})
    result=execute.with_options(secrets=[secret]).remote(task)
    target=ROOT/'notes/ball-hockey/openai-video-20261024'/('summary.json' if task=='run' else 'preparation.json')
    target.parent.mkdir(exist_ok=True,parents=True);target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
