"""Run the context layer and export native-resolution review crops on Modal."""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/work')
app = modal.App('sport-scans-ball-context-20261026')
volume = modal.Volume.from_name('sport-scans-experiments')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('numpy==1.26.4', 'pillow==11.3.0')
         .add_local_dir(ROOT/'hockey_tracking', '/work/hockey_tracking', ignore=['__pycache__'])
         .add_local_file(ROOT/'web/data/overhead-20261025/replay.json', '/work/input.json'))


@app.function(image=image, cpu=2, memory=4096, timeout=300,
              max_containers=1, volumes={'/data':volume})
def run():
    import hashlib
    import json
    import sys
    import tarfile
    import numpy as np
    from PIL import Image
    sys.path.insert(0, '/work')
    from hockey_tracking.ball_context import infer
    volume.reload()
    source = Path('/data/ball-hockey/reliability-20261018/overhead-20261025-v1')
    dest = Path('/data/ball-hockey/reliability-20261018/ball-context-20261026-v1')
    dest.mkdir(exist_ok=True)
    result = infer(json.loads(Path('/work/input.json').read_text()))
    raw = json.dumps(result,separators=(',',':'))+'\n'
    (dest/'replay.json').write_text(raw)
    inferred = [r for r in result['frames'] if r['state']=='inferred']
    selected = sorted(set([inferred[i]['frame'] for i in np.linspace(0,len(inferred)-1,12,dtype=int)]
                          +[r['frame'] for r in inferred if r['possible_occluders']]))
    manifest = []
    for f in selected:
        frame = Image.open(source/'frames'/f'{f+1:06d}.jpg')
        x,y = np.array(result['frames'][f]['pixel'])*[1920,1080]
        left,top = int(np.clip(x-160,0,1600)),int(np.clip(y-100,0,880))
        frame.crop((left,top,left+320,top+200)).save(dest/f'crop-{f:04d}.png')
        manifest.append({'frame':f,'origin_native':[left,top]})
    (dest/'review.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with tarfile.open(dest/'review.tar.gz','w:gz') as tar:
        for f in selected: tar.add(dest/f'crop-{f:04d}.png', arcname=f'crop-{f:04d}.png')
        tar.add(dest/'review.json',arcname='review.json')
    volume.commit()
    return {'replay_sha256':hashlib.sha256(raw.encode()).hexdigest(),
            'review_frames':selected,'archive':str(dest/'review.tar.gz')}


@app.local_entrypoint()
def main():
    import json
    result = run.remote()
    dest = ROOT/'build/ball-context-20261026'; dest.mkdir(parents=True,exist_ok=True)
    (dest/'modal.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
