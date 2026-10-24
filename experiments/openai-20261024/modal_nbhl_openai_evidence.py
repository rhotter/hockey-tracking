"""Export the existing label-free Qwen evidence for a matched OpenAI pilot."""
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App('sport-scans-nbhl-openai-evidence')
volume = modal.Volume.from_name('sport-scans-experiments')
image = (modal.Image.debian_slim(python_version='3.12')
         .env({'PYTHONPATH': '/work'})
         .add_local_file(ROOT/'scripts/nbhl_vlm_reconnect.py', '/work/nbhl_vlm_reconnect.py')
         .add_local_file(ROOT/'data/ball-hockey/openai-reconnect-pilot-v1.json', '/work/protocol.json'))
BASE = Path('/data/ball-hockey/reliability-20261018')
OUT = BASE/'openai-reconnect-20261024-v1'


@app.function(image=image, cpu=1, memory=2048, timeout=600, volumes={'/data': volume})
def export():
    import hashlib
    import io
    import tarfile
    from nbhl_vlm_reconnect import requests, sha

    volume.reload()
    protocol = json.loads(Path('/work/protocol.json').read_text())
    source = BASE/'vlm-reconnect-20261019-v1/development/evidence.json'
    evidence = json.loads(source.read_text())
    assert evidence['selection_uses_reference_labels'] is False
    assert len(evidence['clips']) == 9
    selected = []
    for clip in evidence['clips']:
        ranked = sorted(clip['selected_edges'], key=lambda e: hashlib.sha256(
            (protocol['selection_salt'] + e['id']).encode()).hexdigest())
        selected.append(clip | {'selected_edges': ranked[:protocol['edges_per_clip']]})
    rows = requests(evidence | {'clips': selected}, protocol['tasks'])
    for row in rows:
        assert row['task'] in ['multi', 'reverse']
        row['images'] = [{'path': 'crops/' + im['sha256'] + '.jpg',
                          'sha256': im['sha256']} for im in row['images']]
    manifest = {'scope': 'NBHL ball hockey only', 'phase': 'development-pilot',
                'protocol_sha256': sha('/work/protocol.json'),
                'evidence_sha256': sha(source), 'selection_uses_reference_labels': False,
                'clips': [{'clip': c['clip'], 'edges': len(c['selected_edges'])} for c in selected],
                'requests': rows}
    images = {im['sha256']: Path(im['path']) for c in selected for t in c['tracks'].values()
              for im in t['images']}
    used = {im['sha256'] for r in rows for im in r['images']}
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT/'pilot-evidence.tar.gz'
    with tarfile.open(target, 'w:gz') as bundle:
        for name, content in [('manifest.json', (json.dumps(manifest, indent=2) + '\n').encode()),
                              ('protocol.json', Path('/work/protocol.json').read_bytes())]:
            info = tarfile.TarInfo(name); info.size = len(content); info.mtime = 0
            bundle.addfile(info, io.BytesIO(content))
        for digest in sorted(used):
            path = images[digest]
            assert sha(path) == digest
            bundle.add(path, arcname='crops/' + digest + '.jpg', recursive=False)
    summary = {'path': str(target), 'sha256': sha(target), 'bytes': target.stat().st_size,
               'requests_per_model': len(rows), 'edges': len(rows)//2, 'images': len(used),
               'protocol_sha256': manifest['protocol_sha256'], 'evidence_sha256': sha(source),
               'clips': manifest['clips'], 'api_requests_made': 0}
    (OUT/'evidence-export.json').write_text(json.dumps(summary, indent=2) + '\n')
    volume.commit()
    return summary


@app.local_entrypoint()
def main(task: str = 'export'):
    assert task in ['export', 'baselines']
    result = export.remote() if task == 'export' else baselines.remote()
    target = ROOT/'notes/ball-hockey/openai-20261024'/('evidence-export.json' if task == 'export' else 'qwen-baselines.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


@app.function(image=image, cpu=1, memory=2048, timeout=180, volumes={'/data': volume})
def baselines():
    import tarfile
    from collections import Counter
    from nbhl_vlm_reconnect import sha, decision, strict_same
    volume.reload()
    with tarfile.open(OUT/'pilot-evidence.tar.gz') as bundle:
        manifest = json.load(bundle.extractfile('manifest.json'))
    ids = {r['id'] for r in manifest['requests']}
    output = {'selection_uses_reference_labels': False,
              'protocol_sha256': manifest['protocol_sha256'],
              'evidence_sha256': manifest['evidence_sha256'], 'edges': len(ids)//2, 'models': []}
    for model in ['8b', '235b']:
        path = BASE/'vlm-reconnect-20261019-v1/development'/model/'predictions.json'
        predictions = json.loads(path.read_text())
        assert predictions['evidence_sha256'] == manifest['evidence_sha256']
        rows = [r for r in predictions['results'] if r['id'] in ids]
        assert {r['id'] for r in rows} == ids and len(rows) == len(ids)
        by_id = {r['id']: r for r in rows}
        forward = [r for r in rows if r['task'] == 'multi']
        table = Counter(decision(r)+' / '+decision(by_id[r['edge_id']+'/reverse']) for r in forward)
        output['models'].append({'model': model, 'model_spec': predictions['model'],
                                 'predictions_sha256': sha(path), 'requests': len(rows),
                                 'forward_decisions': dict(Counter(decision(r) for r in forward)),
                                 'forward_strict_same': sum(strict_same(r) for r in forward),
                                 'order_consistency': dict(table),
                                 'order_agreement': sum(n for label,n in table.items() if len(set(label.split(' / '))) == 1),
                                 'responses': [{k:r.get(k) for k in ['id','task','edge_id','clip','parsed']} for r in rows]})
    output['limits'] = 'Matched development pilot; order consistency and model agreement are not accuracy. No new inference.'
    (OUT/'qwen-pilot-baselines.json').write_text(json.dumps(output, indent=2)+'\n')
    volume.commit()
    return output
