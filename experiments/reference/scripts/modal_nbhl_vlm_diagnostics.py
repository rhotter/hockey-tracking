"""Post-inference audit; never changes VLM inputs, linking, or reference labels."""
import json
from pathlib import Path
import modal
from modal_campaign import ROOT, volume
from modal_nbhl_vlm_reconnect import prep_image

app = modal.App('sport-scans-nbhl-vlm-audit')
image = prep_image.add_local_file(ROOT/'scripts/modal_nbhl_vlm_reconnect.py', '/work/modal_nbhl_vlm_reconnect.py')
image = image.add_local_file(ROOT/'scripts/nbhl_vlm_number_normalization.py', '/work/nbhl_vlm_number_normalization.py')
image = image.add_local_file(ROOT/'data/ball-hockey/vlm-reconnect-number-amendment-v1.json', '/work/vlm-number-amendment.json')


@app.function(image=image, cpu=4, memory=16384, timeout=1800, volumes={'/data': volume})
def extend(phase: str, model: str):
    from nbhl_vlm_number_normalization import extend
    volume.reload()
    result = extend(phase, model)
    volume.commit()
    return result


@app.function(image=image, cpu=4, memory=16384, timeout=1800, volumes={'/data': volume})
def audit(phase: str, model: str):
    from collections import Counter
    from nbhl_vlm_reconnect import OUT, dump, sha, number_consensus, decision, strict_same
    volume.reload()
    folder = OUT/phase/model
    report = json.loads((folder/'evaluation-extended.json').read_text())
    manifest = json.loads((OUT/phase/'evidence.json').read_text())
    pred = json.loads((folder/'predictions.json').read_text())
    responses = {r['id']: r for r in pred['results']}
    changes = []
    for base in [r for r in report['clips'] if r['policy']=='osnet_full']:
        bp = {p['id']: p for p in base['pairs']}
        for alternative in [r for r in report['clips'] if r['clip']==base['clip'] and r['policy']!='osnet_full']:
            changed = []
            for p in alternative['pairs']:
                b = bp[p['id']]
                assert p['truth']==b['truth']
                if p['predicted_same'] != b['predicted_same']:
                    changed.append({'id':p['id'], 'a':p['a'], 'b':p['b'], 'truth':p['truth'],
                                    'baseline_same':b['predicted_same'], 'candidate_same':p['predicted_same']})
            changes.append({'clip':base['clip'], 'policy':alternative['policy'], 'changes':changed})
    numbers = []
    for clip in manifest['clips']:
        for tid, track in clip['tracks'].items():
            number = number_consensus(responses.get(clip['clip']+'/number/'+tid), len(track['images']))
            if number is not None:
                numbers.append({'clip':clip['clip'], 'track':tid, 'number':number})
    output = {'phase':phase, 'model':model, 'evaluation_sha256':sha(folder/'evaluation-extended.json'),
              'finish_reasons':dict(Counter(r['finish_reason'] for r in pred['results'])),
              'invalid_json':sum(not isinstance(r.get('parsed'),dict) for r in pred['results']),
              'readable_tracks':numbers, 'track_count':sum(len(c['tracks']) for c in manifest['clips']),
              'strict_same':{task:sum(strict_same(r) for r in pred['results'] if r['task']==task) for task in ['single','multi','reverse']},
              'changes':changes}
    dump(folder/'diagnostics.json', output)
    volume.commit()
    return output


@app.local_entrypoint()
def main(phase: str='development', model: str='8b', task: str='audit'):
    assert task in ['audit','extend']
    result = (extend if task=='extend' else audit).remote(phase, model)
    path = ROOT/'notes/ball-hockey'/f'vlm-reconnect-{task}-{phase}-{model}.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['changes','readable_tracks']}))
