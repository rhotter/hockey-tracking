"""Fixed short-trajectory reranking; reference positions never enter selection."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import numpy as np
from evaluate import evaluate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def rank_candidates(sample, operating_point, config, maximum_candidates=8):
    observations = {}
    qualities = {}
    for frame in sample['frames']:
        candidates = []
        for row in frame['candidates']:
            score = row['score'] * row.get('appearance_score', 1.) ** operating_point['appearance_power']
            if score < operating_point['threshold']: continue
            point = np.array(frame['to_target']) @ np.array([*row['pixel'], 1.])
            candidates.append({'pixel': row['pixel'], 'aligned': point, 'score': score})
        observations[frame['offset']] = sorted(candidates, key=lambda r: -r['score'])[:maximum_candidates]
        qualities[frame['offset']] = frame['registration_quality']
    center = observations[0]
    if not center: return None, {'reason': 'no_target_candidate'}
    if not observations[-3] or not observations[3] or min(qualities[-3], qualities[3]) < .35:
        return center[0]['pixel'], {'reason': 'no_registered_bracket'}
    sigma = config['sigma_px']
    def kernel(error): return math.exp(-.5*(error/sigma)**2)
    def continuation(offset, expected):
        if qualities[offset] < .35: return 0.
        return max((r['score'] * kernel(np.linalg.norm(r['aligned']-expected))
                    for r in observations[offset]), default=0.)
    ranked = []
    for candidate in center:
        support = 0.
        for previous in observations[-3]:
            for following in observations[3]:
                a, b, c = previous['aligned'], candidate['aligned'], following['aligned']
                if max(np.linalg.norm(b-a), np.linalg.norm(c-b)) > 3*140: continue
                continuity = kernel(np.linalg.norm(b-(a+c)/2))
                adjacent = math.sqrt(previous['score']*following['score'])
                outer = math.sqrt(continuation(-6, 2*a-b)*continuation(6, 2*c-b))
                support = max(support, continuity*(.75*adjacent+.25*outer))
        score = candidate['score'] + config['neighbor_weight']*support
        ranked.append((score, candidate['score'], candidate['pixel'], support))
    best = max(ranked, key=lambda r: r[:2])
    return best[2], {'reason': 'motion_reranking', 'ranking_score': best[0],
                     'detector_score': best[1], 'temporal_support': best[3]}


def main():
    path = ROOT/'build/ball-temporal-20261026/motion-selection-inputs.json'
    inputs = read(path); protocol = read(HERE/'motion-selection-protocol.json')
    assert inputs['protocol'] == protocol
    point = read(HERE/'results/development-selection.json')['variants']['yolo-full1920']['operating_point']
    labels = read(HERE/'fresh-reference.json')['rows'] + read(HERE/'extra-reference.json')['rows']
    models = {}
    for config in protocol['variants']:
        frames, decisions = [], []
        for sample in inputs['rows']:
            pixel, decision = rank_candidates(sample, point, config, protocol['maximum_candidates'])
            center = next(r for r in sample['frames'] if r['offset'] == 0)
            frames.append({'clip': sample['clip'], 'frame': sample['frame'],
                           'source_sha256': center['source_sha256'],
                           'candidates': [{'pixel': pixel, 'score': 1.}] if pixel is not None else []})
            decisions.append({'clip': sample['clip'], 'frame': sample['frame'], **decision})
        models[config['name']] = {'config': config, 'decisions': decisions,
            **{f'at_{r}px': evaluate(frames, labels, {'threshold': .5, 'appearance_power': 0}, r)
               for r in [12, 25]}}
    result = {'evaluated_at': datetime.now(timezone.utc).isoformat(), 'protocol': protocol,
              'inputs_sha256': sha(path), 'models': models,
              'limits': 'Already opened sparse transfer checks; all variants reported; no labels enter candidate ranking.'}
    destination = HERE/'results/motion-selection-evaluation.json'
    assert not destination.exists()
    destination.write_text(json.dumps(result, indent=2)+'\n')
    # Preserve small numeric inputs so anyone can reproduce reranking.
    (HERE/'results/motion-selection-inputs.json').write_bytes(path.read_bytes())
    print(json.dumps({name: {k: v for k, v in model['at_25px'].items() if k != 'details'}
                      for name, model in models.items()}, indent=2))


if __name__ == '__main__': main()
