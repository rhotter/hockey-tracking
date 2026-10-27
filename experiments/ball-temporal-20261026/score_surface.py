"""Evaluate saved surface polygons; never derive a polygon from ball labels."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def inside_surface(pixel, surface, margin):
    if not surface['active']: return True
    polygon = surface['hull']; signs = []; distance = float('inf')
    x, y = pixel
    for a, b in zip(polygon, polygon[1:]+polygon[:1]):
        dx, dy = b[0]-a[0], b[1]-a[1]
        signs.append(dx*(y-a[1])-dy*(x-a[0]))
        t = max(0., min(1., ((x-a[0])*dx+(y-a[1])*dy)/max(1e-12, dx*dx+dy*dy)))
        distance = min(distance, math.hypot(x-a[0]-t*dx, y-a[1]-t*dy))
    return all(s >= -1e-8 for s in signs) or all(s <= 1e-8 for s in signs) or distance <= margin*surface['height']


def main():
    from evaluate import evaluate
    path = ROOT/'build/ball-temporal-20261026/surface-filter.json'; surfaces = read(path)
    source = HERE/'results/verifier-rerank-evaluation.json'; candidates = read(source)
    assert surfaces['candidate_source_sha256'] == candidates['source_sha256']
    assert surfaces['protocol'] == read(HERE/'surface-filter-protocol.json')
    bykey = {(r['clip'], r['frame']): r for r in surfaces['rows']}
    labels = sum([read(HERE/f'{scope}-reference.json')['rows'] for scope in ['fresh', 'extra', 'final-check']], [])
    audit = read(HERE/'transfer-output-audit.json')
    point = candidates['protocol']['operating_point']; models = {}
    for name, model in candidates['models'].items():
        for margin in surfaces['protocol']['margins_height_fraction']:
            frames = [{**r, 'candidates': [c for c in r['candidates'] if inside_surface(
                c['pixel'], bykey[(r['clip'], r['frame'])], margin)]} for r in model['frames']]
            outcomes = []
            for row in audit['rows']:
                surface = bykey[(row['clip'], row['frame'])]
                kept = inside_surface(row['pixel'], surface, margin)
                outcomes.append({**row, 'original_position_inside_surface': kept})
            key = f'{name}-margin{margin}'
            models[key] = {'verifier': name, 'margin': margin,
                **{f'at_{r}px': evaluate(frames, labels, point, r) for r in [12, 25]},
                'audit_original_positions': outcomes,
                'audit_counts': dict(Counter(r['verdict']+('_inside' if r['original_position_inside_surface'] else '_outside') for r in outcomes))}
    report = {**surfaces, 'evaluated_at': datetime.now(timezone.utc).isoformat(),
              'source_sha256': sha(path), 'candidate_evaluation_sha256': sha(source), 'models': models,
              'limits': 'Post-audit exploratory evaluation. Removing an audited wrong position does not prove a replacement is correct. Held/airborne balls can fall outside the surface.'}
    destination = HERE/'results/surface-filter-evaluation.json'; assert not destination.exists()
    destination.write_text(json.dumps(report, separators=(',', ':'))+'\n')
    print(json.dumps({name: {'localization': {k: v for k, v in row['at_25px'].items() if k != 'details'}, 'original_output_audit': row['audit_counts']} for name, row in models.items()}, indent=2))


if __name__ == '__main__': main()
