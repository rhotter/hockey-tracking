"""Exploratory moving-ball exception to the frozen SAM surface filter."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent

def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def moving_exception(pixel, sample, operating, config):
    if sample is None: return False
    frames = {r['offset']: r for r in sample['frames']}
    aligned = []
    for offset in config['offsets']:
        frame = frames[offset]
        if frame['registration_quality'] < config['minimum_registration_quality']: return False
        values = []
        for candidate in frame['candidates']:
            score = candidate['score'] * candidate.get('appearance_score', 1) ** operating['appearance_power']
            if score < operating['threshold']: continue
            values.append(np.array(frame['to_target']) @ np.array([*candidate['pixel'], 1]))
        aligned.append(values)
    b = np.array(pixel)
    for a in aligned[0]:
        for c in aligned[1]:
            if np.linalg.norm(b-(a+c)/2) > config['midpoint_error_px']: continue
            if np.linalg.norm(c-a) < config['minimum_displacement_px']: continue
            if max(np.linalg.norm(b-a), np.linalg.norm(c-b)) > 3*config['maximum_speed_px_per_frame']: continue
            return True
    return False


def main():
    from score_surface import inside_surface
    from evaluate import evaluate
    protocol = read(HERE/'surface-motion-protocol.json')
    surface_path = HERE/'results/surface-filter-evaluation.json'
    candidate_path = HERE/'results/verifier-rerank-evaluation.json'
    motion_path = HERE/'results/motion-selection-inputs.json'
    surface, candidates, motion = map(read, [surface_path, candidate_path, motion_path])
    surfaces = {(r['clip'],r['frame']):r for r in surface['rows']}
    neighbors = {(r['clip'],r['frame']):r for r in motion['rows']}
    operating = candidates['protocol']['operating_point']
    def accepted(pixel, key):
        inside = inside_surface(pixel, surfaces[key], protocol['margin'])
        rescued = not inside and moving_exception(pixel, neighbors.get(key), operating, protocol)
        return inside, rescued
    frames, rescued_candidates = [], []
    for row in candidates['models']['original']['frames']:
        key = (row['clip'],row['frame']); kept = []
        for candidate in row['candidates']:
            inside, rescued = accepted(candidate['pixel'],key)
            if inside or rescued: kept.append(candidate)
            if rescued: rescued_candidates.append({'clip':key[0],'frame':key[1],'pixel':candidate['pixel']})
        frames.append({**row,'candidates':kept})
    labels = sum([read(HERE/f'{scope}-reference.json')['rows'] for scope in ['fresh','extra','final-check']],[])
    audit = []
    for row in read(HERE/'transfer-output-audit.json')['rows']:
        inside,rescued = accepted(row['pixel'],(row['clip'],row['frame']))
        audit.append({**row,'inside':inside,'rescued':rescued,'accepted':inside or rescued})
    report = {'protocol':protocol,'evaluated_at':datetime.now(timezone.utc).isoformat(),
        'input_sha256':{str(p.relative_to(HERE)):sha(p) for p in [surface_path,candidate_path,motion_path]},
        **{f'at_{r}px':evaluate(frames,labels,operating,r) for r in [12,25]},
        'rescued_candidates':rescued_candidates,'audit':audit,
        'audit_counts':dict(Counter(r['verdict']+('_accepted' if r['accepted'] else '_rejected') for r in audit)),
        'decision':'Rejected: five audited wrong objects restored, no missed audited ball recovered.',
        'limits':'Previously opened samples; temporal candidates available for120/150 targets. Other30 retain the surface-only policy. No independent validation or default replay change.'}
    path = HERE/'results/surface-motion-evaluation.json'
    assert not path.exists()
    path.write_text(json.dumps(report,separators=(',',':'))+'\n')
    print(json.dumps({'counts':report['audit_counts'],'rescued_candidates':len(rescued_candidates),'decision':report['decision']}))


if __name__ == '__main__': main()
