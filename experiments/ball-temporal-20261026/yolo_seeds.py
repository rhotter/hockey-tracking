"""Propose additional seeds from the retained YOLO detector, without labels."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD = ROOT / 'build/ball-temporal-20261026'
OLD = ROOT / 'build/ball-rebuild-20261026'
sys.path.insert(0, str(ROOT))
from hockey_tracking.ball_path import fuse_candidates


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = [OLD / f'{n}.json' for n in ['full1280', 'full1920', 'tiles640', 'verifier-v4', 'appearance-shape']]
    rows = fuse_candidates([read(p)['frames'] for p in paths[:3]])
    verification, shapes = [read(p)['frames'] for p in paths[3:]]
    baseline = read(ROOT / 'web/data/ball-sam-20261026/positions.json')['frames']
    hashes = read(BUILD / 'rgb-expanded/replay.json')['frames']
    existing = [s for batch in ['dense-full-gpu', 'dense-discovery-gpu', 'dense-detector-gpu']
                for s in read(BUILD / f'tracking-seeds-{batch}.json')['seeds']]
    proposals = []
    for f, row in enumerate(rows):
        assert len(row['detections']) == len(verification[f]['probabilities']) == len(shapes[f]['features'])
        if 466 <= f < 516 or baseline[f]['state'] == 'detected': continue
        for box, appearance, shape in zip(row['detections'], verification[f]['probabilities'], shapes[f]['features']):
            if box[4] < .4 or appearance < .5 or shape['extended_marking'] or shape['elongated_structure']: continue
            point = [(box[i] + box[i+2]) / 2 for i in range(2)]
            if any(abs(s['frame']-f) <= 2 and math.dist(s['pixel'], point) <= 16 for s in existing): continue
            proposals.append({'frame': f, 'pixel': point, 'appearance_score': appearance,
                'detector_score': box[4], 'source': 'detector_consensus', 'support': 'unclear',
                'evidence': 'Multiview YOLO candidate with crop verification and marking rejection; requires sequence verification before publication.',
                'source_sha256': hashes[f]['source_sha256']})
            break
    seeds = []
    for row in sorted(proposals, key=lambda r: (-r['detector_score']*r['appearance_score'], r['frame'])):
        if all(abs(row['frame']-s['frame']) >= 6 for s in seeds): seeds.append(row)
    seeds.sort(key=lambda s: s['frame'])
    protocol = {'created_at': datetime.now(timezone.utc).isoformat(), 'reference_labels_used': False,
        'selection': 'Fused YOLO score>=.4, crop verifier>=.5, no extended marking or elongated structure; first eligible candidate per frame; skip original visual anchors and all nearby existing proposals; rank score times verifier with6frame spacing.',
        'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in paths},
        'proposals': len(proposals), 'seeds': len(seeds),
        'limits': 'Exploratory development addition after comparing detector transfer; sequence verification is mandatory.'}
    result = {'batch': 'dense-yolo-gpu', 'pilot': False, 'seeds': seeds, 'samples': proposals,
        'half_window_frames': 6, 'crop_size': [768, 576], 'offload_state_to_cpu': False, 'selection_protocol': protocol}
    dest = BUILD / 'tracking-seeds-dense-yolo-gpu.json'
    assert not dest.exists(), 'Selection is sealed.'
    dest.write_text(json.dumps(result, indent=2)+'\n')
    (HERE / 'yolo-seed-protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print({'proposals': len(proposals), 'seeds': len(seeds)})


if __name__ == '__main__': main()
