"""Reproduce the bounded ball-context experiment, without weights or raw video."""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hockey_tracking.ball_context import Config, infer

INPUT = ROOT/'web/data/overhead-20261025/replay.json'
OUT = ROOT/'web/data/ball-context-20261026'
VARIANTS = {
    'motion_only': Config(bridge=False, use_occlusion=False),
    'two_sided_motion': Config(use_occlusion=False),
    'context': Config(),
}


def masked_check(replay, config):
    """Hide detector anchors, not human truth. Entire target gap is removed."""
    anchors = {r['frame']: r['ball'] for r in replay['frames'] if r['ball']}
    trials = []
    # Deterministic, disjoint within each gap size; do not select on output error.
    for size in [1, 3, 5]:
        last = -100
        for first in sorted(anchors):
            if first <= last+size+2: continue
            if not all(f in anchors for f in range(first-2, first+size+2)): continue
            last = first
            hidden = set(range(first, first+size))
            masked = {f:a for f,a in anchors.items() if f not in hidden}
            result = infer(replay, masked, config)['frames']
            for f in sorted(hidden):
                p = result[f]; target = np.array(anchors[f]['pixel'])*[1920,1080]
                item = {'frame':f, 'gap_frames':size, 'estimated':p['state']!='unknown'}
                if item['estimated']:
                    delta = np.array(p['pixel'])*[1920,1080]-target
                    item.update(error_px=float(np.linalg.norm(delta)),
                                inside_model_ellipse=bool(delta@np.linalg.solve(p['pixel_covariance'],delta)<=4),
                                model_radius_px=p['model_radius_px'])
                trials.append(item)
    errors = [t['error_px'] for t in trials if t['estimated']]
    return {'label_type':'withheld_detector_centers_not_ground_truth',
            'target_frames':len(trials), 'estimated_frames':len(errors),
            'median_error_px':float(np.median(errors)) if errors else None,
            'p90_error_px':float(np.percentile(errors,90)) if errors else None,
            'inside_model_ellipse_fraction':float(np.mean([t['inside_model_ellipse'] for t in trials if t['estimated']])) if errors else None,
            'trials':trials}


def main():
    replay = json.loads(INPUT.read_text()); OUT.mkdir(parents=True,exist_ok=True)
    report = {'source_id':replay['source_id'],'start_s':replay['start_s'],
              'duration_s':300,'frames':4500,'fps':15,
              'input_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest(),
              'algorithm_sha256':hashlib.sha256((ROOT/'hockey_tracking/ball_context.py').read_bytes()).hexdigest(),
              'independent_ground_truth':False,'uncertainty_calibrated':False,
              'baseline_detected_frames':sum(bool(r['ball']) for r in replay['frames']),
              'variants':{}}
    for name, config in VARIANTS.items():
        result = infer(replay, config=config)
        report['variants'][name] = {
            'config':asdict(config), 'states':dict(Counter(r['state'] for r in result['frames'])),
            'reasons':dict(Counter(r['reason'] for r in result['frames'])),
            'masked_detector_check':masked_check(replay,config)}
        if name=='context':
            (OUT/'replay.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
            review=json.loads((Path(__file__).parent/'visual-review.json').read_text())
            checks=[]
            for item in review['reviewed_frames']:
                if item['pixel'] is None:continue
                estimate=result['frames'][item['frame']]
                error=np.array(estimate['pixel'])*[1920,1080]-item['pixel']
                checks.append({'frame':item['frame'],'error_px':float(np.linalg.norm(error)),
                               'annotation_tolerance_px':item['annotation_tolerance_px']})
            report['visual_review']={
                'label_type':'sparse_assistant_review_not_independent_ground_truth',
                'reviewed_frames':len(review['reviewed_frames']),
                'scored_frames':len(checks),'checks':checks,
                'review_sha256':hashlib.sha256((Path(__file__).parent/'visual-review.json').read_bytes()).hexdigest()}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({name:{k:v for k,v in row.items() if k not in ['config','masked_detector_check']} |
                    {'masked_detector_check':{k:v for k,v in row['masked_detector_check'].items() if k!='trials'}}
                    for name,row in report['variants'].items()},indent=2))


if __name__=='__main__': main()
