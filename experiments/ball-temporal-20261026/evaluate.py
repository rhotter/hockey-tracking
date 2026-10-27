"""One-ball localization evaluation with development-only operating points.

Unknown reference frames are neither positive nor negative. A wrong location on
a visible frame contributes both a false positive and a false negative. These
sparse assistant labels do not establish full-video tracking accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD = ROOT / 'build/ball-temporal-20261026'
THRESHOLDS = (.01, .025, .05, .1, .2, .3, .5, .7, .9)
POWERS = (0., .5, 1., 2.)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def top_candidate(frame, threshold, appearance_power):
    candidates = frame.get('candidates', [])
    if not candidates:
        return None
    def score(candidate):
        return candidate['score'] * candidate.get('appearance_score', 1.) ** appearance_power
    candidate = max(candidates, key=score)
    return candidate if score(candidate) >= threshold else None


def evaluate(frames, labels, operating_point, radius=12):
    by_key = {(r['clip'], r['frame']): r for r in frames}
    assert len(by_key) == len(frames), 'Duplicate prediction keys'
    counts = dict(visible=0, absent=0, unknown=0, correct=0, wrong=0,
                  missing=0, false_on_absent=0, correct_absent=0,
                  outputs_on_unknown=0)
    details, errors = [], []
    for label in labels:
        key = (label['clip'], label['frame'])
        assert key in by_key, f'Missing inference row: {key}'
        frame = by_key[key]
        expected_hash = label.get('source_sha256') or label.get('sha256')
        if expected_hash:
            assert frame['source_sha256'] == expected_hash, f'Image mismatch: {key}'
        prediction = top_candidate(frame, **operating_point)
        visibility = label.get('visibility', 'visible')
        target = label.get('pixel') or (label.get('ball') or {}).get('center')
        error = None
        if visibility == 'visible':
            assert target is not None
            counts['visible'] += 1
            error = math.dist(prediction['pixel'], target) if prediction else None
            outcome = 'missing' if error is None else ('correct' if error <= radius else 'wrong')
            counts[outcome] += 1
            if error is not None:
                errors.append(error)
        elif visibility == 'absent':
            counts['absent'] += 1
            outcome = 'false_on_absent' if prediction else 'correct_absent'
            counts[outcome] += 1
        else:
            counts['unknown'] += 1
            counts['outputs_on_unknown'] += int(prediction is not None)
            outcome = 'unknown_excluded'
        details.append({'clip': key[0], 'frame': key[1], 'outcome': outcome,
                        'reference_pixel': target, 'pixel': prediction['pixel'] if prediction else None,
                        'error_px': error})
    tp = counts['correct']; fp = counts['wrong'] + counts['false_on_absent']
    fn = counts['wrong'] + counts['missing']
    return {**counts, 'radius_px': radius, 'tp': tp, 'fp': fp, 'fn': fn,
            'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            'visible_recall': tp/counts['visible'] if counts['visible'] else None,
            'precision_on_evaluable_frames': tp/(tp+fp) if tp+fp else None,
            'median_error_all_outputs_px': statistics.median(errors) if errors else None,
            'details': details}


def development_selection(variants):
    labels = [r for r in read(ROOT/'experiments/ball-rebuild-20261026/regression-labels.json')['rows']
              if r['clip'] == 'nj-2400']
    assert len(labels) == 11
    selected = {}
    for variant in variants:
        path = BUILD/variant/'regression.json'; predictions = read(path)
        assert predictions['complete']
        trials = []
        for power in POWERS:
            for threshold in THRESHOLDS:
                point = {'threshold': threshold, 'appearance_power': power}
                metrics = evaluate(predictions['frames'], labels, point)
                trials.append({'operating_point': point, **{k: v for k, v in metrics.items() if k != 'details'}})
        # Conservative tie break: fewer false positives, then stronger minimum
        # evidence; prefer less appearance reweighting when otherwise identical.
        best = max(trials, key=lambda t: (t['f1'], -t['fp'], t['operating_point']['threshold'],
                                          -t['operating_point']['appearance_power']))
        selected[variant] = {'operating_point': best['operating_point'], 'development': best,
                             'predictions_sha256': digest(path), 'weights_sha256': predictions['weights_sha256'],
                             'trials': trials}
    return {'frozen_at': datetime.now(timezone.utc).isoformat(), 'fresh_scores_opened': False,
            'selection_split': '11 reused NJ2400 frames only, native full-frame localization at 12px',
            'tie_break': 'Max F1; fewer FP; highest threshold; lowest appearance power.',
            'limits': 'Small correlated development sample; operating points are not calibrated probabilities.',
            'variants': selected}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['select', 'score'])
    parser.add_argument('--variants', required=True)
    parser.add_argument('--scope', default='regression', choices=['regression', 'fresh', 'extra', 'final-check'])
    args = parser.parse_args()
    variants = args.variants.split(',')
    result_dir = HERE/'results'; result_dir.mkdir(exist_ok=True)
    selection_path = result_dir/'development-selection.json'
    if args.task == 'select':
        assert not selection_path.exists(), 'Selection is sealed; create an explicit amendment instead of overwriting.'
        result = development_selection(variants)
        selection_path.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({v: r['development'] for v, r in result['variants'].items()}, indent=2))
        return
    selection = read(selection_path)
    label_path = (ROOT/'experiments/ball-rebuild-20261026/regression-labels.json') if args.scope == 'regression' else HERE/(args.scope+'-reference.json')
    labels = read(label_path)['rows']
    reports = {}
    for variant in variants:
        path = BUILD/variant/(args.scope+'.json')
        if not path.exists() and args.scope != 'regression':
            path = BUILD/variant/(args.scope+'-review.json')
        predictions = read(path)
        assert predictions['complete']
        frozen = selection['variants'][variant]
        assert predictions['weights_sha256'] == frozen['weights_sha256']
        reports[variant] = {'predictions_file': str(path.relative_to(ROOT)), 'predictions_sha256': digest(path), 'weights_sha256': predictions['weights_sha256'],
                            'operating_point': frozen['operating_point'],
                            'at_12px': evaluate(predictions['frames'], labels, frozen['operating_point'], 12),
                            'at_25px': evaluate(predictions['frames'], labels, frozen['operating_point'], 25)}
    result = {'evaluated_at': datetime.now(timezone.utc).isoformat(), 'scope': args.scope,
              'selection_sha256': digest(selection_path), 'reference_sha256': digest(label_path),
              'limits': 'Sparse assistant-reviewed localization; unknown samples excluded; not full-video accuracy.',
              'models': reports}
    destination = result_dir/(args.scope+'-evaluation.json')
    assert not destination.exists(), 'Do not overwrite a sealed evaluation.'
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({v: {k: x for k, x in r['at_25px'].items() if k != 'details'} for v, r in reports.items()}, indent=2))


if __name__ == '__main__':
    main()
