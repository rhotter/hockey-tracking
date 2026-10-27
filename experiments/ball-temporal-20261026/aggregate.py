"""Combine disjoint fixed-time samples without changing their frozen policies."""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent


def main():
    reports = [json.loads((HERE/f'results/{s}-evaluation.json').read_text())
               for s in ['fresh', 'extra', 'final-check']]
    keys = ['visible', 'absent', 'unknown', 'correct', 'wrong', 'missing',
            'false_on_absent', 'correct_absent', 'outputs_on_unknown', 'tp', 'fp', 'fn']
    result = {'samples': 150, 'visible': 33, 'absent': 1, 'unknown_excluded': 116,
        'models': {}, 'decision': 'Retain YOLO. No new detector demonstrates a gain over it on these samples.',
        'scope': 'Six disjoint90second intervals in two recordings. First120samples informed follow-up experiments; final30times were reviewed before opening predictions. All thresholds and weights stayed frozen.',
        'limits': 'Sparse assistant labels, not independent human ground truth or full-game accuracy.'}
    for name in reports[0]['models']:
        result['models'][name] = {}
        for radius in ['at_12px', 'at_25px']:
            metrics = {k: sum(r['models'][name][radius][k] for r in reports) for k in keys}
            tp, fp, fn = [metrics[k] for k in ['tp', 'fp', 'fn']]
            metrics.update(f1=2*tp/(2*tp+fp+fn), visible_recall=tp/metrics['visible'],
                           precision_on_evaluable_frames=tp/(tp+fp) if tp+fp else 0)
            result['models'][name][radius] = metrics
    (HERE/'results/all-intervals.json').write_text(json.dumps(result, indent=2)+'\n')
    print({n:{k:d['at_25px'][k] for k in ['correct','wrong','missing']} for n,d in result['models'].items()})


if __name__ == '__main__': main()
