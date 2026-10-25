"""Detection metrics for complete reviewed frames, with no sparse-label shortcut."""
from __future__ import annotations


def iou(a, b):
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection/union if union > 0 else 0.


def match(rows, overlap=.5):
    """Confidence-ordered one-to-one matching, including unmatched duplicates."""
    ranked, total = [], 0
    for row in rows:
        targets = row['players']; total += len(targets); used = set()
        for prediction in sorted(row['predictions'], key=lambda p: -p['confidence']):
            candidates = [(iou(prediction['bbox'], target['bbox']), j) for j, target in enumerate(targets) if j not in used]
            score, j = max(candidates, default=(0., -1))
            hit = j >= 0 and score >= overlap
            if hit: used.add(j)
            ranked.append({'id': row['id'], 'confidence': prediction['confidence'], 'hit': hit,
                'target_index': j if hit else None, 'role': targets[j]['role'] if hit else None,
                'iou': score, 'bbox': prediction['bbox']})
    return sorted(ranked, key=lambda p: -p['confidence']), total


def summarize(rows, threshold=.2, overlap=.5):
    ranked, total = match(rows, overlap)
    selected = [p for p in ranked if p['confidence'] >= threshold]
    tp = sum(p['hit'] for p in selected); fp = len(selected)-tp; fn = total-tp
    precision = tp/(tp+fp) if tp+fp else None
    recall = tp/total if total else None
    goalies = sum(p['role']=='goalie' for r in rows for p in r['players'])
    goalie_hits = sum(p['role']=='goalie' for p in selected)
    return {'confidence_threshold': threshold, 'iou_threshold': overlap, 'frames': len(rows),
        'targets': total, 'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall,
        'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        'false_positives_per_frame': fp/len(rows) if rows else None,
        'goalies': goalies, 'goalie_hits': goalie_hits,
        'goalie_recall': goalie_hits/goalies if goalies else None}


def average_precision(rows, overlap=.5):
    ranked, total = match(rows, overlap)
    if not total: return None
    tp = 0; curve = []
    for i, prediction in enumerate(ranked):
        tp += prediction['hit']; curve.append((tp/total, tp/(i+1)))
    return sum(max((p for r, p in curve if r >= level/100), default=0.) for level in range(101))/101


def evaluate(rows):
    curve = [summarize(rows, value/100) for value in range(5, 96, 5)]
    best = max(curve, key=lambda r: (r['f1'], r['confidence_threshold']))
    aps = [average_precision(rows, .5+i*.05) for i in range(10)]
    safe = [r for r in curve if r['precision'] is not None and r['precision'] >= .95 and r['tp'] > 0]
    return {'ap50': aps[0], 'interpolated_ap50_95': sum(aps)/len(aps) if all(x is not None for x in aps) else None,
        'ap30_diagnostic': average_precision(rows, .3), 'at_020': summarize(rows, .2),
        'validation_selected_best_f1': best,
        'validation_selected_precision_095': max(safe, key=lambda r: r['recall']) if safe else None,
        'threshold_curve': curve,
        'metric_definition': 'Single-class visible floor-player boxes. Confidence-ordered one-to-one matching; 101-point interpolated AP. Not official COCO evaluation; no area/crowd rules. Assistant-reviewed labels, not independent human ground truth.'}
