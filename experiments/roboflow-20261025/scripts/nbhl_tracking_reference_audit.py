"""Versioned reference corrections, preserving the original frozen evaluations."""
import copy
import json
from pathlib import Path
from nbhl_tracking_fourhour import sha, dump


def corrected_report(report, corrections):
    result = copy.deepcopy(report)
    counts = result['counts']
    for key in ['correct', 'wrong', 'unknown_same', 'same_total', 'different_total', 'missing']:
        counts[key] = 0
    for pair in result['pairs']:
        correction = corrections.get(pair['id'])
        if correction:
            assert pair['truth'] == correction['previous_truth']
            pair['previous_truth'] = pair['truth']
            pair['truth'] = correction['truth']
        truth, predicted = pair['truth'], pair['predicted_same']
        counts['same_total'] += truth == 'same'
        counts['different_total'] += truth == 'different'
        counts['missing'] += predicted is None
        if predicted:
            counts[{'same': 'correct', 'different': 'wrong', 'uncertain': 'unknown_same'}[truth]] += 1
    return result


def rescore(base, out, variant, volume):
    from nbhl_tracking_tune_v2 import objective, summarize
    from nbhl_tracking_features_v2 import load_features
    from nbhl_tracking_fourhour import replay, evaluate, tracklet_evidence, link_candidates, link_tracklets
    cp = Path('/work/tracking-four-hour-reference-corrections-v1.json')
    corrections = {r['pair_id']: r for r in json.loads(cp.read_text())['corrections']}
    source = out/'tuning-v2'/variant
    path = source/'all-settings.json'
    results = json.loads(path.read_text())['results']
    for row in results:
        row['original_counts'] = row['counts']
        row['clips'] = [corrected_report(report, corrections) for report in row['clips']]
        row['counts'] = summarize(row['clips'])
    chosen = max(results, key=objective)
    motion = max((r for r in results if r['online']['variant']=='none'), key=objective)
    dest = out/'tuning-v3-corrected'/variant
    for meta, features in load_features(base, out, variant, cuts=True):
        tracks = replay(meta, features, chosen['online'])
        if chosen['linking']:
            evidence = tracklet_evidence(meta, features, tracks)
            tracks = link_tracklets(tracks, evidence, link_candidates(meta, features, evidence), chosen['linking'])
        report = corrected_report(evaluate(meta, tracks), corrections)
        expected = next(r for r in chosen['clips'] if r['clip']==tracks['clip'])
        assert report['counts'] == expected['counts']
        dump(dest/'tracks'/(tracks['clip']+'.json'), tracks)
    result = {'variant': variant, 'chosen': chosen, 'motion_control': motion, 'settings_recounted': len(results),
              'corrections_sha256': sha(cp), 'original_all_settings_sha256': sha(path), 'code_sha256': sha(__file__),
              'selection': 'Same objective; versioned raw-video reference correction applied identically to every method. Original results preserved.',
              'final_test_used': False}
    dump(dest/'completed.json', result)
    volume.commit()
    return result
