"""Exploratory global path-cover association on development clips only.

Hungarian assignment selects one predecessor/successor per fragment. Edges move
strictly forward in time, so the selected paths cannot contain cycles or two
co-visible fragments. Observations and missing intervals remain untouched.
"""
import copy
import itertools
import json
from pathlib import Path


def global_links(result, evidence, candidates, setting):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    key = 'cosine_'+setting['pooling']
    edges = [r for r in candidates if 0 < r['gap'] <= setting['gap'] and
             evidence[r['a']]['end'] < evidence[r['b']]['start'] and
             r['distance_heights'] <= setting['motion_base']+r['gap']*setting['motion_per_frame'] and
             min(r['support_a'], r['support_b']) >= setting['min_support'] and r[key] >= setting['cosine']]
    by_a, by_b = {}, {}
    for r in edges:
        by_a.setdefault(r['a'], []).append(r[key])
        by_b.setdefault(r['b'], []).append(r[key])
    keys = sorted(evidence)
    indices = {k: i for i, k in enumerate(keys)}
    n = len(keys)
    benefits = np.zeros((n, 2*n), dtype=np.float64)
    lookup = {}
    for r in edges:
        margins = [r[key]-max([v for v in group if v != r[key]], default=-1.)
                   for group in [by_a[r['a']], by_b[r['b']]]]
        # A zero margin permits global competition. Positive margin requires
        # local uniqueness first; tied candidates never pass positive margin.
        if setting['margin'] > 0 and (min(margins) < setting['margin'] or
                by_a[r['a']].count(r[key]) > 1 or by_b[r['b']].count(r[key]) > 1):
            continue
        value = r[key]-setting['cosine']-setting['gap_penalty']*r['gap']/15.
        if value <= 0:
            continue
        a, b = indices[r['a']], indices[r['b']]
        benefits[a, b] = value
        lookup[a, b] = r | {'benefit': value}
    accepted = []
    parent = {k: k for k in keys}
    if n:
        aa, bb = linear_sum_assignment(-benefits)
        for a, b in zip(aa, bb):
            if benefits[a, b] > 0:
                edge = lookup[a, b]
                assert evidence[edge['a']]['end'] < evidence[edge['b']]['start']
                parent[edge['b']] = edge['a']
                accepted.append(edge)
    def find(k):
        seen = set()
        while parent[k] != k:
            assert k not in seen
            seen.add(k)
            k = parent[k]
        return k
    output = copy.deepcopy(result)
    for row in output['frames']:
        for obj in row['objects']:
            obj['local_track_id'] = obj['track_id']
            if obj['track_id'] in parent:
                obj['track_id'] = find(obj['track_id'])
        assert len({o['track_id'] for o in row['objects']}) == len(row['objects'])
    output.update(linking=setting, links=accepted, labels_used_for_inference=False)
    return output


def grid():
    result = []
    for pool, cosine, margin, gap, penalty in itertools.product(
            ['mean', 'views'], [.75, .85, .92], [0., .03], [30, 90, 300], [0., .001]):
        result.append({'solver': 'global_assignment', 'pooling': pool, 'cosine': cosine,
                       'margin': margin, 'gap': gap, 'gap_penalty': penalty,
                       'motion_base': 1., 'motion_per_frame': .12, 'min_support': 10,
                       'id': f'global-{len(result):03}'})
    return result


def run(base, out, variant, volume, source='tracktrack'):
    from nbhl_tracking_features_v2 import load_features
    from nbhl_tracking_fourhour import replay, evaluate, tracklet_evidence, link_candidates, dump, sha
    from nbhl_tracking_reference_audit import corrected_report
    from nbhl_tracking_tune_v2 import objective, summarize
    cp = Path('/work/tracking-four-hour-reference-corrections-v1.json')
    corrections = {r['pair_id']: r for r in json.loads(cp.read_text())['corrections']}
    online = {'family': 'tracktrack', 'variant': 'none', 'buffer': 30, 'profile': 'matched', 'id': 'motion-control'}
    cache = []
    for meta, features in load_features(base, out, variant, cuts=True):
        if source == 'tracktrack':
            result = replay(meta, features, online)
        else:
            result = json.loads((out/'sam2'/source/(meta['summary']['clip']+'.json')).read_text())
        evidence = tracklet_evidence(meta, features, result)
        edges = link_candidates(meta, features, evidence)
        cache.append((meta, result, evidence, edges))
    rows = []
    for setting in [None]+grid():
        reports, count = [], 0
        for meta, result, evidence, edges in cache:
            linked = global_links(result, evidence, edges, setting) if setting else result
            reports.append(corrected_report(evaluate(meta, linked), corrections))
            count += len(linked.get('links', []))
        rows.append({'online': online if source == 'tracktrack' else {'family': 'sam2', 'size': source},
                     'linking': setting, 'counts': summarize(reports), 'clips': reports, 'links': count})
    chosen = max(rows, key=objective)
    dest = out/'global-links'/source/variant
    for meta, result, evidence, edges in cache:
        linked = global_links(result, evidence, edges, chosen['linking']) if chosen['linking'] else result
        dump(dest/'tracks'/(result['clip']+'.json'), linked)
    report = {'variant': variant, 'source': source, 'chosen': chosen, 'settings_tried': len(rows),
              'code_sha256': sha(__file__), 'corrections_sha256': sha(cp), 'final_test_used': False}
    dump(dest/'all-settings.json', {'results': rows})
    dump(dest/'completed.json', report)
    volume.commit()
    return report
