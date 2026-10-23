"""Offline temporal path assignment; extracted unchanged from the measured linker."""
import copy

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
