"""Controlled NBHL tracker comparison and conservative offline tracklet linking.

All algorithms consume identical cached detections. References are evaluation-only;
neither online replay nor offline linking reads identity labels.
"""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def candidates(family):
    """Predeclared bounded sweep, with native defaults and matched thresholds."""
    rows = []
    variants = ['none', 'generic', 'extended_full'] if family in ['botsort', 'deepocsort', 'tracktrack'] else ['none']
    for variant in variants:
        for buffer in [30, 90, 180]:
            rows.append({'family': family, 'variant': variant, 'buffer': buffer, 'profile': 'matched'})
        rows.append({'family': family, 'variant': variant, 'buffer': 30, 'profile': 'native'})
        if family == 'tracktrack':
            for weight in [.25, .75]:
                for relaxed in [0., .85]:
                    if variant != 'none':
                        rows.append({'family': family, 'variant': variant, 'buffer': 90, 'profile': 'matched',
                                     'overrides': {'iou_weight': 1-weight, 'reid_weight': weight, 'lost_match_thr': relaxed}})
        if family == 'deepocsort' and variant != 'none':
            for proximity in [.1, .3]:
                for appearance in [.75, .9]:
                    rows.append({'family': family, 'variant': variant, 'buffer': 90, 'profile': 'matched',
                                 'overrides': {'proximity_thresh': proximity, 'appearance_thresh': appearance}})
    for i, row in enumerate(rows):
        row['id'] = f'{family}-{i:03}'
    return rows


def config_for(meta, setting):
    import ultralytics
    import yaml
    cfg = yaml.safe_load((Path(ultralytics.__file__).parent/'cfg/trackers'/(setting['family']+'.yaml')).read_text())
    if setting['profile'] == 'matched':
        for key in ['track_high_thresh', 'track_low_thresh', 'new_track_thresh', 'fuse_score', 'match_thresh']:
            if key in meta['config']:
                cfg[key] = meta['config'][key]
    cfg.update(setting.get('overrides', {}))
    cfg.update(track_buffer=setting['buffer'], with_reid=False)
    return cfg


def replay(meta, features, setting):
    import numpy as np
    from types import SimpleNamespace
    from scipy.optimize import linear_sum_assignment
    from ultralytics.trackers import BOTSORT, BYTETracker, OCSORT, DeepOCSORT, TRACKTRACK, FASTTracker
    from ultralytics.engine.results import Boxes
    from nbhl_detection_metrics import iou
    classes = {'botsort': BOTSORT, 'bytetrack': BYTETracker, 'ocsort': OCSORT,
               'deepocsort': DeepOCSORT, 'tracktrack': TRACKTRACK, 'fasttrack': FASTTracker}
    current = {}
    calls = 0
    def encode(img, boxes):
        nonlocal calls
        xywh = np.asarray(boxes)[:, :4]
        xyxy = np.column_stack([xywh[:, 0]-xywh[:, 2]/2, xywh[:, 1]-xywh[:, 3]/2,
                               xywh[:, 0]+xywh[:, 2]/2, xywh[:, 1]+xywh[:, 3]/2])
        distance = np.max(np.abs(xyxy[:, None, :]-current['arr'][None, :, :4]), axis=2)
        indices = distance.argmin(1)
        assert np.all(distance[np.arange(len(indices)), indices] < .002)
        z = current['z'][indices]
        assert np.isfinite(z).all() and np.allclose(np.linalg.norm(z, axis=1), 1., atol=1e-4)
        calls += 1
        return z
    cfg = config_for(meta, setting)
    def make():
        tr = classes[setting['family']](SimpleNamespace(**cfg))
        if setting['variant'] != 'none':
            tr.args.with_reid = True
            tr.args.model = 'cached_nbhl_reid'
            tr.encoder = encode
        if hasattr(tr, 'gmc'):
            # The cached warp was computed from real video, never this blank frame.
            tr.gmc = SimpleNamespace(method='cached', apply=lambda img, dets: current['warp'])
        return tr
    tracker = make()
    blank = np.zeros((*meta['shape'], 3), np.uint8)
    rows = []
    for f, row in enumerate(meta['frames']):
        if features['resets'][f]:
            tracker = make()
        arr = np.asarray(row['detections'], np.float32).reshape(-1, 6)
        current.update(arr=arr, z=features['features'][features['offsets'][f]:features['offsets'][f+1]], warp=features['warps'][f])
        result = tracker.update(Boxes(arr, meta['shape']), blank)
        objects = []
        if len(result) and len(arr):
            overlap = np.array([[iou(a[:4], b[:4]) for b in arr] for a in result])
            aa, bb = linear_sum_assignment(-overlap)
            for a, b in zip(aa, bb):
                if overlap[a, b] < .3:
                    continue
                objects.append({'track_id': f'{row["scene"]}-{int(result[a][4])}',
                                'bbox': [round(float(v), 2 if meta['long'] else 3) for v in arr[b, :4]],
                                'detection_index': int(b), 'observed': True})
        assert len({o['track_id'] for o in objects}) == len(objects)
        rows.append({'frame': f, 'source_time_s': row['source_time_s'], 'objects': objects})
    if setting['variant'] != 'none':
        assert calls > 0, 'Appearance was requested but never used'
    return {'clip': meta['summary']['clip'], 'setting': setting, 'config': cfg, 'frames': rows,
            'encoder_calls': calls, 'raw_detection_sha256': meta['summary']['input_sha256'],
            'feature_sha256': meta['summary']['feature_sha256'], 'labels_used_for_inference': False,
            'scope': 'Pinned Ultralytics implementation; post-NMS fixed detections. TrackTrack loose-NMS recovery is not exercised.'}


def evaluate(meta, result):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from nbhl_detection_metrics import iou
    obs_by_frame = {}
    for obs in meta['observations']:
        obs_by_frame.setdefault(obs['frame'], []).append(obs)
    assignments = {}
    for row in result['frames']:
        obs = obs_by_frame.get(row['frame'], [])
        objects = row['objects']
        if obs and objects:
            overlap = np.array([[iou(a['bbox'], b['bbox']) for b in objects] for a in obs])
            aa, bb = linear_sum_assignment(np.where(overlap >= .5, 1-overlap, 1e6))
            for a, b in zip(aa, bb):
                if overlap[a, b] >= .5:
                    assignments[obs[a]['id']] = objects[b]['track_id']
    counts = {'correct': 0, 'wrong': 0, 'unknown_same': 0, 'same_total': 0, 'different_total': 0,
              'missing': 0, 'matched_observations': len(assignments), 'reference_observations': len(meta['observations'])}
    pairs = []
    for p in meta['pairs']:
        a, b = assignments.get(p['a']), assignments.get(p['b'])
        same = a == b if a is not None and b is not None else None
        pairs.append(p | {'predicted_same': same, 'track_a': a, 'track_b': b})
        counts['same_total'] += p['truth'] == 'same'
        counts['different_total'] += p['truth'] == 'different'
        counts['missing'] += same is None
        if same:
            counts[{'same': 'correct', 'different': 'wrong', 'uncertain': 'unknown_same'}[p['truth']]] += 1
    return {'clip': result['clip'], 'counts': counts, 'pairs': pairs,
            'fragments': len({o['track_id'] for row in result['frames'] for o in row['objects']}),
            'observations': sum(len(row['objects']) for row in result['frames'])}


def load_development(base, variant):
    import numpy as np
    root = base/'reid-finetune-v1/features'/('generic' if variant == 'none' else variant)
    rows = []
    for p in sorted(root.glob('*/metadata.json')):
        meta = json.loads(p.read_text())
        assert meta['summary']['source_id'] in ['plDp2kPtk_g', 'X83xW6iQKoc']
        npz = p.with_name('features.npz')
        assert sha(npz) == meta['summary']['feature_sha256']
        rows.append((meta, dict(np.load(npz))))
    assert len(rows) == 9, (variant, len(rows))
    return rows


def sweep(base, out, family, volume):
    import time
    cached = {}
    results = []
    for setting in candidates(family):
        t = time.monotonic()
        if setting['variant'] not in cached:
            cached[setting['variant']] = load_development(base, setting['variant'])
        reports = []
        for meta, features in cached[setting['variant']]:
            tracks = replay(meta, features, setting)
            dump(out/'online'/setting['id']/(tracks['clip']+'.json'), tracks)
            reports.append(evaluate(meta, tracks))
        totals = {k: sum(r['counts'][k] for r in reports) for k in reports[0]['counts']}
        result = {'setting': setting, 'counts': totals, 'clips': reports, 'seconds': time.monotonic()-t}
        results.append(result)
        dump(out/('sweep-'+family+'.json'), {'family': family, 'results': results, 'code_sha256': sha(__file__)})
        volume.commit()
        print(json.dumps({k: v for k, v in result.items() if k != 'clips'}), flush=True)
    return {'family': family, 'results': [{k: v for k, v in r.items() if k != 'clips'} for r in results]}


def tracklet_evidence(meta, features, result):
    """Aggregate clear observed crops, retaining multiple views and actual time support."""
    import numpy as np
    from nbhl_detection_metrics import iou
    tracks = {}
    for row in result['frames']:
        f = row['frame']
        detections = meta['frames'][f]['detections']
        vectors = features['features'][features['offsets'][f]:features['offsets'][f+1]]
        for obj in row['objects']:
            k = obj['detection_index']
            box = obj['bbox']
            score = float(detections[k][4])
            clutter = max([iou(box, d[:4]) for j, d in enumerate(detections) if j != k and d[4] >= .5] or [0])
            z = vectors[k]
            if np.linalg.norm(z) < .5:
                continue
            quality = score * (1-clutter)**2 * min(1., (box[3]-box[1])/80.)
            tracks.setdefault(obj['track_id'], []).append({'frame': f, 'bbox': box, 'quality': quality,
                                                          'clutter': clutter, 'score': score, 'feature': z})
    summary = {}
    for tid, rows in tracks.items():
        ranked = sorted(rows, key=lambda r: -r['quality'])
        chosen = []
        for r in ranked:
            if r['quality'] < .35:
                continue
            if all(abs(r['frame']-v['frame']) >= 5 for v in chosen):
                chosen.append(r)
            if len(chosen) >= 12:
                break
        if not chosen:
            chosen = ranked[:1]
        z = np.stack([r['feature'] for r in chosen])
        mean = z.mean(0)
        mean /= max(1e-8, np.linalg.norm(mean))
        summary[tid] = {'id': tid, 'start': rows[0]['frame'], 'end': rows[-1]['frame'],
                        'frames': {r['frame'] for r in rows}, 'count': len(rows), 'mean': mean,
                        'prototypes': z, 'first_box': rows[0]['bbox'], 'last_box': rows[-1]['bbox'],
                        'samples': [{k: v for k, v in r.items() if k != 'feature'} for r in chosen]}
    return summary


def link_candidates(meta, features, evidence):
    """Geometry and appearance only. No evaluation label is an input."""
    import numpy as np
    tracks = sorted(evidence.values(), key=lambda r: (r['start'], r['id']))
    candidates = []
    warps = features['warps']
    for a in tracks:
        if a['count'] < 5:
            continue
        for b in tracks:
            gap = b['start']-a['end']
            if gap <= 0 or gap > 450 or b['count'] < 5:
                continue
            if meta['frames'][a['end']]['scene'] != meta['frames'][b['start']]['scene']:
                continue
            last = a['last_box']
            first = b['first_box']
            point = np.array([(last[0]+last[2])/2, (last[1]+last[3])/2, 1.])
            for warp in warps[a['end']+1:b['start']+1]:
                point[:2] = warp @ point
            destination = np.array([(first[0]+first[2])/2, (first[1]+first[3])/2])
            scale = max(10., ((last[3]-last[1])+(first[3]-first[1]))/2)
            distance = float(np.linalg.norm(point[:2]-destination)/scale)
            cross = a['prototypes'] @ b['prototypes'].T
            # Symmetric best-view support reduces dependence on one lucky frame pair.
            view = float((np.sort(cross.max(0))[-min(3, cross.shape[1]):].mean() +
                          np.sort(cross.max(1))[-min(3, cross.shape[0]):].mean())/2)
            candidates.append({'a': a['id'], 'b': b['id'], 'gap': gap,
                               'cosine_mean': float(a['mean'] @ b['mean']), 'cosine_views': view,
                               'distance_heights': distance, 'support_a': a['count'], 'support_b': b['count']})
    return candidates


def link_tracklets(result, evidence, candidates, setting):
    """Greedy disjoint temporal chains with mutual nearest-neighbor margin checks."""
    import copy
    score_key = 'cosine_'+setting['pooling']
    admissible = [r for r in candidates if r['gap'] <= setting['gap'] and
                  r['distance_heights'] <= setting['motion_base'] + r['gap']*setting['motion_per_frame'] and
                  r['support_a'] >= setting['min_support'] and r['support_b'] >= setting['min_support']]
    by_a, by_b = {}, {}
    for r in admissible:
        by_a.setdefault(r['a'], []).append(r)
        by_b.setdefault(r['b'], []).append(r)
    for group in list(by_a.values()) + list(by_b.values()):
        group.sort(key=lambda r: (-r[score_key], r['gap'], r['a'], r['b']))
    parent = {key: key for key in evidence}
    support = {key: set(value['frames']) for key, value in evidence.items()}
    intervals = {key: [(value['start'], value['end'])] for key, value in evidence.items()}
    def find(key):
        while parent[key] != key:
            key = parent[key]
        return key
    accepted = []
    for r in sorted(admissible, key=lambda r: (-r[score_key], r['gap'], r['a'], r['b'])):
        if r[score_key] < setting['cosine']:
            continue
        aa, bb = by_a[r['a']], by_b[r['b']]
        if aa[0] is not r or bb[0] is not r:
            continue
        margins = [r[score_key]-group[1][score_key] if len(group)>1 else 1. for group in [aa, bb]]
        if min(margins) < setting['margin']:
            continue
        a, b = find(r['a']), find(r['b'])
        if a == b or support[a] & support[b]:
            continue
        # Conservative full-interval conflict also protects unobserved interior gaps.
        if any(max(u, x) <= min(v, y) for u, v in intervals[a] for x, y in intervals[b]):
            continue
        parent[b] = a
        support[a] |= support[b]
        intervals[a] += intervals[b]
        accepted.append(r | {'minimum_margin': min(margins)})
    output = copy.deepcopy(result)
    for row in output['frames']:
        for obj in row['objects']:
            obj['local_track_id'] = obj['track_id']
            if obj['track_id'] in parent:
                obj['track_id'] = find(obj['track_id'])
        assert len({o['track_id'] for o in row['objects']}) == len(row['objects'])
    output.update(linking=setting, links=accepted, labels_used_for_inference=False)
    return output


def linking_grid():
    import itertools
    rows = []
    for pool, cosine, margin, gap in itertools.product(['mean', 'views'], [.75, .85, .92], [0., .03, .07], [30, 90, 300]):
        rows.append({'pooling': pool, 'cosine': cosine, 'margin': margin, 'gap': gap,
                     'motion_base': 1., 'motion_per_frame': .12, 'min_support': 10})
    for i, row in enumerate(rows):
        row['id'] = f'link-{i:03}'
    return rows


def sweep_linking(base, out, online_id, volume):
    import time
    source = out/'online'/online_id
    files = sorted(source.glob('*.json'))
    assert len(files) == 9, (online_id, len(files))
    cached = []
    for meta, features in load_development(base, 'generic'):
        result = json.loads((source/(meta['summary']['clip']+'.json')).read_text())
        evidence = tracklet_evidence(meta, features, result)
        edges = link_candidates(meta, features, evidence)
        cached.append((meta, result, evidence, edges))
    results = []
    for setting in linking_grid():
        started = time.monotonic()
        reports = []
        links = 0
        for meta, result, evidence, edges in cached:
            linked = link_tracklets(result, evidence, edges, setting)
            reports.append(evaluate(meta, linked))
            links += len(linked['links'])
        counts = {k: sum(r['counts'][k] for r in reports) for k in reports[0]['counts']}
        results.append({'online_id': online_id, 'setting': setting, 'counts': counts, 'clips': reports,
                        'links': links, 'seconds': time.monotonic()-started})
    best = max(results, key=lambda r: (r['counts']['correct']-5*r['counts']['wrong'], -r['counts']['unknown_same'], -r['links']))
    for meta, result, evidence, edges in cached:
        linked = link_tracklets(result, evidence, edges, best['setting'])
        dump(out/'linked'/online_id/(result['clip']+'.json'), linked)
    report = {'online_id': online_id, 'appearance': 'generic', 'results': results, 'chosen': best,
              'code_sha256': sha(__file__), 'scope': 'Development only. Labels select settings after inference; no reference labels enter linking.'}
    dump(out/('linking-'+online_id+'.json'), report)
    volume.commit()
    return {k: v for k, v in report.items() if k != 'results'}
