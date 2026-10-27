"""Frozen single-candidate SAM/TAP fusion, preserving the existing context gates."""
from collections import Counter
import numpy as np
from evaluate import accepted, read, write, ROOT, HERE, sequence, summarize
from hockey_tracking.ball_sequence import fuse_point_tracks, PointFusionConfig


def choose(tap, sam):
    if tap is None:
        return sam
    if sam is None:
        return tap
    distance = float(np.linalg.norm(np.array(tap['pixel'])-sam['pixel']))
    margin = sam['appearance_score']-tap['appearance_score']
    if distance <= 25:
        return sam if margin > 0 else tap
    return sam if margin >= .2 else tap


def eligible(p, f, context, config):
    if not (p['visible'] and p['inside_crop'] and p['inside_image']):
        return False
    if p['appearance_score'] < config.appearance_threshold or not p.get('shape_valid', True):
        return False
    matches = any(np.linalg.norm(np.array(h['pixel'])*[1920,1080]-p['pixel']) <= h['radius_px']
                  for h in context['frames'][f].get('hypotheses', []))
    standalone = p['appearance_score'] >= config.standalone_appearance_threshold and abs(f-p['seed_frame']) <= config.standalone_seed_distance_frames
    return matches or standalone


def best(proposals):
    return max(proposals, key=lambda p: (p['appearance_score'], -abs(p['frame']-p['seed_frame']))) if proposals else None


def choose_supported(tap, sams, policy):
    sam=best(sams)
    assert policy in ['conservative','sam_consensus']
    if policy=='sam_consensus' and sam is not None and tap is not None and np.linalg.norm(np.array(tap['pixel'])-sam['pixel'])>25:
        support={p['seed_frame'] for p in sams if p['appearance_score']>=.9
                 and np.linalg.norm(np.array(p['pixel'])-sam['pixel'])<=12}
        if sam['appearance_score']>=tap['appearance_score']-.05 and len(support)>=2:
            return sam
    return choose(tap,sam)


def sam_proposals(data):
    frames = [{'frame': f, 'proposals': []} for f in range(4500)]
    for track in data['tracks']:
        for row in track['rows']:
            frames[row['frame']]['proposals'].append({**row, 'seed_frame': track['seed_frame'], 'tracker': data['model']})
    return frames


def proposal_scores(frames, seeds):
    protocol = read(HERE/'protocol.json')
    rows = []
    for label in sequence.reference():
        if any(abs(label['frame']-s) <= 3 for s in seeds):
            continue
        p = best([p for p in frames[label['frame']]['proposals'] if accepted(p, 'filtered', protocol)])
        error = float(np.linalg.norm(np.array(p['pixel'])-label['pixel'])) if p else None
        rows.append({**label, 'prediction': p['pixel'] if p else None, 'error_px': error})
    return summarize(rows)


def derive(sam, policy='conservative'):
    previous = ROOT/'experiments/ball-sequence-20261026'
    source = read(ROOT/'web/data/overhead-20261025/replay.json')
    visual = read(ROOT/'web/data/ball-rebuild-20261026/replay.json')
    context = read(ROOT/'web/data/ball-sequence-20261026/context-replay.json')
    baseline = read(ROOT/'web/data/ball-sequence-20261026/replay.json')
    tap = read(previous/'results/tapnextpp512-verified.json')['frames']
    frames = sam_proposals(sam)
    config = PointFusionConfig(**read(previous/'point-fusion-config.json'))
    combined = []
    for f in range(4500):
        t = best([{**p, 'tracker': 'tapnextpp512'} for p in tap[f]['proposals'] if eligible(p, f, context, config)])
        s = [p for p in frames[f]['proposals'] if eligible(p, f, context, config)]
        p = choose_supported(t, s, policy)
        combined.append({'frame': f, 'proposals': [p] if p else []})
    fused = fuse_point_tracks(source, visual, context, {'frames': combined}, config)
    fused['method'] = 'Visual anchors, SAM3/TAPNext++ video tracking, and existing sequence context'
    for f, r in enumerate(fused['frames']):
        if r['state'] == 'tracked':
            r['tracker'] = combined[f]['proposals'][0]['tracker']
            r['reason'] = r['tracker']+'_with_appearance_verification'
    seeds = [t['seed_frame'] for t in sam['tracks']]
    changes = []
    for f, (a, b) in enumerate(zip(baseline['frames'], fused['frames'])):
        pa, pb = a.get('pixel'), b.get('pixel')
        distance = float(np.linalg.norm((np.array(pa)-pb)*[1920,1080])) if pa is not None and pb is not None else None
        if a['state'] != b['state'] or (distance is not None and distance > .01):
            changes.append({'frame': f, 'before_state': a['state'], 'after_state': b['state'], 'pixel': pb,
                            'before_pixel': pa, 'change_px': distance, 'tracker': b.get('tracker')})
    candidates = [r for r in changes if r['after_state']=='tracked' and r['tracker'].startswith('sam')]
    audit = [candidates[i] for i in np.linspace(0,len(candidates)-1,min(24,len(candidates)),dtype=int)] if candidates else []
    reviewed = []
    for row in read(previous/'output-audit.json')['rows']:
        if row.get('reviewed_ball_pixel') is None:
            continue
        f=row['frame'];truth=row['reviewed_ball_pixel']
        item={'frame':f,'reviewed_ball_pixel':truth}
        for name,replay in [('before',baseline),('after',fused)]:
            r=replay['frames'][f];p=r.get('pixel')
            item[name]={'state':r['state'],'pixel':p,'error_px':float(np.linalg.norm(np.array(p)*[1920,1080]-truth)) if p is not None else None}
        reviewed.append(item)
    return fused, {'policy':policy,'sam': proposal_scores(frames, seeds), 'tap': proposal_scores(tap, seeds),
                   'hybrid_context_gated': proposal_scores(combined, seeds),
                   'states': dict(Counter(r['state'] for r in fused['frames'])),
                   'baseline_states': dict(Counter(r['state'] for r in baseline['frames'])),
                   'tracker_counts': dict(Counter(r.get('tracker') for r in fused['frames'] if r['state']=='tracked')),
                   'changes': changes, 'audit_selection': audit,
                   'reused_previous_output_audit':reviewed}
