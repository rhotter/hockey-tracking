"""Detector/appearance constraints and independent tracker agreement for SAM3.

This module never reads identity labels. All thresholds are external settings.
"""
import copy
import numpy as np
from scipy.optimize import linear_sum_assignment
from nbhl_detection_metrics import iou


def matches(left, right, cutoff=.3, margin=.05):
    if not left or not right:
        return []
    overlap = np.asarray([[iou(a, b) for b in right] for a in left])
    aa, bb = linear_sum_assignment(np.where(overlap >= cutoff, 1-overlap, 1e6))
    result = []
    for a, b in zip(aa, bb):
        rival = max(np.max(np.delete(overlap[a], b)) if len(right)>1 else 0,
                    np.max(np.delete(overlap[:, b], a)) if len(left)>1 else 0)
        if overlap[a, b] >= cutoff and overlap[a, b]-rival >= margin:
            result.append((int(a), int(b), float(overlap[a, b])))
    return result


def anchor(meta, native, cutoff=.3):
    rows = []
    for row in native['frames']:
        f = row['frame']; detections = meta['frames'][f]['detections']
        eligible = [(j, d) for j, d in enumerate(detections) if d[4] >= .1]
        objects = []
        for a, b, score in matches([o['bbox'] for o in row['objects']], [d[:4] for _,d in eligible], cutoff):
            obj = row['objects'][a]; j, d = eligible[b]
            objects.append({**obj, 'bbox': d[:4], 'confidence': d[4], 'detection_index': j,
                            'sam3_track_id': obj['track_id'], 'anchor_iou': score})
        rows.append({**row, 'objects': objects})
    return {**native, 'frames': rows, 'anchor_iou': cutoff, 'output_requires_current_detection': True}


def split_appearance(meta, features, raw, threshold=0.):
    """Offline change-point split; three corroborating clear crops avoid one-frame noise."""
    result = copy.deepcopy(raw); grouped = {}; splits = []
    for row in result['frames']:
        for obj in row['objects']:
            grouped.setdefault(obj['track_id'], []).append((row['frame'], obj))
    for tid, entries in grouped.items():
        segment = 0; reference = None; pending = []; previous = None
        for f, obj in entries:
            boundary = previous is not None and (f-previous>15 or meta['frames'][f]['scene'] != meta['frames'][previous]['scene'])
            if boundary:
                segment += 1; reference = None; pending = []
                splits.append({'track':tid,'frame':f,'reason':'gap_or_scene'})
            obj['track_id'] = f'{tid}/s{segment}'
            j = obj['detection_index']; z = features['features'][int(features['offsets'][f])+j]
            clutter = max([iou(obj['bbox'], d[:4]) for k,d in enumerate(meta['frames'][f]['detections']) if k!=j and d[4]>=.5] or [0.])
            if obj['confidence']>=.5 and clutter<.25 and np.linalg.norm(z)>.5:
                if reference is None:
                    reference = z.copy()
                elif threshold>0 and float(reference@z)<threshold:
                    previous_cosine=float(reference@z)
                    pending.append((f,obj,z))
                    if len(pending)>=3:
                        segment += 1
                        for ef,p in entries:
                            if pending[0][0]<=ef<=f:p['track_id']=f'{tid}/s{segment}'
                        reference = np.mean([v for _,_,v in pending],axis=0)
                        reference /= max(np.linalg.norm(reference),1e-8)
                        splits.append({'track':tid,'frame':pending[0][0],'reason':'appearance','cosine':previous_cosine})
                        pending=[]
                else:
                    pending=[]; reference=.9*reference+.1*z; reference/=max(np.linalg.norm(reference),1e-8)
            previous=f
    result['appearance_split']={'threshold':threshold,'consecutive_clear_crops':3,'splits':splits}
    return result


def endpoint_votes(raw, native):
    """Independent tracker votes on the same image observations, with no labels."""
    from collections import Counter
    grouped={};bounds={}
    for row, sam in zip(raw['frames'], native['frames']):
        assert row['frame']==sam['frame']
        for obj in row['objects']:
            bound=bounds.setdefault(obj['track_id'],[row['frame'],row['frame']]);bound[1]=row['frame']
        for a,b,score in matches([o['bbox'] for o in row['objects']], [o['bbox'] for o in sam['objects']],.5,.05):
            grouped.setdefault(row['objects'][a]['track_id'],[]).append((row['frame'],sam['objects'][b]['track_id']))
    output={}
    for tid, values in grouped.items():
        ends={}
        for label,rows in [('start',values[:15]),('end',values[-15:])]:
            counts=Counter(t for _,t in rows); best,n=counts.most_common(1)[0]
            endpoint=rows[0 if label=='start' else -1][0]
            ends[label]={'id':best,'support':n,'fraction':n/len(rows),'frame':endpoint,'distance_to_endpoint':abs(endpoint-bounds[tid][0 if label=='start' else 1])}
        output[tid]=ends
    return output


def consensus_edges(edges, votes, mode='bonus', bonus=.05, purity=.8):
    output=[]
    for edge in edges:
        a=votes.get(edge['a'],{}).get('end'); b=votes.get(edge['b'],{}).get('start')
        usable=a is not None and b is not None and min(a['support'],b['support'])>=5 and min(a['fraction'],b['fraction'])>=purity and max(a['distance_to_endpoint'],b['distance_to_endpoint'])<=15
        agree=usable and a['id']==b['id']
        if mode=='require' and not agree:continue
        if mode=='veto' and usable and not agree:continue
        output.append({**edge,'cosine_consensus':min(1.,edge['cosine_mean']+(bonus if agree and mode=='bonus' else 0.)),
                       'sam3_agreement':bool(agree),'sam3_votes_usable':bool(usable)})
    return output
