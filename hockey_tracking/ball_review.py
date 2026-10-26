"""Apply recorded visual-review decisions without creating ball positions."""
from copy import deepcopy


def additional_visual_proposals(conservative, sensitive):
    if len(conservative['frames'])!=len(sensitive['frames']):raise ValueError('Proposal frame count mismatch')
    result=deepcopy(sensitive)
    for f,row in enumerate(result['frames']):
        if row['state']!='detected' or conservative['frames'][f]['state']!='unknown':
            result['frames'][f]={'frame':f,'state':'unknown','reason':'not_additional_visual_candidate'}
    return result


def merge_reviewed(primary, additional):
    if len(primary['frames'])!=len(additional['frames']):raise ValueError('Review frame count mismatch')
    result=deepcopy(primary)
    for f,row in enumerate(additional['frames']):
        if row['state']=='unknown':continue
        if row['state']!='detected':raise ValueError('Expansion cannot introduce inferred positions')
        if result['frames'][f]['state']!='unknown':raise ValueError('Expansion cannot overwrite primary positions')
        result['frames'][f]={**deepcopy(row),'candidate_pass':'recall'}
    result['method']='Visual candidate path with OpenAI review and additional reviewed detections'
    return result


def require_floor_support(replay):
    """Restrict the weaker recovery pass to reviewed, projectable floor balls.

    The recovery classifier admits many more background objects. Its visual
    reviewer confused red lights with airborne balls in a development audit.
    This gate sacrifices airborne recall; it does not infer a ball's height.
    """
    result=deepcopy(replay)
    for f,row in enumerate(result['frames']):
        if row['state']=='unknown':continue
        if row.get('review_support')!='floor' or row.get('xy') is None:
            result['frames'][f]={'frame':f,'state':'unknown','reason':'recovery_floor_support_unconfirmed'}
    return result


def apply_review(replay, ledger):
    decisions={}
    for record in ledger['requests'].values():
        if record['status']!='completed' or not record.get('answer'):continue
        samples=record['answer']['samples']
        if sorted(r['frame'] for r in samples)!=sorted(record['targets']):raise ValueError('Review target mismatch')
        if len(set(record['frames']))!=len(record['frames']):raise ValueError('Duplicate episode frame')
        for f in record['frames']:
            if f in decisions:raise ValueError('Overlapping review episodes')
            decisions[f]=min(samples,key=lambda r:(abs(r['frame']-f),r['frame']))
    result=deepcopy(replay)
    for f,row in enumerate(result['frames']):
        if row['state']=='unknown':continue
        review=decisions.get(f)
        if review is None or review['object']!='ball':
            result['frames'][f]={'frame':f,'state':'unknown','reason':'review_'+(review['object'] if review else 'unavailable')}
            continue
        row['review_frame']=review['frame'];row['review_support']=review['support']
        if review['support']!='floor':
            row['xy']=None;row['covariance']=None;row['mapping_status']='review_'+review['support']
    for f,row in enumerate(result['frames']):
        if row['state']=='inferred':
            anchors=[result['frames'][a] for a in row['evidence_frames']]
            if not all(a['state']=='detected' and a.get('xy') is not None and a.get('review_support')=='floor' for a in anchors):
                result['frames'][f]={'frame':f,'state':'unknown','reason':'review_inference_anchor_unconfirmed'}
    result['method']='Visual candidate path with OpenAI episode review'
    result['review_is_ground_truth']=False
    return result
