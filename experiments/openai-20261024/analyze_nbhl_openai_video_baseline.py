"""Recount direct-video sparse observations/pairs against the same saved baseline."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

from run_nbhl_openai_pilot import sha, atomic_json


def iou(a,b):
    w=max(0,min(a[2],b[2])-max(a[0],b[0]))
    h=max(0,min(a[3],b[3])-max(a[1],b[1]))
    intersection=w*h
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0


def score(reference, clip, frames):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    observations=[r for r in reference['observations'] if r['clip']==clip]
    reference_pairs=[r for r in reference['pairs'] if r['clip']==clip]
    by_frame={}
    for obs in observations:by_frame.setdefault(obs['frame'],[]).append(obs)
    assignments={};invalid=0
    for frame,obs in by_frame.items():
        objects=frames.get(frame,[])
        ids=Counter(r['track_id'] for r in objects)
        valid=[]
        for row in objects:
            b=row['bbox']
            if (ids[row['track_id']]!=1 or len(b)!=4 or
                not all(isinstance(v,(int,float)) and math.isfinite(v) for v in b) or
                b[0]>=b[2] or b[1]>=b[3]):
                invalid+=1;continue
            valid.append(row)
        if not valid:continue
        overlaps=np.array([[iou(a['bbox'],b['bbox']) for b in valid] for a in obs])
        aa,bb=linear_sum_assignment(np.where(overlaps>=.5,1-overlaps,1e6))
        for a,b in zip(aa,bb):
            if overlaps[a,b]>=.5:
                assignments[obs[a]['id']]={'track_id':valid[b]['track_id'],'iou':float(overlaps[a,b])}
    counts={'reference_observations':len(observations),'matched_observations':len(assignments),
            'same_total':0,'different_total':0,'correct':0,'wrong':0,'unknown_same':0,'missing':0}
    pairs=[]
    for p in reference_pairs:
        a=assignments.get(p['a']);b=assignments.get(p['b'])
        same=None if a is None or b is None else a['track_id']==b['track_id']
        counts['same_total']+=p['truth']=='same';counts['different_total']+=p['truth']=='different'
        counts['missing']+=same is None
        if same:counts[{'same':'correct','different':'wrong','uncertain':'unknown_same'}[p['truth']]]+=1
        pairs.append(p|{'predicted_same':same,'track_a':a['track_id'] if a else None,'track_b':b['track_id'] if b else None})
    return {'clip':clip,'counts':counts,'pairs':pairs,'assignments':assignments,
            'invalid_observations_excluded':invalid,
            'false_positives':None,'reason':'Sparse reference is not an exhaustive list of all players.'}


def direct_frames(record,clip):
    value=record.get('parsed') or {}
    ids={p['player_id'] for p in value.get('players',[])}
    sizes={r['frame']:r['source_dimensions'] for r in clip['images']}
    counts=Counter(r['frame'] for r in value.get('frames',[]))
    frames={}
    for frame in value.get('frames',[]):
        f=frame['frame']
        if f not in clip['output_frames'] or counts[f]!=1:continue
        width,height=sizes[f]
        objects=[]
        for row in frame['objects']:
            if row['player_id'] not in ids:continue
            b=row['bbox']
            if len(b)!=4 or not all(isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<=1000 for v in b):continue
            objects.append({'track_id':row['player_id'],'bbox':[b[0]*width/1000,b[1]*height/1000,b[2]*width/1000,b[3]*height/1000]})
        frames[f]=objects
    return frames


def analyze(manifest_path,ledger_path,reference_path,baseline_dir,protocol_path):
    manifest=json.loads(manifest_path.read_text());ledger=json.loads(ledger_path.read_text())
    protocol=json.loads(protocol_path.read_text());reference=json.loads(reference_path.read_text())
    assert ledger['manifest_sha256']==sha(manifest_path)
    assert ledger['protocol_sha256']==manifest['protocol_sha256']==sha(protocol_path)
    assert manifest['reference_labels_supplied'] is False and manifest['native_video'] is False
    models=[]; baselines=[]
    for clip in manifest['clips']:
        inputs={r['frame']:r for r in clip['images']}
        for ref in reference['observations']:
            if ref['clip']==clip['clip']:
                assert ref['frame'] in clip['output_frames']
                assert inputs[ref['frame']]['source_sha256']==ref['frame_sha256'],'Source frame changed'
        bp=baseline_dir/(clip['clip']+'-sam2-qwen-joint.json')
        b=json.loads(bp.read_text())
        baselines.append(score(reference,clip['clip'],{r['frame']:r['objects'] for r in b['frames']})|{'file_sha256':sha(bp)})
    for spec in protocol['models']:
        rows=[]
        for clip in manifest['clips']:
            record=ledger['requests'].get(spec['id']+'/'+clip['clip'],{})
            result=score(reference,clip['clip'],direct_frames(record,clip))
            value=record.get('parsed') or {}
            result.update(status=record.get('status','not_attempted'),validation_errors=record.get('validation_errors'),
                          seconds=record.get('seconds'),cost_ceiling_usd=record.get('cost_ceiling_usd',0),
                          usage=record.get('usage'),returned_model=record.get('returned_model'),
                          player_count=len(value.get('players',[])),event_count=len(value.get('events',[])),
                          non_null_player_metrics={k:sum(p.get(k) is not None for p in value.get('players',[])) for k in
                                                   ['jersey_number','time_on_floor_s','distance_miles','shots','shots_on_goal','goals','assists']},
                          event_and_stat_accuracy=None, events=value.get('events',[]),
                          limitations=value.get('limitations',[]))
            rows.append(result)
        models.append({'model':spec['id'],'clips':rows,
                       'counts':{k:sum(r['counts'][k] for r in rows) for k in rows[0]['counts']},
                       'cost_ceiling_usd':sum(r['cost_ceiling_usd'] for r in rows)})
    return {'protocol_sha256':sha(protocol_path),'reference_sha256':sha(reference_path),
            'ledger_sha256':sha(ledger_path),'manifest_sha256':sha(manifest_path),
            'models':models,'pipeline':{'method':'SAM2 + OSNet + joint Qwen2B', 'clips':baselines,
                                      'counts':{k:sum(r['counts'][k] for r in baselines) for k in baselines[0]['counts']}},
            'stop_reason':ledger.get('stop_reason'),
            'limitations':['Reused sparse assistant-reviewed reference; not independent or full-game identity accuracy.',
                           'No detection precision/false-positive count: references are not exhaustive.',
                           'OpenAI receives 1 / 0.5 fps; pipeline used 15 fps. This is a deployment baseline, not a sampling-matched ablation.',
                           'Direct model events and stats are unverified predictions, not established game statistics.',
                           'Native video/audio not supplied; clip statistics do not establish full-game totals.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','ledger','reference','baseline-dir','protocol','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    value=analyze(a.manifest,a.ledger,a.reference,a.baseline_dir,a.protocol)
    atomic_json(a.output,value)
    print(json.dumps({k:v for k,v in value.items() if k not in ['models','pipeline']}|
                     {'models':[{'model':m['model'],'counts':m['counts'],'cost_ceiling_usd':m['cost_ceiling_usd']} for m in value['models']],
                      'pipeline':value['pipeline']['counts']}))


if __name__=='__main__':main()
