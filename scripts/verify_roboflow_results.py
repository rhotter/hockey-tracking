"""Recount Roboflow associations from published predictions and frozen references."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'web/data/reliability/roboflow-20261025'
EXP=ROOT/'experiments/roboflow-20261025'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def overlap(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return inter/max(1e-12,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)

def recount(reference,prediction,clip):
    refs=[o for o in reference['observations'] if o['clip']==clip]
    assert {r['frame'] for r in prediction['frames']}=={o['frame'] for o in refs}
    assignment={}
    for frame in prediction['frames']:
        obs=[o for o in refs if o['frame']==frame['frame']];objects=frame['objects']
        assert len({o['track_id'] for o in objects})==len(objects)
        if not objects:continue
        matrix=np.array([[overlap(o['bbox'],p['bbox']) for p in objects] for o in obs])
        aa,bb=linear_sum_assignment(np.where(matrix>=.5,1-matrix,1e6))
        for a,b in zip(aa,bb):
            if matrix[a,b]>=.5:assignment[obs[a]['id']]=objects[b]['track_id']
    counts=dict(correct=0,wrong=0,unknown_same=0,same_total=0,different_total=0,missing=0,
                matched_observations=len(assignment),reference_observations=len(refs))
    outcomes={}
    for pair in reference['pairs']:
        if pair['clip']!=clip:continue
        a,b=assignment.get(pair['a']),assignment.get(pair['b'])
        same=a==b if a is not None and b is not None else None
        outcomes[pair['id']]=same
        counts['same_total']+=pair['truth']=='same';counts['different_total']+=pair['truth']=='different'
        counts['missing']+=same is None
        if same:counts[{'same':'correct','different':'wrong','uncertain':'unknown_same'}[pair['truth']]]+=1
    return counts,outcomes

def main():
    exported=read(DATA/'export-manifest.json')
    for item in exported['files']:
        prefix,name=item['name'].split('/',1)
        roots={'reports':DATA,'replays':ROOT/'web/media/tracking-fourhour','review':ROOT/'web/media/roboflow-20261025'}
        assert prefix in roots and '..' not in Path(name).parts
        path=roots[prefix]/name
        assert sha(path)==item['sha256'] and path.stat().st_size==item['bytes'],item['name']
    protocol=read(EXP/'data/ball-hockey/roboflow-replacement-protocol-20261025.json')
    protocol_sha=sha(EXP/'data/ball-hockey/roboflow-replacement-protocol-20261025.json')
    reference_path=ROOT/'web/data/reliability/tracking-four-hour-final-reference-v1.json'
    assert sha(reference_path)=='d75c360c51a07a32c5369955cc3a4afc6190ae7c46822015a061808757dc12d1'
    reference=read(reference_path);pairs={p['id']:p for p in reference['pairs']}
    selection=read(DATA/'selection.json');results=read(DATA/'results.json')
    shapes={r['clip']:r['shape'] for r in read(DATA/'preflight.json')['clips']}
    assert selection['protocol_sha256']==results['protocol_sha256']==protocol_sha
    assert not selection['selection_uses_regression']
    assert results['selection_sha256']==sha(DATA/'selection.json')
    def objective(r):
        c=r['counts'];return (-(c['correct']-10*c['wrong']),-c['matched_observations'],c['unknown_same'],r['fragments'],r['setting'][1]!='default',r['method'])
    assert len(selection['development'])==20
    for detector in protocol['detectors']:
        assert selection['winners'][detector]==min((r for r in selection['development'] if r['detector']==detector),key=objective)
    expected={d+'-'+f+'-'+v for d in protocol['detectors'] for f,v in [('sports_bytetrack','default'),selection['winners'][d]['setting']]}
    expected.add(selection['winners']['yolo_cached']['method']+'-appearance')
    assert results['hybrid_protocol_sha256']==sha(EXP/'data/ball-hockey/roboflow-hybrid-protocol-20261025.json')
    assert {r['method'] for r in results['tracking']}==expected
    checked=0
    for method in results['tracking']:
        total=Counter()
        assert {r['clip'] for r in method['clips']}==set(protocol['regression'])
        for row in method['clips']:
            clip=row['clip'];name=clip+'-'+method['method']+'.json'
            pred=read(DATA/'sampled-predictions'/name);report=read(DATA/'pairs'/name)
            counts,outcomes=recount(reference,pred,clip)
            assert counts==report['counts']==row['counts'],name
            assert {p['id'] for p in report['pairs']}==set(outcomes)
            for pair in report['pairs']:
                for key in ['a','b','truth']:assert pair[key]==pairs[pair['id']][key]
                assert pair['predicted_same'] is outcomes[pair['id']]
            replay=read(ROOT/'web/media/tracking-fourhour'/(clip+'-roboflow-'+method['method']+'.json'))
            assert len(replay['frames'])==(4500 if 'stress' in clip else 450)
            assert replay['fps']==15 and replay['fragment_ids_only'] and not replay['interpolated_positions']
            assert replay['source_tracks_sha256']==report['tracks_sha256']
            height,width=shapes[clip]
            for f in pred['frames']:
                expected_boxes=[[replay['track_id_map'][o['track_id']],*[max(0,min(10000,round(v/s*10000))) for v,s in zip(o['bbox'],[width,height,width,height])]] for o in f['objects']]
                assert replay['frames'][f['frame']]==expected_boxes
            total.update(counts);checked+=1
        assert dict(total)==method['counts']
    baseline=Counter()
    for clip in protocol['regression']:
        pred=read(ROOT/'web/data/reliability/sam3-20261024/sampled-predictions'/(clip+'-sam2-qwen-joint.json'))
        provenance=next(r for r in results['existing_pipeline']['clips'] if r['clip']==clip)
        assert provenance['source_prediction_sha256']==sha(ROOT/'web/data/reliability/sam3-20261024/sampled-predictions'/(clip+'-sam2-qwen-joint.json'))
        baseline.update(recount(reference,pred,clip)[0])
    assert dict(baseline)==results['existing_pipeline']['counts']
    assert read(DATA/'prefetch-check.json')['ordered_frames_equal']==32
    teams=read(DATA/'teams.json');torso=read(DATA/'teams-torso.json')
    assert torso['followup_protocol_sha256']==sha(EXP/'data/ball-hockey/roboflow-team-torso-followup-20261025.json')
    obs={o['id']:o for o in reference['observations']};team_count=0
    for source in teams['sources']+torso['sources']:
        if source['counts'] is None:continue
        assert not source['fit_uses_reference_labels'] and source['fit_uses_unlabeled_evaluation_video']
        auto=source['automatic_predictions'];mapping={int(k):v for k,v in source['scoring_label_permutation'].items()}
        for r in auto:
            assert r['accepted']==(r['margin']>=.2 and r['distance']<=source['training_distance_95pct'])
        for r in source['matches']:
            a=auto[r['automatic_index']]
            assert r['truth']==obs[r['id']]['team'] and overlap(obs[r['id']]['bbox'],a['bbox'])>=.5
            assert r['prediction']==mapping[a['cluster']] and r['correct']==(r['prediction']==r['truth'])
            assert r['accepted']==a['accepted']
        scored=source['matches'];c=source['counts']
        assert c=={'reference_observations':len(scored)+len(source['unmatched_observation_ids']),'matched':len(scored),
                   'correct_all':sum(r['correct'] for r in scored),'accepted':sum(r['accepted'] for r in scored),
                   'correct_accepted':sum(r['correct'] and r['accepted'] for r in scored),'unmatched':len(source['unmatched_observation_ids'])}
        team_count+=len(scored)
    utilities=read(DATA/'utilities.json');g=utilities['geometry']
    assert not g['metric_distances_allowed'] and not g['dimensions_verified'] and g['maximum_difference_relative_units']<1e-5
    for split,methods in utilities['ball']['counts'].items():
        rows=[r for r in utilities['ball']['rows'] if split=='all' or r['split']==split]
        for method,c in methods.items():
            for r in rows:
                v=r[method]
                if v['answered']:
                    b=v['bbox'];error=math.dist([(b[0]+b[2])/2,(b[1]+b[3])/2],r['reference_center'])
                    # Upstream boxes use float32 center arithmetic; the independent
                    # Python recount uses float64. Allow subpixel rounding only.
                    assert math.isclose(error,v['error_pixels'],abs_tol=2e-4) and v['correct']==(error<=r['radius'])
            assert c=={'reviewed_visible':len(rows),'answered':sum(r[method]['answered'] for r in rows),'correct':sum(r[method]['correct'] for r in rows)}
    print(json.dumps({'tracking_reports_recounted':checked,'baseline_counts':dict(baseline),'team_observations_checked':team_count,'ball_frames':len(utilities['ball']['rows']),'frozen_reference_verified':True}))

if __name__=='__main__':main()
