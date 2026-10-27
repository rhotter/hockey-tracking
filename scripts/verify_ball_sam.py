"""Recompute SAM/TAP comparisons, fusion, audit selection and one-position output."""
from pathlib import Path
import hashlib, json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/ball-sam-20261026'
sys.path.insert(0,str(EXP))
from publish import artifacts, digest
from evaluate import read


def equal(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys(),(a.keys(),b.keys())
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    elif isinstance(a,float):np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-9)
    else:assert a==b,(a,b)


def main():
    positions,report=artifacts();dest=ROOT/'web/data/ball-sam-20261026'
    equal(positions,read(dest/'positions.json'));equal(report,read(dest/'report.json'))
    protocol=read(EXP/'protocol.json');results=EXP/'results'
    for path,expected in protocol['input_sha256'].items():assert digest(ROOT/path)==expected,path
    for field in ['input_sha256','code_sha256']:
        for path,expected in report[field].items():assert digest(ROOT/path)==expected,path
    selected=report['pilot']['selected_variant']
    equal(report['pilot'],read(EXP/'pilot-selection.json'))
    assert digest(EXP/'pilot-selection.json')==report['exploratory_amendment']['pilot_evaluation_sha256']
    variants=[read(results/'pilot'/(name+'.json')) for name in protocol['models']]
    assert len({d['weights_sha256']for d in variants})==1
    assert len({d['verifier_weights_sha256']for d in variants})==1
    expected_seeds=[s['frame']for s in protocol['pilot_seeds']]
    for d in variants:
        equal(d['protocol'],protocol)
        assert [t['seed_frame']for t in d['tracks']]==expected_seeds
        for t in d['tracks']:
            assert [r['frame']for r in t['rows']]==list(range(*t['interval']))
            for r in t['rows']:
                shape=protocol['minimum_mask_area']<=r['mask_area']<=protocol['maximum_mask_area'] and max(r['mask_extent'])<=protocol['maximum_mask_extent']
                assert r['shape_valid']==shape
                assert np.isfinite(r['pixel']).all() and 0<=r['appearance_score']<=1
    for i in range(6):assert len({d['tracks'][i]['input_frame_sequence_sha256']for d in variants})==1
    full=read(results/'all'/(report['decision']['replay_variant']+'.json'))
    tap=read(ROOT/'experiments/ball-sequence-20261026/results/tapnextpp512-all.json')
    assert [t['seed_frame']for t in full['tracks']]==[t['seed_frame']for t in tap['tracks']]
    assert full['weights_sha256']==variants[0]['weights_sha256']
    primary=read(results/'all'/(selected+'.json'))
    assert [t['seed_frame']for t in primary['tracks']]==[t['seed_frame']for t in tap['tracks']]
    for a,b in zip(full['tracks'],primary['tracks']):
        assert a['input_frame_sequence_sha256']==b['input_frame_sequence_sha256']
    equal(report['point_conservative_full']['changes'],report['point_consensus_full']['changes'])
    assert len(positions['frames'])==4500
    old=read(ROOT/'web/data/ball-rebuild-20261026/replay.json')
    for f,row in enumerate(positions['frames']):
        assert row['frame']==f
        assert not {'hypotheses','model_radius_px','pixel_covariance','covariance','known_weight'}&row.keys()
        if old['frames'][f]['state']!='unknown':
            assert row['state']==old['frames'][f]['state']
            equal(row['pixel'],old['frames'][f]['pixel'])
        if 466<=f<516:assert row['state']=='unknown'
        if row.get('pixel') is not None:assert all(0<=v<=1 for v in row['pixel'])
        if row.get('review_support')in['held','airborne','unclear']:assert row['xy'] is None
    audit=report['output_audit'];wanted=report['full']['audit_selection']
    assert [r['frame']for r in audit['rows']]==[r['frame']for r in wanted]
    assert audit['sam_result_sha256']==digest(results/'all'/(report['decision']['replay_variant']+'.json'))
    for row,selection in zip(audit['rows'],wanted):
        np.testing.assert_allclose(row['pixel'],np.array(selection['pixel'])*[1920,1080])
        assert len(row['source_sha256'])==64
        if row.get('reviewed_ball_pixel') is not None:
            error=float(np.linalg.norm(np.array(row['pixel'])-row['reviewed_ball_pixel']))
            assert (error<=25)==(row['verdict']=='visible_within_25px')
            assert row['verdict']in['visible_within_25px','visible_outside_25px']
        else:assert row['verdict']in['unverified','unclear']
    print(json.dumps({'selected':selected,'states':report['full']['states'],'audit_samples':len(audit['rows']),'single_positions':4500}))


if __name__=='__main__':main()
