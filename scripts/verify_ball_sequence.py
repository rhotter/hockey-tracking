"""Reproduce sequence inference and fusion; check provenance and uncertainty contracts."""
from pathlib import Path
import sys,json,collections,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments/ball-sequence-20261026'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(EXP))
from publish import derive,merge_ledgers,read,single_positions
from select_audit import select

def equal(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    elif isinstance(a,float):np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-10)
    else:assert a==b,(a,b)


def main():
    dest=ROOT/'web/data/ball-sequence-20261026';context,published,report=derive()
    equal(context,read(dest/'context-replay.json'));equal(published,read(dest/'replay.json'));equal(report,read(dest/'report.json'))
    equal(single_positions(published),read(dest/'positions.json'))
    ledger=read(EXP/'results/context-ledger.json');equal(ledger,merge_ledgers(EXP/'results'))
    assert len(ledger['requests'])==30
    assert {r['core'][0]for r in ledger['requests'].values()}==set(range(0,4500,150))
    for record in ledger['requests'].values():
        assert record['status']=='completed'
        assert sorted(s['frame']for s in record['answer']['samples'])==record['targets']
        assert len(record['targets'])==20 and len(record['images'])>=20
    source=read(ROOT/'web/data/overhead-20261025/replay.json');previous=read(ROOT/'web/data/ball-rebuild-20261026/replay.json')
    assert len(published['frames'])==4500 and published['duration_s']==300
    assert not published['uncertainty_calibrated'] and not published['independent_human_ground_truth']
    for f,row in enumerate(published['frames']):
        assert row['frame']==f
        if previous['frames'][f]['state']!='unknown':equal(row,previous['frames'][f])
        if 466<=f<516:assert row['state']=='unknown'
        if row['state']=='unknown':continue
        assert np.isfinite(row['pixel']).all()and all(0<=v<=1 for v in row['pixel'])
        assert np.isfinite(row['pixel_covariance']).all()
        if row['xy']is not None:
            assert source['frames'][f]['camera_valid']
            assert np.isfinite(row['xy']).all()and np.isfinite(row['covariance']).all()
        if row['state']in['context','tracked'] and row['review_support']!='floor':assert row['xy']is None
        if row['state']=='context':
            assert row['hypotheses'] and row['known_weight']>=published['config']['minimum_known_weight']
            for h in row['hypotheses']:
                assert 60<=h['radius_px']<=180
                if h['support']!='floor':assert h['xy']is None
        if row['state']=='tracked':
            assert row['appearance_score']>=published['point_fusion_config']['appearance_threshold']
            if not row['context_agreement']:
                assert row['appearance_score']>=.9 and abs(f-row['seed_frame'])<=12 and row['xy']is None
    assert report['states']==dict(collections.Counter(r['state']for r in published['frames']))
    assert sum(r['estimated_or_reserved_usd']for r in report['api_accounting'].values())<=50
    for name,digest in report['input_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    for name,digest in report['algorithm_sha256'].items():assert hashlib.sha256((ROOT/'hockey_tracking'/name).read_bytes()).hexdigest()==digest
    for name,digest in report['experiment_code_sha256'].items():assert hashlib.sha256((EXP/name).read_bytes()).hexdigest()==digest
    expected=select(published,previous);audit=read(EXP/'output-audit.json')
    assert audit['replay_sha256']==hashlib.sha256((dest/'replay.json').read_bytes()).hexdigest()
    assert {r['frame']for r in audit['rows']}==set(expected['context']+expected['tracked'])
    assert audit['counts']==dict(collections.Counter(r['verdict']for r in audit['rows']))
    for row in audit['rows']:
        current=published['frames'][row['frame']];assert row['state']==current['state']
        np.testing.assert_allclose(row['pixel'],np.array(current['pixel'])*[1920,1080])
        if row.get('reviewed_ball_pixel'):
            hs=current.get('hypotheses') or [{'pixel':current['pixel'],'radius_px':current['model_radius_px']}]
            inside=any(np.linalg.norm(np.array(h['pixel'])*[1920,1080]-row['reviewed_ball_pixel'])<=h['radius_px']for h in hs)
            assert inside==(row['verdict']=='visible_inside')
    print(json.dumps({'reproduced_frames':4500,'completed_sequences':30,'states':report['states'],'new_estimate_audit':audit['counts']}))

if __name__=='__main__':main()
