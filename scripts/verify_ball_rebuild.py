"""Recompute the published path, gap inference, and ball evaluation from frozen inputs."""
import collections,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
EXP=ROOT/'experiments/ball-rebuild-20261026';sys.path.insert(0,str(EXP))
from hockey_tracking.ball_path import select_path,PathConfig
from hockey_tracking.ball_track import make_replay
from hockey_tracking.ball_review import apply_review,additional_visual_proposals,merge_reviewed,require_floor_support
from evaluate import score,evaluate_regression,multiview_regression,verified_regression


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
    dest=ROOT/'web/data/ball-rebuild-20261026';source=ROOT/'web/data/overhead-20261025/replay.json'
    report=json.loads((dest/'report.json').read_text());published=json.loads((dest/'replay.json').read_text())
    replay=json.loads(source.read_text());candidates=json.loads((dest/'candidates.json').read_text())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert report['input_sha256']==digest(source)
    assert report['candidates_sha256']==digest(dest/'candidates.json')
    for name,h in report['algorithm_sha256'].items():assert h==digest(ROOT/'hockey_tracking'/name)
    for name,h in report['experiment_code_sha256'].items():assert h==digest(EXP/name)
    choice=json.loads((EXP/'selection.json').read_text());assert choice==report['selection']
    selected=select_path(candidates,replay,PathConfig(**choice['path_config']),excluded_frames=range(466,516))
    cv_result=make_replay(replay,selected['detections'],choice.get('gap_seconds',.2),choice.get('max_inferred_radius_px',24));regenerated=cv_result
    equal(cv_result,json.loads((dest/'proposals-replay.json').read_text()))
    if choice.get('visual_review'):
        ledger=json.loads((EXP/choice['visual_review']).read_text())
        assert digest(EXP/choice['visual_review'])==report['visual_review']['ledger_sha256']
        assert hashlib.sha256((json.dumps(cv_result,separators=(',',':'))+'\n').encode()).hexdigest()==ledger['protocol']['source_replay_sha256']
        regenerated=apply_review(cv_result,ledger)
    primary_reviewed=regenerated
    if choice.get('recall'):
        extra=choice['recall'];extra_candidates=json.loads((dest/'recall-candidates.json').read_text())
        assert digest(dest/'recall-candidates.json')==report['recall_review']['candidates_sha256']
        extra_path=select_path(extra_candidates,replay,PathConfig(**extra['path_config']),excluded_frames=range(466,516))
        sensitive=make_replay(replay,extra_path['detections'],choice.get('gap_seconds',.2),choice.get('max_inferred_radius_px',24))
        proposals=additional_visual_proposals(cv_result,sensitive)
        equal(proposals,json.loads((dest/'recall-proposals-replay.json').read_text()))
        extra_ledger=json.loads((EXP/extra['visual_review']).read_text())
        assert digest(EXP/extra['visual_review'])==report['recall_review']['ledger_sha256']
        assert hashlib.sha256((json.dumps(proposals,separators=(',',':'))+'\n').encode()).hexdigest()==extra_ledger['protocol']['source_replay_sha256']
        extra_reviewed=apply_review(proposals,extra_ledger)
        assert sum(r['state']=='detected'for r in extra_reviewed['frames'])==report['recall_review']['accepted_before_support_gate']
        if extra.get('require_floor_support'):extra_reviewed=require_floor_support(extra_reviewed)
        equal(extra_reviewed,json.loads((dest/'recall-reviewed-replay.json').read_text()))
        assert sum(r['state']=='detected'for r in proposals['frames'])==report['recall_review']['proposed_detected']
        assert sum(r['state']=='detected'for r in extra_reviewed['frames'])==report['recall_review']['accepted_detected']
        if extra['enabled']:regenerated=merge_reviewed(regenerated,extra_reviewed)
    equal(published,regenerated)
    assert len(published['frames'])==4500
    assert dict(collections.Counter(r['state'] for r in published['frames']))==report['states']
    assert not published['uncertainty_calibrated'] and not report['independent_human_ground_truth']
    for row in published['frames']:
        if row['state']=='unknown':continue
        assert np.isfinite(row['pixel']).all()
        assert np.all(np.linalg.eigvalsh(row['pixel_covariance'])>0)
        if row['xy'] is not None:assert np.all(np.linalg.eigvalsh(row['covariance'])>0)
        if row['state']=='inferred':
            a,b=row['evidence_frames'];assert b-a<=3
            assert selected['detections'][a] and selected['detections'][b]
            assert row['model_radius_px']<=24
    assert all(published['frames'][f]['state']=='unknown' for f in range(466,516))
    baseline=json.loads((ROOT/'web/data/ball-context-20261026/replay.json').read_text())
    labels=[]
    for name in ['review','targeted']:labels.extend(json.loads((EXP/f'{name}-labels.json').read_text())['rows'])
    for name,data in [('baseline',baseline),('cv_before_review',cv_result),('primary_review_only',primary_reviewed),('improved',published)]:
        for split in ['development','test']:
            subset=[r for r in labels if r['split']==split];pred=[]
            for r in subset:
                point=data['frames'][r['frame']].get('pixel');pred.append([point[0]*1920,point[1]*1080] if point else None)
            equal(score(subset,pred),report['review'][name][split])
    equal(evaluate_regression(EXP/'predictions'),report['regression'])
    equal(multiview_regression(EXP/'predictions'),json.loads((EXP/'predictions/multi-view-regression.json').read_text()))
    if choice.get('verifier'):
        equal(verified_regression(EXP/'predictions',choice['verifier'],choice['verification_threshold']),json.loads((EXP/'predictions/verified-multi-view-regression.json').read_text()))
        training=EXP/'training'/choice['verifier']/'completed.json'
        assert report['training_report_sha256']==digest(training)
        equal(report['verifier_training'],json.loads(training.read_text()))
        samples=training.parent/'training-samples.json'
        assert report['training_samples_sha256']==digest(samples)
        samples=json.loads(samples.read_text())
        assert sum(r['label']==1 for r in samples['train'])==report['verifier_training']['training_positive_frames']
        assert sum(r['label']==0 for r in samples['train'])==report['verifier_training']['training_negative_crops']
        train=json.loads((EXP/'training-labels.json').read_text())['rows']
        adapted=json.loads((EXP/'verifier-adaptation-labels.json').read_text())['rows']
        heldout=[r for r in json.loads((EXP/'regression-labels.json').read_text())['rows'] if r['split']=='regression']
        training_sources={r['source_id'] for r in train+adapted if r['split']=='train'}
        assert not training_sources & {r['source_id'] for r in heldout}
        assert all(r['frame']<1500 for r in adapted)
        assert len(adapted)==23 and len(heldout)==59
        for name,h in choice['labels_sha256'].items():assert digest(EXP/name)==h
    if 'output_audit' in report:
        audit=json.loads((EXP/'output-audit-final.json').read_text());equal(audit,report['output_audit'])
        assert audit['replay_sha256']==digest(dest/'replay.json')
        assert audit['selection_sha256']==digest(EXP/'selection.json')
        assert dict(collections.Counter(r['verdict'] for r in audit['rows']))==audit['counts']
        for row in audit['rows']:
            actual=published['frames'][row['frame']]
            assert actual['state']==row['state']
            np.testing.assert_allclose(np.array(actual['pixel'])*[1920,1080],row['pixel'])
    print(json.dumps({'reproduced_frames':4500,'states':report['states'],'review_labels':len(labels),'regression_models':len(report['regression'])}))


if __name__=='__main__':main()
