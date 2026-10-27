"""Recount published ball experiments; this does not rerun GPU inference."""
from collections import Counter
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hockey_tracking.ball_refinement import refine_sequence, RefinementConfig
EXP = ROOT/'experiments/ball-temporal-20261026'
spec = importlib.util.spec_from_file_location('ball_temporal_evaluate',EXP/'evaluate.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_numeric_equal(actual, expected, path='result'):
    """Allow only roundoff from different BLAS CPUs, not different decisions."""
    if isinstance(expected, float):
        assert isinstance(actual, (float,int)) and not isinstance(actual,bool), path
        assert math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12), (path,actual,expected)
    elif isinstance(expected, dict):
        assert actual.keys()==expected.keys(), path
        for key,value in expected.items(): assert_numeric_equal(actual[key],value,f'{path}.{key}')
    elif isinstance(expected, list):
        assert len(actual)==len(expected), path
        for i,(a,b) in enumerate(zip(actual,expected)): assert_numeric_equal(a,b,f'{path}[{i}]')
    else:
        assert actual==expected, (path,actual,expected)


def recount(metrics, labels):
    details = metrics['details']
    references = {(r['clip'],r['frame']):r for r in labels}
    assert len(details) == len(labels) == len(references)
    frames = []
    for row in details:
        reference = references[(row['clip'],row['frame'])]
        assert row['reference_pixel'] == reference.get('pixel')
        frames.append({'clip':row['clip'],'frame':row['frame'],
            'source_sha256':reference.get('source_sha256') or reference.get('sha256'),
            'candidates':[{'pixel':row['pixel'],'score':1.}] if row['pixel'] is not None else []})
    result = evaluation.evaluate(frames,labels,{'threshold':.5,'appearance_power':0},metrics['radius_px'])
    assert result == metrics


def main():
    selection_path = EXP/'results/development-selection.json'
    selection = read(selection_path)
    choice = read(EXP/'results/model-choice.json')
    assert choice['selection_sha256'] == sha(selection_path)
    assert not choice['fresh_scores_opened']
    verified = 0
    reports = []
    for scope in ['fresh','extra','final-check']:
        reference_path = EXP/f'{scope}-reference.json'
        labels = read(reference_path)['rows']
        report = read(EXP/f'results/{scope}-evaluation.json')
        assert report['selection_sha256'] == sha(selection_path)
        assert report['reference_sha256'] == sha(reference_path)
        assert datetime.fromisoformat(choice['frozen_at']) < datetime.fromisoformat(report['evaluated_at'])
        for name,data in report['models'].items():
            frozen = selection['variants'][name]
            assert data['operating_point'] == frozen['operating_point']
            assert data['weights_sha256'] == frozen['weights_sha256']
            for radius in [12,25]:
                recount(data[f'at_{radius}px'],labels)
                verified += 1
        reports.append(report)
    combined = read(EXP/'results/fresh-combined.json')
    keys = ['visible','absent','unknown','correct','wrong','missing','false_on_absent',
            'correct_absent','outputs_on_unknown','tp','fp','fn']
    for file, inputs in [('fresh-combined.json',reports[:2]),('all-intervals.json',reports)]:
        for name,data in read(EXP/'results'/file)['models'].items():
            for radius in ['at_12px','at_25px']:
                for key in keys:
                    assert data[radius][key] == sum(r['models'][name][radius][key] for r in inputs)
    protocol=read(EXP/'final-check-protocol.json')
    assert protocol['selection_sha256']==sha(selection_path)
    assert datetime.fromisoformat(protocol['created_at']) < datetime.fromisoformat(reports[-1]['evaluated_at'])
    labels=read(EXP/'final-check-reference.json')['rows']
    assert {(r['clip'],r['frame']) for r in labels} == {
        (c['clip'],f) for c in protocol['clips'] for f in protocol['sample_frames']}
    transfer_path = EXP/'results/transfer-evaluation.json'
    if transfer_path.exists():
        transfer = read(transfer_path)
        labels = read(EXP/'fresh-reference.json')['rows'] + read(EXP/'extra-reference.json')['rows']
        assert transfer['protocol'] == read(EXP/'transfer-protocol.json')
        for data in transfer['models'].values():
            for radius in [12,25]:
                recount(data[f'at_{radius}px'],labels)
                verified += 1
    for name in ['ensemble','adjudication']:
        report=read(EXP/f'results/{name}-evaluation.json')
        assert report['protocol']==read(EXP/f'{name}-protocol.json')
        for data in report['models'].values():
            for radius in [12,25]:
                recount(data[f'at_{radius}px'],labels)
                verified+=1
    for filename in ['refinement-full-audit.json','output-audit-box10.json','output-audit-detector-ungated.json',
                     'output-audit-final.json','reference-quality-audit.json']:
        audit = read(EXP/filename)
        assert audit['counts'] == dict(Counter(r['verdict'] for r in audit['rows']))
        assert len({r['frame'] for r in audit['rows']}) == len(audit['rows'])
    quality=read(EXP/'reference-quality-audit.json')
    assert quality['reference_sha256']==sha(EXP/'reused-reference.json')
    reference={r['frame']:r for r in read(EXP/'reused-reference.json')['rows']}
    assert set(reference)=={r['frame'] for r in quality['rows']}
    for row in quality['rows']:
        assert row['pixel']==reference[row['frame']]['pixel']
    distilled=read(EXP/'results/distillation-evaluation.json')
    assert distilled['protocol']==read(EXP/'distill-protocol.json')
    assert distilled['teacher_labels_sha256']==sha(EXP/'distill-labels.json')
    assert distilled['transfer_inputs_sha256']==sha(EXP/'results/distill-transfer-inputs.json')
    teacher=read(EXP/'distill-labels.json')['rows']
    transfer=read(EXP/'results/distill-transfer-inputs.json')['rows']
    labels=sum([read(EXP/f'{scope}-reference.json')['rows'] for scope in ['fresh','extra','final-check']],[])
    for variant in distilled['variants'].values():
        assert len(variant['rows'])==len(teacher)
        for row,original in zip(variant['rows'],teacher):
            assert all(row[k]==v for k,v in original.items())
            assert row['accepted']==(row['score']>=variant['threshold'])
        for split,key in [('development','development'),('test','test')]:
            rows=[r for r in variant['rows'] if r['split']==split]
            counts={name:sum(bool(r['label'])==positive and r['accepted']==accepted for r in rows)
                    for name,positive,accepted in [('tp',True,True),('fp',False,True),('fn',True,False),('tn',False,False)]}
            for name,value in counts.items():assert variant[key][name]==value
            assert variant[key]['f1']==2*counts['tp']/max(1,2*counts['tp']+counts['fp']+counts['fn'])
        assert len(variant['transfer'])==len(transfer)
        for row,original in zip(variant['transfer'],transfer):
            assert all(row[k]==v for k,v in original.items())
            assert row['accepted']==(row['verifier_score'] is not None and row['verifier_score']>=variant['threshold'])
    selected=max(['image-head','temporal-head'],key=lambda name:(
        distilled['variants'][name]['development']['f1'],
        -distilled['variants'][name]['development']['fp'],name=='image-head'))
    assert selected==distilled['selected_variant']
    for name,data in distilled['models'].items():
        for radius in [12,25]:
            recount(data[f'at_{radius}px'],labels)
            for row,prediction in zip(data[f'at_{radius}px']['details'],distilled['variants'][name]['transfer']):
                assert (row['clip'],row['frame'])==(prediction['clip'],prediction['frame'])
                assert row['pixel']==(prediction['pixel'] if prediction['accepted'] else None)
            verified+=1
    motion=read(EXP/'results/motion-selection-evaluation.json')
    motion_inputs=read(EXP/'results/motion-selection-inputs.json')
    assert motion['protocol']==motion_inputs['protocol']==read(EXP/'motion-selection-protocol.json')
    assert motion['inputs_sha256']==sha(EXP/'results/motion-selection-inputs.json')
    motion_spec=importlib.util.spec_from_file_location('motion_selection',EXP/'score_motion.py')
    motion_module=importlib.util.module_from_spec(motion_spec)
    sys.modules['evaluate']=evaluation
    motion_spec.loader.exec_module(motion_module)
    labels=read(EXP/'fresh-reference.json')['rows']+read(EXP/'extra-reference.json')['rows']
    operating=selection['variants']['yolo-full1920']['operating_point']
    for name,data in motion['models'].items():
        for radius in [12,25]:
            recount(data[f'at_{radius}px'],labels);verified+=1
        for sample,row,decision in zip(motion_inputs['rows'],data['at_25px']['details'],data['decisions']):
            point,reason=motion_module.rank_candidates(sample,operating,data['config'],motion['protocol']['maximum_candidates'])
            assert (sample['clip'],sample['frame'])==(row['clip'],row['frame'])
            assert point==row['pixel']
            assert_numeric_equal(reason,{k:v for k,v in decision.items() if k not in ['clip','frame']})
    labels=sum([read(EXP/f'{scope}-reference.json')['rows'] for scope in ['fresh','extra','final-check']],[])
    replay_training=read(EXP/'results/distillation-replay-report-evaluation.json')
    assert replay_training['protocol']==read(EXP/'distill-replay-protocol.json')
    def classification(rows,threshold):
        counts={key:sum(bool(r['label'])==positive and (r['score']>=threshold)==accepted for r in rows)
                for key,positive,accepted in [('tp',True,True),('fp',False,True),('fn',True,False),('tn',False,False)]}
        return {**counts,'f1':2*counts['tp']/max(1,2*counts['tp']+counts['fp']+counts['fn'])}
    for variant in replay_training['variants'].values():
        threshold=variant['threshold']
        assert [{k:v for k,v in r.items() if k not in ['score','accepted']} for r in variant['rows']]==teacher
        assert classification([r for r in variant['rows'] if r['split']=='test'],threshold)==variant['test']
        assert classification([r for r in variant['rows'] if r['split']=='development'],threshold)==variant['development']['teacher']
        assert classification(variant['original_validation'],threshold)==variant['development']['original']
        dev=variant['development']
        assert dev['mean_f1']==(dev['teacher']['f1']+dev['original']['f1'])/2
        for row in variant['rows']:assert row['accepted']==(row['score']>=threshold)
    for name,metrics in replay_training['evaluation'].items():
        frames=[{**r,'candidates':[{'pixel':r['pixel'],'score':1.}] if r['accepted'] else []}
                for r in replay_training['variants'][name]['transfer']]
        for radius in [12,25]:
            assert evaluation.evaluate(frames,labels,{'threshold':.5,'appearance_power':0},radius)==metrics[f'at_{radius}px']
            verified+=1
    for filename in ['verifier-rerank','multiview-transfer']:
        followup=read(EXP/f'results/{filename}-evaluation.json')
        assert followup['protocol']==read(EXP/f'{filename}-protocol.json')
        assert followup['protocol']['operating_point']==operating
        for name,model in followup['models'].items():
            if name!='original':assert model['weights_sha256']==replay_training['variants'][name]['weights_sha256']
            for radius in [12,25]:
                assert evaluation.evaluate(model['frames'],labels,operating,radius)==followup['evaluation'][name][f'at_{radius}px']
                verified+=1
    transfer_audit=read(EXP/'transfer-output-audit.json')
    audit_selection=read(EXP/'transfer-output-audit-selection.json')
    assert transfer_audit['counts']==dict(Counter(r['verdict'] for r in transfer_audit['rows']))
    assert len(transfer_audit['rows'])==60
    for row,selected_row in zip(transfer_audit['rows'],audit_selection['rows']):
        assert all(row[k]==v for k,v in selected_row.items())
    gates=read(EXP/'results/transfer-output-audit-gates.json')
    assert gates['audit_sha256']==sha(EXP/'transfer-output-audit.json')
    for name,model in gates['models'].items():
        source='distillation-replay-report-evaluation.json' if name.endswith('-replay') else 'distillation-evaluation.json'
        assert model['source_result_sha256']==sha(EXP/'results'/source)
        original=read(EXP/'results'/source)['variants'][name]
        bykey={(r['clip'],r['frame']):r for r in original['transfer']}
        assert model['counts']==dict(Counter(r['verdict']+('_retained' if r['accepted'] else '_rejected') for r in model['rows']))
        for row,label in zip(model['rows'],transfer_audit['rows']):
            assert all(row[k]==v for k,v in label.items())
            old=bykey[(row['clip'],row['frame'])]
            assert row['pixel']==old['pixel'] and row['accepted']==old['accepted']
    surface=read(EXP/'results/surface-filter-evaluation.json')
    assert surface['protocol']==read(EXP/'surface-filter-protocol.json')
    assert surface['candidate_evaluation_sha256']==sha(EXP/'results/verifier-rerank-evaluation.json')
    source_candidates=read(EXP/'results/verifier-rerank-evaluation.json')
    assert surface['candidate_source_sha256']==source_candidates['source_sha256']
    surface_spec=importlib.util.spec_from_file_location('surface_selection',EXP/'score_surface.py')
    surface_module=importlib.util.module_from_spec(surface_spec);surface_spec.loader.exec_module(surface_module)
    bykey={(r['clip'],r['frame']):r for r in surface['rows']}
    for model in surface['models'].values():
        frames=[{**r,'candidates':[c for c in r['candidates'] if surface_module.inside_surface(
            c['pixel'],bykey[(r['clip'],r['frame'])],model['margin'])]}
            for r in source_candidates['models'][model['verifier']]['frames']]
        for radius in [12,25]:
            assert evaluation.evaluate(frames,labels,operating,radius)==model[f'at_{radius}px'];verified+=1
        assert model['audit_counts']==dict(Counter(r['verdict']+('_inside' if r['original_position_inside_surface'] else '_outside') for r in model['audit_original_positions']))
        for row in model['audit_original_positions']:
            assert row['original_position_inside_surface']==surface_module.inside_surface(row['pixel'],bykey[(row['clip'],row['frame'])],model['margin'])
    runtime = read(EXP/'results/sam-state-ablation.json')
    for run in runtime['runs'].values():
        assert run['valid_seed_masks'] == sum(r['valid'] for r in run['rows'])
        assert run['seeds'] == len(run['rows'])
    for filename in ['prompt-ablation.json','prompt-ablation-extended.json','prompt-ablation-sam2.json','tracking-ablations.json']:
        for variant in read(EXP/'results'/filename)['variants'].values():
            assert variant['counts'] == dict(Counter(r['verdict'] for r in variant['rows'])) | {
                k:sum(r['verdict']==k for r in variant['rows']) for k in ['correct','wrong','missing']}
            assert variant['comparisons'] == len(variant['rows'])
    public=ROOT/'web/data/ball-temporal-20261026'
    if public.exists():
        published=read(public/'positions.json');report=read(public/'report.json')
        packed_path=EXP/'results/fusion-inputs.json';packed=read(packed_path)
        assert sha(packed_path)==report['inputs']['fusion_inputs_sha256']
        assert sha(public/'positions.json')==report['inputs']['positions_sha256']
        assert sha(ROOT/'hockey_tracking/ball_refinement.py')==report['inputs']['implementation_sha256']
        assert sha(EXP/report['inputs']['config_file'])==report['inputs']['config_sha256']
        audit=read(EXP/report['inputs']['audit_file'])
        assert sha(EXP/report['inputs']['audit_file'])==report['inputs']['audit_sha256']
        if 'carried_forward_from' in audit:
            original=read(EXP/audit['carried_forward_from'])
            assert sha(EXP/audit['carried_forward_from'])==audit['original_audit_sha256']
            assert audit['rows']==original['rows'] and audit['counts']==original['counts']
        assert audit==report['output_audit']
        assert audit['candidate_positions_sha256']==sha(public/'positions.json')
        assert audit['counts']==dict(Counter(r['verdict'] for r in audit['rows']))
        assert report['reference_quality_audit']==quality
        for row in audit['rows']:
            point=published['frames'][row['frame']]
            assert row['state']==point['state']
            assert all(abs(a-b)<1e-8 for a,b in zip(row['pixel'],[point['pixel'][0]*1920,point['pixel'][1]*1080]))
        baseline=read(ROOT/'web/data/ball-sam-20261026/positions.json')
        rebuilt,diagnostics=refine_sequence(read(ROOT/'web/data/overhead-20261025/replay.json'),
            baseline,packed['seed_streams'],packed['track_streams'],packed['detector_frames'],
            RefinementConfig(**packed['config']),packed['excluded_frames'])
        assert_numeric_equal(rebuilt,published)
        assert report['after_states']==diagnostics['after_states']
        assert report['before_states']==diagnostics['before_states']
        assert_numeric_equal(report['isolated_observation_rejections'],diagnostics['isolated_observation_rejections'])
        assert len(published['frames'])==4500
        for f,row in enumerate(published['frames']):
            assert row['frame']==f
            if row['state']=='unknown': assert row.get('pixel') is None
            else: assert len(row['pixel'])==2 and all(0<=v<=1 for v in row['pixel'])
            if 466<=f<516: assert row['state']=='unknown'
            if baseline['frames'][f]['state']=='detected': assert row==baseline['frames'][f]
        for source in ['detector','yolo']:
            verification=read(EXP/f'results/{source}-seed-verification.json')
            assert verification['counts']==dict(Counter(r['verdict'] for r in verification['rows']))
            stream=next(s for s in packed['seed_streams'] if s['batch']==f'dense-{source}-verified-gpu')
            accepted={r['frame']:r for r in verification['rows'] if r['verdict']=='visible_ball'}
            assert {s['frame'] for s in stream['seeds']}==set(accepted)
            for seed in stream['seeds']:
                assert seed['pixel']==accepted[seed['frame']]['pixel']
                assert seed['support']==accepted[seed['frame']]['support']
        assert report['openai_cost_estimate_usd']<=50
    print(json.dumps({'localization_reports_recounted':verified,
                      'combined_detector_variants':len(combined['models']),
                      'runtime_controls':len(runtime['runs']),
                      'scope':'Recounts exported predictions and verifies selection/reference provenance; not new inference or independent ground truth.'}))


if __name__ == '__main__': main()
