"""Recount the SAM3 follow-up from predictions, frozen references and selections."""
from pathlib import Path
from collections import Counter
import hashlib
import importlib.util
import json
import math

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'web/data/reliability/sam3-20261024'
EXP=ROOT/'experiments/sam3-20261024'


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def objective(row):
    c=row['counts'];return c['correct']-10*c['wrong'],c['matched_observations'],-c['unknown_same']


def verify_detection():
    spec=importlib.util.spec_from_file_location('metrics',EXP/'nbhl_detection_metrics.py')
    metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
    selection=read(DATA/'finetune/selection.json')
    assert selection['protocol_sha256']==sha(EXP/'sam3-finetune-protocol-20261024.json')
    assert selection['extension_protocol_sha256']==sha(EXP/'sam3-finetune-extension-20261024.json')
    assert selection['new_regression_scores_seen'] is False
    def score(c):
        m=c['validation']['selected'];return m['f1'],m['precision'] or 0,m['confidence_threshold'],c['variant']=='original'
    assert max(selection['candidates'],key=score)==selection['best']
    features=read(DATA/'finetune/features.json')
    assert features['protocol_sha256']==selection['protocol_sha256']
    for f in features['frames']:
        assert sha(EXP/'annotations'/(f['source']+'.json'))==f['annotation_sha256']
    assert Counter(f['split'] for f in features['frames'])=={'train':392,'validation':23}
    for entry in selection['reports']:
        path=DATA/'finetune'/(entry['variant']+'.json');assert sha(path)==entry['report_sha256']
        r=read(path)
        assert r['changed_tensors']>0 and r['nonzero_finite_gradients']
        assert r['frozen_parameters_before_sha256']==r['frozen_parameters_after_sha256']
        assert r['features_manifest_sha256']==sha(DATA/'finetune/features.json')
        def epoch_score(row):
            m=row['validation']['selected'];return m['f1'],m['precision'] or 0,m['confidence_threshold']
        assert r['best']==max(r['history'],key=epoch_score)
        candidate=next(c for c in selection['candidates'] if c['variant']==r['variant'])
        assert candidate['epoch']==r['best']['epoch'] and candidate['validation']==r['best']['validation']
        assert candidate['delta_sha256']==r['delta_sha256']
    report=read(DATA/'finetune/regression.json')
    assert report['selection_sha256']==sha(DATA/'finetune/selection.json')
    refs={f['id']:f for source in ['1f0Y5BIr5Ps','5Qw_g9E7EqE'] for f in read(EXP/'annotations'/(source+'.json'))['frames']}
    for r in report['results']:
        rows=read(DATA/'finetune'/(r['variant']+'-predictions.json'))
        assert {row['id'] for row in rows}==set(refs)
        for row in rows:assert row['players']==refs[row['id']]['players']
        candidate=next(c for c in selection['candidates'] if c['variant']==r['variant'])
        assert r['threshold']==candidate['validation']['selected']['confidence_threshold']
        assert metrics.summarize(rows,r['threshold'])==r['metrics']
        assert math.isclose(metrics.average_precision(rows),r['ap50'],abs_tol=1e-12)
    return {'training_frames':196,'validation_frames':23,'regression_frames':36,'weight_training_variants':len(selection['reports'])}


def verify_tracking():
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    spec=importlib.util.spec_from_file_location('metrics',EXP/'nbhl_detection_metrics.py')
    metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
    reference=read(ROOT/'web/data/reliability/tracking-four-hour-final-reference-v1.json')
    pairs={r['id']:r for r in reference['pairs']}
    selection=read(DATA/'selection.json');development=read(DATA/'development.json')
    assert selection['protocol_sha256']==sha(EXP/'sam3-hybrid-protocol-20261024.json')
    assert selection['development_sha256']==sha(DATA/'development.json')
    assert selection['new_regression_scores_seen'] is False
    for r in selection['winners']:
        assert r==max([c for c in development['results'] if c['family']==r['family']],key=objective)
    for family in ['native','visual','visual-memory','adapted']:
        s=read(DATA/f'video-{family}/selection.json');d=read(DATA/f'video-{family}/development.json')
        assert s['selection_uses_regression'] is False
        assert s['development_sha256']==sha(DATA/f'video-{family}/development.json')
        assert s['best']==max(d['results'],key=lambda r:(*objective(r),r['variant']=='default'))
        if family=='adapted':
            assert s['extension_protocol_sha256']==sha(EXP/'sam3-adapted-person-extension-20261024.json')
            assert s['capacity_extension_protocol_sha256']==sha(EXP/'sam3-adapted-capacity-extension-20261024.json')
    summary=read(DATA/'results.json');tables=0
    for method in summary['tracking']:
        summed=Counter()
        for clip in method['clips']:
            r=read(DATA/'pairs'/(clip['clip']+'-'+method['method']+'.json'))
            expected_pair_ids={p['id'] for p in reference['pairs'] if p['clip']==clip['clip']}
            assert {p['id'] for p in r['pairs']}==expected_pair_ids and len(r['pairs'])==len(expected_pair_ids)
            predictions=read(DATA/'sampled-predictions'/(clip['clip']+'-'+method['method']+'.json'))
            refs=[o for o in reference['observations'] if o['clip']==clip['clip']]
            assert {f['frame'] for f in predictions['frames']}=={o['frame'] for o in refs}
            assignments={}
            for frame in predictions['frames']:
                obs=[o for o in refs if o['frame']==frame['frame']];objects=frame['objects']
                if not obs or not objects:continue
                overlaps=np.array([[metrics.iou(a['bbox'],b['bbox']) for b in objects] for a in obs])
                aa,bb=linear_sum_assignment(np.where(overlaps>=.5,1-overlaps,1e6))
                for a,b in zip(aa,bb):
                    if overlaps[a,b]>=.5:assignments[obs[a]['id']]=objects[b]['track_id']
            assert len(assignments)==r['counts']['matched_observations']==clip['counts']['matched_observations']
            assert len(refs)==r['counts']['reference_observations']==clip['counts']['reference_observations']
            counted=Counter(correct=0,wrong=0,unknown_same=0,same_total=0,different_total=0,missing=0)
            for p in r['pairs']:
                original=pairs[p['id']]
                assert all(p[k]==original[k] for k in ['a','b','truth','clip'])
                prediction=p['predicted_same'];truth=p['truth']
                a,b=assignments.get(p['a']),assignments.get(p['b'])
                assert prediction==(a==b if a is not None and b is not None else None)
                assert p['track_a']==a and p['track_b']==b
                assert prediction is None or type(prediction) is bool
                counted['same_total']+=truth=='same';counted['different_total']+=truth=='different'
                counted['missing']+=prediction is None
                if prediction:counted[{'same':'correct','different':'wrong','uncertain':'unknown_same'}[truth]]+=1
            for k,v in counted.items():assert v==r['counts'][k]==clip['counts'][k],(method['method'],clip['clip'],k)
            summed.update(counted);tables+=1
        for k,v in summed.items():assert v==method['counts'][k]
    return tables


def verify_fresh_audit():
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    spec=importlib.util.spec_from_file_location('metrics',EXP/'nbhl_detection_metrics.py')
    metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
    rp=EXP/'sam3-new-interval-audit-reference-20261024.json';reference=read(rp)
    report=read(DATA/'fresh-audit.json');predictions=read(DATA/'fresh-audit-predictions.json')
    assert sha(rp)==report['reference_sha256']
    assert sha(EXP/'sam3-new-interval-audit-protocol-20261024.json')==reference['protocol_sha256']==report['protocol_sha256']
    assert report['selection_uses_this_audit'] is False
    assert Counter(r['state'] for r in reference['observations'])==report['reference_counts']
    for r in report['results']:
        rows=predictions[r['method']];assert rows['source_tracks_sha256']==report['inputs'][r['method']]
        assigned={}
        for frame in rows['frames']:
            refs=[a for a in reference['observations'] if a['state']=='visible' and a['frame']==frame['frame']]
            objects=frame['objects']
            if not refs or not objects:continue
            overlaps=np.array([[metrics.iou(a['bbox'],b['bbox']) for b in objects] for a in refs])
            aa,bb=linear_sum_assignment(np.where(overlaps>=r['iou_threshold'],1-overlaps,1e6))
            for a,b in zip(aa,bb):
                if overlaps[a,b]>=r['iou_threshold']:assigned[(refs[a]['identity'],frame['frame'])]=objects[b]['track_id']
        assert len(assigned)==r['matched_observations']
        events=Counter()
        for identity in reference['identities']:
            seq=sorted([a for a in reference['observations'] if a['identity']==identity],key=lambda a:a['frame'])
            for a,b in zip(seq,seq[1:]):
                if a['state']!='visible' or b['state']!='visible':continue
                x=assigned.get((identity,a['frame']));y=assigned.get((identity,b['frame']))
                state=(('missing_both' if x is None and y is None else 'missing_one') if x is None or y is None else ('same_track' if x==y else 'identity_change'))
                events[state]+=1
        assert dict(events)==r['sampled_transitions']
    return len(reference['observations'])


def verify_replays():
    count=0
    assets=read(ROOT/'site-assets.json')['files']
    for clip in read(DATA/'replay-catalog.json'):
        folder='media/sam3-20261024' if clip['clip']=='pittsburgh-sam3-20min' else 'media/tracking-fourhour'
        for method in clip['methods']:
            rel=folder+'/'+clip['clip']+'-'+method['id']+'.json'
            if rel in assets:continue # The pinned asset builder verifies this existing baseline.
            r=read(ROOT/'web'/rel)
            assert r['clip']==clip['clip'] and r['method']==method['id'] and r['fps']==15
            assert len(r['frames'])==clip['frames'] and r['fragment_ids_only'] and not r['manually_corrected']
            assert len(set(r['track_id_map'].values()))==len(r['track_id_map'])
            for frame in r['frames']:
                assert len({b[0] for b in frame})==len(frame)
                assert all(len(b)==5 and all(type(x) is int for x in b) and 0<=b[1]<b[3]<=10000 and 0<=b[2]<b[4]<=10000 for b in frame)
            count+=len(r['frames'])
    return count


def verify_rink_filter():
    import base64,zlib
    import numpy as np
    spec=importlib.util.spec_from_file_location('rink_filter',EXP/'nbhl_sam3_rink_filter.py')
    filtering=importlib.util.module_from_spec(spec);spec.loader.exec_module(filtering)
    spec=importlib.util.spec_from_file_location('metrics',EXP/'nbhl_detection_metrics.py')
    metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
    selection=read(DATA/'rink-filter/selection.json');development=read(DATA/'rink-filter/development.json')
    assert selection['protocol_sha256']==sha(EXP/'sam3-rink-filter-protocol-20261024.json')
    assert selection['development_sha256']==sha(DATA/'rink-filter/development.json')
    assert selection['prior_regression_results_seen'] and not selection['filter_regression_scores_seen']
    chosen=max(development['results'],key=lambda r:(r['metrics']['f1'],r['metrics']['precision'] or 0,r['setting']['key']=='unchanged'))
    assert chosen==selection['best']
    if not selection['improves_validation_f1']:return {'promoted':False}
    variant=read(DATA/'finetune/selection.json')['best']['variant']
    comparisons=0
    for phase in ['development','regression']:
        prediction_name=variant+('-validation-predictions.json' if phase=='development' else '-predictions.json')
        original=read(DATA/'finetune'/prediction_name)
        packed=read(DATA/'rink-filter'/(phase+'-masks.json'));masks={}
        for r in packed:
            bits=np.frombuffer(zlib.decompress(base64.b64decode(r['packed_bits_zlib_base64'])),dtype=np.uint8)
            mask=np.unpackbits(bits).reshape(r['mask_shape']).astype(bool)
            masks[(r['id'],r['prompt'],r['mask_threshold'])]=(mask,r)
        report=read(DATA/'rink-filter'/(phase+'.json'))
        for candidate in report['results']:
            setting=candidate['setting'];rows=[]
            for row in original:
                if setting['prompt']:
                    mask,info=masks[(row['id'],setting['prompt'],setting['mask_threshold'])]
                    value,_=filtering.filter_row(row,mask,info['width'],info['height'],setting['dilation'])
                else:value=row
                rows.append(value)
            assert metrics.summarize(rows,candidate['metrics']['confidence_threshold'])==candidate['metrics']
            if phase=='regression' and setting['prompt']:
                assert rows==read(DATA/'rink-filter/regression-filtered-predictions.json')
            comparisons+=1
    return {'promoted':True,'recomputed_comparisons':comparisons,'post_regression_followup':True}


def verify_fresh_detection():
    import base64,zlib
    import numpy as np
    modules={}
    for name in ['nbhl_detection_metrics','nbhl_sam3_rink_filter']:
        spec=importlib.util.spec_from_file_location(name,EXP/(name+'.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);modules[name]=module
    metrics=modules['nbhl_detection_metrics'];filtering=modules['nbhl_sam3_rink_filter']
    reference_path=EXP/'sam3-fresh-detection-audit-reference-20261024.json'
    reference=read(reference_path);report=read(DATA/'fresh-detection/report.json')
    assert sha(reference_path)==report['reference_sha256']
    assert sha(EXP/'sam3-fresh-detection-audit-protocol-20261024.json')==report['protocol_sha256']==reference['protocol_sha256']
    assert report['selection_uses_this_audit'] is False
    assert not reference['sam3_predictions_seen'] and not reference['rink_filter_predictions_seen']
    assert reference['proposal_assisted'] and reference['counts']['players_and_goalies']==77
    assert report['fine_tune_selection_sha256']==sha(DATA/'finetune/selection.json')
    assert report['rink_selection_sha256']==sha(DATA/'rink-filter/selection.json')
    assert report['predictions_sha256']==sha(DATA/'fresh-detection/predictions.json')
    assert report['masks_sha256']==sha(DATA/'fresh-detection/masks.json')
    predictions=read(DATA/'fresh-detection/predictions.json');refs={f['id']:f for f in reference['frames']}
    assert sum(len(f['players']) for f in refs.values())==77
    fine=read(DATA/'finetune/selection.json');rink=read(DATA/'rink-filter/selection.json')
    thresholds={fine['candidates'][0]['variant']:fine['candidates'][0]['validation']['selected']['confidence_threshold'],
                fine['best']['variant']:fine['best']['validation']['selected']['confidence_threshold'],
                'fine-tuned-rink':fine['best']['validation']['selected']['confidence_threshold'],'yolo-pipeline':.1}
    for result in report['results']:
        assert result['threshold']==thresholds[result['variant']]
        if result['variant']=='fine-tuned-rink':assert result['setting']==rink['best']['setting']
        rows=predictions[result['variant']]
        assert {r['id'] for r in rows}==set(refs) and len(rows)==11
        for row in rows:assert row['players']==refs[row['id']]['players']
        assert metrics.summarize(rows,result['threshold'])==result['metrics']
        assert math.isclose(metrics.average_precision(rows),result['ap50'],abs_tol=1e-12)
    assert fine['frozen_at']<reference['sealed_at'] and rink['frozen_at']<reference['sealed_at']
    fine_rows={r['id']:r for r in predictions[fine['best']['variant']]}
    expected={r['id']:r for r in predictions['fine-tuned-rink']}
    for row in read(DATA/'fresh-detection/masks.json'):
        assert row['frame_sha256']==refs[row['id']]['sha256']
        bits=np.frombuffer(zlib.decompress(base64.b64decode(row['packed_bits_zlib_base64'])),dtype=np.uint8)
        mask=np.unpackbits(bits).reshape(row['mask_shape']).astype(bool)
        filtered,_=filtering.filter_row(fine_rows[row['id']],mask,1920,1080,rink['best']['setting']['dilation'])
        assert filtered==expected[row['id']]
    control=read(DATA/'fresh-detection/baseline-control.json');frozen=read(EXP/'detector-heldout-freeze.json')
    assert control['baseline_freeze_sha256']==sha(EXP/'detector-heldout-freeze.json')
    spec=next(r for r in frozen['models'] if r['name']=='selected_finetuned')
    assert control['threshold']==spec['confidence_threshold']==.65
    assert control['weights_sha256']==spec['weights_sha256']
    assert frozen['frozen_at']<reference['sealed_at']
    assert control['added_after_audit_results'] and not control['threshold_selected_using_this_audit']
    assert metrics.summarize(predictions['yolo-pipeline'],control['threshold'])==control['metrics']
    return reference['counts']


def main():
    for name,digest in read(DATA/'export-manifest.json').items():assert sha(DATA/name)==digest,name
    result=verify_detection()
    result['tracking_pair_tables']=verify_tracking()
    result['fresh_audit_observations']=verify_fresh_audit()
    result['replay_frames']=verify_replays()
    result['rink_filter']=verify_rink_filter()
    result['fresh_detection']=verify_fresh_detection()
    assert read(DATA/'long-video.json')['frames']==18225
    assert read(DATA/'long-run.json')['frames']==18225
    assert read(DATA/'memory-score-cache-verification.json')['total_equivalent_frames']==1200
    assert read(DATA/'memory-score-cache-cpu-verification.json')['exact_memory_selection_checks']==48000
    print(json.dumps(result))


if __name__=='__main__':main()
