"""Export reproducible candidate selection, replay, and explicitly scoped evaluation."""
from pathlib import Path
import sys,json,hashlib,collections
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from hockey_tracking.ball_path import select_path,PathConfig,fuse_candidates,filter_person_parts,filter_markings,filter_verified
from hockey_tracking.ball_track import make_replay
from hockey_tracking.ball_review import apply_review,additional_visual_proposals,merge_reviewed,require_floor_support
from evaluate import score,evaluate_regression,multiview_regression,verified_regression


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,separators=(',',':'))+'\n')


def main(folder):
    exp=Path(__file__).parent;choice=json.loads((exp/'selection.json').read_text())
    source=ROOT/'web/data/overhead-20261025/replay.json';replay=json.loads(source.read_text())
    streams=[json.loads((folder/f'{name}.json').read_text())['frames'] for name in choice['sources']]
    fused=fuse_candidates(streams);raw_fused=fused
    appearance=json.loads((folder/choice['appearance']).read_text())['frames'] if choice.get('appearance') else None
    if choice.get('verifier'):
        verification=json.loads((folder/(choice['verifier']+'.json')).read_text())
        fused=filter_verified(fused,verification['frames'],choice['verification_threshold'],appearance,choice.get('shape_config'))
    elif appearance is not None:fused=filter_markings(fused,appearance)
    people=json.loads((folder/choice['person_context']).read_text())['frames']
    candidates=filter_person_parts(fused,people)
    path=select_path(candidates,replay,PathConfig(**choice['path_config']),excluded_frames=range(466,516))
    result=make_replay(replay,path['detections'],choice.get('gap_seconds',.2),choice.get('max_inferred_radius_px',24))
    cv_result=result
    if choice.get('visual_review'):
        ledger=json.loads((exp/choice['visual_review']).read_text())
        encoded=(json.dumps(result,separators=(',',':'))+'\n').encode()
        assert hashlib.sha256(encoded).hexdigest()==ledger['protocol']['source_replay_sha256']
        result=apply_review(result,ledger)
    primary_reviewed=result;dest=ROOT/'web/data/ball-rebuild-20261026'
    if choice.get('recall'):
        extra=choice['recall']
        extra_candidates=filter_person_parts(filter_verified(raw_fused,verification['frames'],extra['verification_threshold'],appearance,choice.get('shape_config')),people)
        extra_path=select_path(extra_candidates,replay,PathConfig(**extra['path_config']),excluded_frames=range(466,516))
        sensitive=make_replay(replay,extra_path['detections'],choice.get('gap_seconds',.2),choice.get('max_inferred_radius_px',24))
        extra_proposals=additional_visual_proposals(cv_result,sensitive)
        extra_ledger=json.loads((exp/extra['visual_review']).read_text())
        assert hashlib.sha256((json.dumps(extra_proposals,separators=(',',':'))+'\n').encode()).hexdigest()==extra_ledger['protocol']['source_replay_sha256']
        extra_reviewed=apply_review(extra_proposals,extra_ledger)
        extra_before_support=extra_reviewed
        if extra.get('require_floor_support'):extra_reviewed=require_floor_support(extra_reviewed)
        save(dest/'recall-candidates.json',extra_candidates);save(dest/'recall-proposals-replay.json',extra_proposals);save(dest/'recall-reviewed-replay.json',extra_reviewed)
        if extra['enabled']:result=merge_reviewed(result,extra_reviewed)
    save(dest/'candidates.json',candidates);save(dest/'replay.json',result);save(dest/'proposals-replay.json',cv_result)
    labels=[]
    for name in ['review','targeted']:labels.extend(json.loads((exp/f'{name}-labels.json').read_text())['rows'])
    baseline=json.loads((ROOT/'web/data/ball-context-20261026/replay.json').read_text())
    metrics={}
    for name,dataset in [('baseline',baseline),('cv_before_review',cv_result),('primary_review_only',primary_reviewed),('improved',result)]:
        metrics[name]={}
        for split in ['development','test']:
            subset=[r for r in labels if r['split']==split]
            points=[]
            for label in subset:
                r=dataset['frames'][label['frame']];points.append([r['pixel'][0]*1920,r['pixel'][1]*1080] if r.get('pixel') else None)
            metrics[name][split]=score(subset,points)
    save(folder/'multi-view-regression.json',multiview_regression(folder))
    if choice.get('verifier'):
        save(folder/'verified-multi-view-regression.json',verified_regression(folder,choice['verifier'],choice['verification_threshold']))
        save(exp/'predictions'/(choice['verifier']+'-regression.json'),json.loads((folder/(choice['verifier']+'-regression.json')).read_text()))
    regression=evaluate_regression(folder)
    # Preserve raw sparse predictions for reproducible model comparisons.
    for name in regression:
        save(exp/'predictions'/f'{name}-regression.json',json.loads((folder/f'{name}-regression.json').read_text()))
    report={'schema_version':1,'source_id':replay['source_id'],'start_s':900,'duration_s':300,'fps':15,
            'selection':choice,'states':dict(collections.Counter(r['state'] for r in result['frames'])),
            'baseline_states':dict(collections.Counter(r['state'] for r in baseline['frames'])),
            'image_only_detections':sum(r['state']=='detected' and r['xy'] is None for r in result['frames']),
            'review':metrics,'regression':regression,'review_points':sorted([r for r in labels if r['visibility']=='visible'],key=lambda r:r['frame']),
            'independent_human_ground_truth':False,'uncertainty_calibrated':False,'match_radius_native_px':12,
            'input_sha256':digest(source),'candidates_sha256':digest(dest/'candidates.json'),
            'algorithm_sha256':{name:digest(ROOT/'hockey_tracking'/name) for name in ['ball_path.py','ball_track.py','ball_context.py','ball_review.py']},
            'experiment_code_sha256':{name:digest(exp/name) for name in ['modal_experiment.py','modal_alternatives.py','modal_people.py','modal_appearance.py','modal_verifier.py','crop_verifier.py','evaluate.py']},
            'raw_inputs_sha256':{name:digest(folder/name) for name in [*[s+'.json' for s in choice['sources']],choice['person_context'],*([choice['appearance']] if choice.get('appearance') else []),*([choice['verifier']+'.json'] if choice.get('verifier') else [])]},
            'limitations':['Existing detector weights trained on 92 visible NBHL frames; a separate crop verifier is newly trained on NBHL examples and hard negatives.',
                           'Sparse assistant-reviewed labels with approximate centers. Unknown visibility excluded, not a negative.',
                           'Eight current-game development positives and 15 development audit positives now train the verifier. Later audit errors informed model development. The nine-point temporal check is reused, not a new independent test.',
                           'Cross-game labels are existing regression sets from four recordings excluded from verifier weight training.',
                           'Coverage is output frequency, not accuracy. Wrong-object tracks remain possible.',
                           'Person filtering assumes a floor ball and may withhold a real ball near shoes, in a hand, or in flight.',
                           'Color-shape rejection targets this rink\'s red lines and elongated stick tape; it may withhold a ball on a marking or a severely blurred ball.',
                           'Error ellipses assume correct visual anchors and floor geometry. They are not calibrated probability guarantees.',
                           'Does not establish possession, shots, goals, assists, or metric ball speed.']}
    if choice.get('verifier'):
        training=exp/'training'/choice['verifier']/'completed.json'
        report['verifier_training']=json.loads(training.read_text())
        report['training_report_sha256']=digest(training)
        report['training_samples_sha256']=digest(training.parent/'training-samples.json')
    if choice.get('visual_review'):
        records=list(ledger['requests'].values())
        report['visual_review']={'model':ledger['protocol']['model'],'requests':len(records),
            'completed':sum(r['status']=='completed'for r in records),
            'estimated_usd':sum(r.get('estimated_usd',r.get('reserved_usd',0))for r in records),
            'ledger_sha256':digest(exp/choice['visual_review']),
            'pre_review_states':dict(collections.Counter(r['state']for r in cv_result['frames'])),
            'label_counts':dict(collections.Counter(s['object']for r in records if r.get('answer')for s in r['answer']['samples']))}
        report['experiment_code_sha256']['modal_vlm_review.py']=digest(exp/'modal_vlm_review.py')
        report['limitations'].append('OpenAI reviews at most five sampled positions per short candidate episode. Decisions extend to nearby frames, can be wrong, and are not ground truth. Missing/uncertain decisions withhold output. Non-floor support has no overhead position.')
    if choice.get('recall'):
        records=list(extra_ledger['requests'].values())
        report['recall_review']={'published':choice['recall']['enabled'],'model':extra_ledger['protocol']['model'],
            'requests':len(records),'completed':sum(r['status']=='completed' for r in records),
            'proposed_detected':sum(r['state']=='detected' for r in extra_proposals['frames']),
            'accepted_detected':sum(r['state']=='detected' for r in extra_reviewed['frames']),
            'accepted_before_support_gate':sum(r['state']=='detected' for r in extra_before_support['frames']),
            'primary_states':dict(collections.Counter(r['state']for r in primary_reviewed['frames'])),
            'estimated_usd':sum(r.get('estimated_usd',r.get('reserved_usd',0))for r in records),
            'ledger_sha256':digest(exp/choice['recall']['visual_review']),
            'candidates_sha256':digest(dest/'recall-candidates.json')}
    audit=exp/'output-audit-final.json'
    if audit.exists():report['output_audit']=json.loads(audit.read_text())
    report['baseline_output_audit']=json.loads((exp/'output-audit-baseline.json').read_text())
    save(dest/'report.json',report)
    print(json.dumps({'states':report['states'],'review':{n:{s:{k:v for k,v in r.items() if k!='details'} for s,r in group.items()}for n,group in metrics.items()}},indent=2))


if __name__=='__main__':main(Path(sys.argv[1]))
