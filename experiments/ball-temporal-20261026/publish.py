"""Publish a reviewed replay plus compact, fully recountable fusion inputs."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD = ROOT/'build/ball-temporal-20261026'
sys.path.insert(0,str(ROOT))
from hockey_tracking.ball_refinement import refine_sequence, RefinementConfig


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,separators=(',',':'))+'\n')


def reference_score(data,labels):
    details=[]
    for label in labels:
        row=data['frames'][label['frame']]
        point=[row['pixel'][0]*1920,row['pixel'][1]*1080] if row.get('pixel') else None
        error=math.dist(point,label['pixel']) if point else None
        details.append({'frame':label['frame'],'error_px':error,
                        'outcome':'missing' if error is None else 'correct' if error<=25 else 'wrong'})
    return {'radius_px':25,'counts':dict(Counter(r['outcome'] for r in details)),'rows':details}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--tracker',default='sam3-box6')
    parser.add_argument('--audit',required=True)
    parser.add_argument('--batches',default='dense-full-gpu,dense-discovery-gpu')
    args=parser.parse_args()
    candidate=BUILD/args.candidate
    positions=read(candidate/'positions.json')
    raw_report=read(candidate/'report.json')
    audit_path=HERE/args.audit;audit=read(audit_path)
    assert audit['candidate_positions_sha256']==sha(candidate/'positions.json')
    config=RefinementConfig(**raw_report['config'])
    packed={'config':raw_report['config'],'seed_streams':[],'track_streams':[],
            'detector_frames':[],'excluded_frames':list(range(466,516)),
            'raw_input_sha256':raw_report['input_sha256']}
    crop_sizes=[]
    for batch in args.batches.split(','):
        seeds=read(BUILD/f'tracking-seeds-{batch}.json')
        tracks=read(BUILD/f'short-{args.tracker}-{batch}.json')
        crop_sizes.append(seeds['crop_size'])
        packed['seed_streams'].append({'batch':batch,'seeds':[
            {k:r[k] for k in ['frame','pixel','support','appearance_score','evidence','source_sha256','source'] if k in r}
            for r in seeds['seeds']]})
        packed['track_streams'].append({'complete':tracks['complete'],
            'protocol':{k:tracks['protocol'][k] for k in ['batch','offload_state_to_cpu']},
            'tracks':[{'seed_frame':t['seed_frame'],'seed_pixel':t['seed_pixel'],
                'rows':[{k:r[k] for k in ['frame','pixel','visible','shape_valid','inside_image','inside_crop','appearance_score']}
                        for r in t['rows']]} for t in tracks['tracks']]})
    for frame in read(BUILD/'rgb-expanded/replay.json')['frames']:
        packed['detector_frames'].append({'candidates':[r for r in frame['candidates']
            if r['score']>=config.detector_score and r['appearance_score']>=config.detector_appearance]})
    rink=read(ROOT/'web/data/overhead-20261025/replay.json')
    baseline=read(ROOT/'web/data/ball-sam-20261026/positions.json')
    rebuilt,diagnostics=refine_sequence(rink,baseline,packed['seed_streams'],packed['track_streams'],
                                       packed['detector_frames'],config,packed['excluded_frames'])
    assert rebuilt==positions,'Compact inputs must reproduce the selected replay exactly.'
    input_path=HERE/'results/fusion-inputs.json';write(input_path,packed)
    write(HERE/'results/selected-fusion.json',raw_report)
    labels=read(HERE/'reused-reference.json')['rows']
    before=raw_report['before_states'];after=raw_report['after_states']
    old_visual=before['detected']+before['tracked'];new_visual=after['detected']+after['tracked']
    report={'created_at':datetime.now(timezone.utc).isoformat(),
        'source_id':positions['source_id'],'fps':15,'frames':4500,
        'method':positions['method'],'tracker_variant':args.tracker,
        'crop_sizes':crop_sizes,
        'decision':{'promoted':True,'summary':
            f'{new_visual} detected or video-tracked frames, up from {old_visual}. Held-ball and gap estimates stay separate. '
            f'Separate-game checks still expose many false detections.'},
        'before_states':before,'after_states':after,
        'estimated_coverage':(4500-after['unknown'])/4500,
        'visual_and_tracked_ratio':new_visual/old_visual,
        'output_audit':audit,
        'same_game_reference':{'scope':'94 reused approximate assistant labels, including disputed labels; raw historical agreement only, not an accuracy estimate.',
            'previous':reference_score(baseline,labels),'improved':reference_score(positions,labels)},
        'reference_quality_audit':read(HERE/'reference-quality-audit.json'),
        'transfer_output_audit':{k:v for k,v in read(HERE/'transfer-output-audit.json').items() if k!='rows'},
        'surface_filter':{'visible_correct_wrong_missing':[29,3,1],
            'audited_wrong_positions_removed':32,'audited_wrong_positions':52,
            'audited_visible_positions_removed':1,'audited_visible_positions':3,
            'scope':'Post-audit exploratory setting; one airborne ball rejected. No default replay change.'},
        'detector_new_games':read(HERE/'results/all-intervals.json'),
        'seed_verification':{source:{k:r for k,r in read(HERE/f'results/{source}-seed-verification.json').items() if k!='rows'}
            for source in ['detector','yolo']},
        'language_model_adjudication':{'baseline_correct_wrong_missing':[22,4,1],
            'with_adjudication_correct_wrong_missing':[23,3,1],
            'scope':'Eight detector-disagreement requests on the already opened 120-sample set; only two evaluable target frames. Exploratory, not independent.'},
        'distillation':{'selected_variant':'temporal-head','decision':'Rejected: too many missed balls.',
            'transfer_correct_wrong_missing':[14,0,19],
            'replay_training_followup':{'encoder_gate':[22,2,9],'encoder_rerank':[24,5,4],
                'multiview_original':[27,5,1],'decision':'None beats the frozen full-frame baseline.'},
            'scope':'Model-generated labels, small time-separated same-game test and previously opened transfer frames.'},
        'automatic_tracker_transfer':{k:{m:v for m,v in r['at_25px'].items() if m!='details'}
            for k,r in read(HERE/'results/transfer-evaluation.json')['models'].items()},
        'runtime_fix':{'before_valid_seeds':[15,6],'after_valid_seeds':[29,29,29],
            'seed_count':29,'identical_gpu_repeat_and_synchronized_cpu_positions':377,
            'scope':'Matched runtime controls, not unprompted ball accuracy.'},
        'openai_cost_estimate_usd':read(HERE/'results/yolo-seed-verification.json')['total_openai_estimate_usd'],
        'cost_note':'Conservative estimate including reserved cost for incomplete pilot calls; not an invoice. Modal compute is separate.',
        'inputs':{'fusion_inputs_sha256':sha(input_path),'positions_sha256':sha(candidate/'positions.json'),
            'config_sha256':sha(HERE/'fusion-config-v2.json'),'audit_sha256':sha(audit_path),
            'implementation_sha256':sha(ROOT/'hockey_tracking/ball_refinement.py')},
        'limits':['Sparse assistant review, not independent human ground truth.',
            'Policies and prompt choice were tuned using development checks.',
            'SAM can consistently follow the wrong object; agreement is not a calibrated probability.',
            'Context and interpolation are estimates, not observations.',
            'No validated shots, goals, possession, speed or metric ball distance.']}
    destination=ROOT/'web/data/ball-temporal-20261026'
    write(destination/'positions.json',positions)
    write(destination/'report.json',report)
    print(json.dumps({'published':str(destination.relative_to(ROOT)),'visual_before':old_visual,
                      'visual_after':new_visual,'audit':audit['counts'],'coverage':report['estimated_coverage']}))


if __name__=='__main__':main()
