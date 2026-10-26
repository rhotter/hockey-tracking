"""Reconstruct the public replay from saved numeric model results."""
from pathlib import Path
import sys,json,hashlib,collections,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from hockey_tracking.ball_sequence import build_context_replay,fuse_point_tracks,SequenceConfig,PointFusionConfig
from evaluate import point_scores,context_scores,reference
EXP=Path(__file__).parent

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,separators=(',',':'))+'\n')
def read(path):return json.loads(path.read_text())


def merge_ledgers(results):
    names=['context-long-ledger','context-rest-ledger','context-repair-ledger'];requests={};sources=[]
    for name in names:
        path=results/(name+'.json')
        if not path.exists():continue
        sources.append(path.name)
        for key,record in read(path)['requests'].items():
            if key in requests:
                assert name=='context-repair-ledger' and requests[key]['status']!='completed'
                if record['status']!='completed':continue
            requests[key]={**record,'source_ledger':path.name}
    return {'schema_version':1,'source_ledgers':sources,'requests':dict(sorted(requests.items()))}


def appearance_ablation(verified,tracks):
    seeds=[t['seed_frame']for t in tracks['tracks']]
    labels=[r for r in reference()if all(abs(r['frame']-s)>3 for s in seeds)]
    results=[]
    for threshold in [.1,.3,.5,.7,.9]:
        predictions={}
        for row in verified['frames']:
            proposals=[p for p in row['proposals']if p['visible']and p['appearance_score']>=threshold]
            if proposals:predictions[row['frame']]=max(proposals,key=lambda p:p['appearance_score'])
        details=[]
        for r in labels:
            if r['frame']in predictions:
                p=predictions[r['frame']];details.append({**r,'prediction':p['pixel'],'error_px':float(np.linalg.norm(np.array(p['pixel'])-r['pixel']))})
        errors=[r['error_px']for r in details]
        results.append({'threshold':threshold,'emitted_frames':len(predictions),'reviewed_positions':len(labels),'answered':len(errors),
                        'within_25px':sum(e<=25 for e in errors),'median_error_px':float(np.median(errors))if errors else None,'details':details})
    return results


def derive():
    results=EXP/'results';source=ROOT/'web/data/overhead-20261025/replay.json';old=ROOT/'web/data/ball-rebuild-20261026/replay.json'
    replay,baseline=read(source),read(old);ledger=read(results/'context-ledger.json');verified=read(results/'tapnextpp512-verified.json')
    context=build_context_replay(replay,ledger,SequenceConfig(**read(EXP/'sequence-config.json')),range(466,516))
    combined=fuse_point_tracks(replay,baseline,context,verified,PointFusionConfig(**read(EXP/'point-fusion-config.json')))
    point_models={name:point_scores(read(results/(name+'.json')))for name in ['cotracker3-pilot','tapnextpp512-pilot','tapnextpp512-all']}
    ablation=appearance_ablation(verified,read(results/'tapnextpp512-all.json'))
    comparisons=[{'name':label,**{k:v for k,v in point_models[name]['scores']['visible'].items()if k!='details'}}for name,label in [
        ('cotracker3-pilot','CoTracker3 · 6 seeds'),('tapnextpp512-pilot','TAPNext++ · 6 seeds'),('tapnextpp512-all','TAPNext++ · 31 seeds')]]
    comparisons.append({'name':'TAPNext++ + appearance · 31 seeds',**{k:v for k,v in next(r for r in ablation if r['threshold']==.3).items()if k!='details'}})
    states=dict(collections.Counter(r['state']for r in combined['frames']));before=dict(collections.Counter(r['state']for r in baseline['frames']))
    records=[r for r in ledger['requests'].values()if r['status']=='completed'];intervals=[r['core']for r in records]
    report={'schema_version':1,'source_id':replay['source_id'],'start_s':900,'duration_s':300,'fps':15,
        'states':states,'baseline_states':before,'estimated_coverage':1-states.get('unknown',0)/4500,'baseline_estimated_coverage':1-before['unknown']/4500,
        'context_states':dict(collections.Counter(r['state']for r in context['frames'])),
        'context_play_states':dict(collections.Counter(r['play_state']for r in context['frames']if r['state']=='context')),
        'context_radius_px':{k:float(np.quantile([r['model_radius_px']for r in context['frames']if r['state']=='context'],q))for k,q in [('median',.5),('p90',.9)]},
        'sequences':[{'core':r['core'],'summary':r['answer']['sequence_summary']}for r in sorted(records,key=lambda r:r['core'][0])],
        'context_checks':context_scores(context,intervals),'context_checks_previously_unknown':context_scores(context,intervals,baseline),
        'context_checks_frames_not_shown_to_model':context_scores(context,intervals,exclude_frames={im['frame']for r in records for im in r['images']}),
        'point_models':point_models,'appearance_ablation':ablation,'point_comparisons':comparisons,
        'independent_human_ground_truth':False,'uncertainty_calibrated':False,
        'input_sha256':{str(p.relative_to(ROOT)):digest(p)for p in [source,old,*sorted(results.glob('*.json')),EXP/'sequence-config.json',EXP/'point-fusion-config.json']},
        'algorithm_sha256':{'ball_sequence.py':digest(ROOT/'hockey_tracking/ball_sequence.py')},
        'experiment_code_sha256':{p.name:digest(p)for p in sorted(EXP.glob('*.py'))},
        'api_accounting':{},
        'limitations':['The sequence model sees 2-fps chronological samples in overlapping 16-second windows and occasional existing visual proposals. It infers ball state every half-second; carrier motion and camera compensation reconstruct the intermediate frames.',
          'Contextual estimates, possession states, weights, and error radii are model hypotheses. Invisible ball positions have no independent ground truth.',
          'The reused localization checks mostly concern previously detected positions. They do not measure accuracy of newly inferred intervals. A separate conditional audit inspects newly emitted frames.',
          'Point-tracker appearance thresholds were compared on reused assistant labels; this is development selection, not independent validation. Seed frames and their +-3 neighbors are excluded from point checks.',
          'All experiments here use one previously examined NBHL recording. No new cross-game generalization claim.',
          'A camera cut resets inference. Held, airborne and height-uncertain hypotheses have no floor projection. Floor geometry remains approximate.',
          'Output coverage includes stoppages and inferred regions. It is not visible-ball recall, possession accuracy, shot/goal accuracy, or a calibrated probability guarantee.']}
    for name in ['context-timeout-ledger','context-stream-ledger','context-long-ledger','context-rest-ledger','context-repair-ledger']:
        ledger_file=results/(name+'.json')
        if ledger_file.exists():
            requests=list(read(ledger_file)['requests'].values())
            report['api_accounting'][name]={'requests':len(requests),'statuses':dict(collections.Counter(r['status']for r in requests)),
                'estimated_or_reserved_usd':sum(r.get('estimated_usd',r.get('reserved_usd',0))for r in requests)}
    audit=EXP/'output-audit.json'
    if audit.exists():report['output_audit']=read(audit)
    return context,combined,report


def main(folder=None):
    if folder:
        results=EXP/'results';results.mkdir(exist_ok=True)
        for name in ['cotracker3-pilot','tapnextpp512-pilot','tapnextpp512-all','tapnextpp512-verified','context-timeout-ledger','context-stream-ledger','context-long-ledger','context-rest-ledger']:
            shutil.copyfile(folder/(name+'.json'),results/(name+'.json'))
        if (folder/'context-repair-ledger.json').exists():shutil.copyfile(folder/'context-repair-ledger.json',results/'context-repair-ledger.json')
        ledger=merge_ledgers(results)
        assert len(ledger['requests'])==30 and all(r['status']=='completed'for r in ledger['requests'].values())
        save(results/'context-ledger.json',ledger)
    context,combined,report=derive();dest=ROOT/'web/data/ball-sequence-20261026'
    save(dest/'context-replay.json',context);save(dest/'replay.json',combined);save(dest/'report.json',report)
    print(json.dumps({'states':report['states'],'coverage':report['estimated_coverage'],'context_radius_px':report['context_radius_px']}))

if __name__=='__main__':main(Path(sys.argv[1])if len(sys.argv)>1 else None)
