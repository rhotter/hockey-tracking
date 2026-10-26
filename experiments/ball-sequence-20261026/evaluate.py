"""Scoped localization/region checks; output frequency is never accuracy."""
from pathlib import Path
import json,sys,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from hockey_tracking.ball_sequence import build_context_replay,merge_visual_and_context,SequenceConfig,fuse_point_tracks,PointFusionConfig


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,separators=(',',':'))+'\n')


def reference():
    exp=ROOT/'experiments/ball-rebuild-20261026';labels={}
    for name in ['review-labels','targeted-labels']:
        for row in json.loads((exp/(name+'.json')).read_text())['rows']:
            if row['visibility']=='visible':labels[row['frame']]={'frame':row['frame'],'pixel':row['center'],'source':'reused_raw_review'}
    for name in ['development-audit','output-audit-v1','output-audit-v4','output-audit-baseline','output-audit-final','primary-height-audit']:
        for row in json.loads((exp/(name+'.json')).read_text())['rows']:
            if row['verdict']=='visible_ball' and row['frame']not in labels:
                labels[row['frame']]={'frame':row['frame'],'pixel':row['pixel'],'source':'reused_output_review'}
    return list(sorted(labels.values(),key=lambda x:x['frame']))


def point_scores(data):
    seeds=[t['seed_frame']for t in data['tracks']];predictions={}
    for track in data['tracks']:
        for row in track['rows']:
            f=row['frame'];distance=abs(f-track['seed_frame'])
            if f not in predictions or distance<predictions[f]['seed_distance']:
                predictions[f]={**row,'seed_distance':distance,'seed_frame':track['seed_frame']}
    labels=[r for r in reference()if r['frame']in predictions and all(abs(r['frame']-s)>3 for s in seeds)]
    reports={}
    for mode in ['visible','all_latent']:
        rows=[]
        for label in labels:
            r=predictions[label['frame']];supported=r['inside_crop']and r['inside_image']and(mode=='all_latent'or r['visible'])
            error=float(np.linalg.norm(np.asarray(r['pixel'])-label['pixel']))if supported else None
            rows.append({**label,'prediction':r['pixel']if supported else None,'error_px':error,'seed_frame':r['seed_frame'],'seed_distance':r['seed_distance']})
        errors=[r['error_px']for r in rows if r['error_px']is not None]
        reports[mode]={'reviewed_positions':len(rows),'answered':len(errors),'within_12px':sum(e<=12 for e in errors),'within_25px':sum(e<=25 for e in errors),
                       'median_error_px':float(np.median(errors))if errors else None,'details':rows}
    return {'model':data['model'],'seed_count':len(seeds),'eligible_frames':len(predictions),'visible_proposals':sum(r['visible']and r['inside_crop']and r['inside_image']for r in predictions.values()),'scores':reports,
            'limits':'Reused approximate assistant labels, excluding seed frames and their +-3 neighbors. Not an independent game or human benchmark.'}


def context_scores(data,intervals,baseline=None,exclude_frames=()):
    excluded=set(exclude_frames)
    labels=[r for r in reference()if r['frame']not in excluded and any(a<=r['frame']<b for a,b in intervals)and(baseline is None or baseline['frames'][r['frame']]['state']=='unknown')];details=[]
    for label in labels:
        row=data['frames'][label['frame']];hypotheses=row.get('hypotheses',[])
        errors=[float(np.linalg.norm(np.array(h['pixel'])*[1920,1080]-label['pixel']))for h in hypotheses]
        details.append({**label,'answered':bool(hypotheses),'top_center_error_px':errors[0]if errors else None,
                        'top_radius_px':hypotheses[0]['radius_px']if hypotheses else None,
                        'inside_top_region':bool(errors and errors[0]<=hypotheses[0]['radius_px']),
                        'inside_any_region':any(e<=h['radius_px']for e,h in zip(errors,hypotheses))})
    answered=[r for r in details if r['answered']]
    return {'reviewed_positions':len(details),'answered':len(answered),'inside_top_region':sum(r['inside_top_region']for r in details),
            'inside_any_region':sum(r['inside_any_region']for r in details),'median_center_error_px':float(np.median([r['top_center_error_px']for r in answered]))if answered else None,
            'median_radius_px':float(np.median([r['top_radius_px']for r in answered]))if answered else None,'details':details}


def main(folder):
    out={};config=SequenceConfig(**json.loads((Path(__file__).parent/'sequence-config.json').read_text()))
    for path in sorted(list(folder.glob('*-pilot.json'))+list(folder.glob('*-all.json'))):
        data=json.loads(path.read_text())
        if 'tracks'in data:out[path.stem]=point_scores(data)
    ledger_path=folder/'context-ledger.json'
    if ledger_path.exists():
        ledger=json.loads(ledger_path.read_text());source=json.loads((ROOT/'web/data/overhead-20261025/replay.json').read_text());baseline=json.loads((ROOT/'web/data/ball-rebuild-20261026/replay.json').read_text())
        context=build_context_replay(source,ledger,config,range(466,516));combined=merge_visual_and_context(baseline,context)
        verified=folder/'tapnextpp512-verified.json'
        if verified.exists():combined=fuse_point_tracks(source,baseline,context,json.loads(verified.read_text()),PointFusionConfig(**json.loads((Path(__file__).parent/'point-fusion-config.json').read_text())))
        intervals=[r['core']for r in ledger['requests'].values()if r['status']=='completed']
        eligible=[f for a,b in intervals for f in range(a,b)if not 466<=f<516]
        out['context']={'completed_windows':len(intervals),'evaluated_frames':len(eligible),
                        'context_states':dict(collections.Counter(context['frames'][f]['state']for f in eligible)),
                        'combined_states':dict(collections.Counter(combined['frames'][f]['state']for f in eligible)),
                        'baseline_states':dict(collections.Counter(baseline['frames'][f]['state']for f in eligible)),
                        'regions':context_scores(context,intervals),
                        'regions_on_previous_unknown_frames':context_scores(context,intervals,baseline)}
        save(folder/'context-replay.json',context);save(folder/'combined-replay.json',combined)
    save(folder/'evaluation.json',out)
    for name,result in out.items():
        if 'scores'in result:print(name,{k:{a:b for a,b in s.items()if a!='details'}for k,s in result['scores'].items()})
        else:print(name,{k:({a:b for a,b in v.items()if a!='details'}if isinstance(v,dict)else v)for k,v in result.items()})


if __name__=='__main__':main(Path(sys.argv[1]))
