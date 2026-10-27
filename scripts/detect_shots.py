"""Turn sequence-model ledgers into reproducible, reviewable shot events.

Inference is performed by experiments/sog-20261027/modal_run.py. This command
does not call an API or require credentials. Use --experiment for another run
with the same protocol/ledger schema, and --output for its website data folder.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hockey_tracking.shots import combine_events,event_totals

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n')

def goal_sanity_check(events,clips):
    """Post-inference comparison with existing official goals/reviewed times."""
    path=ROOT/'web/data/stats/nbhl-752704.json'
    eligible=[c for c in clips if c['source_id']=='OLpK8enSjJs']
    if not eligible or not path.exists():return None
    reference=read(path);rows=[];used=set()
    for clip in eligible:
        for goal in reference['events']:
            a,b=goal['video_interval_s']
            if not clip['start_s']<=a<clip['start_s']+clip['duration_s']:continue
            color='white'if goal['side']=='home'else'red'
            candidates=[e for e in events if e['clip']==clip['id']and e['outcome']=='goal'and e['automatic_status']=='sog'
                and e['team_color']==color and a-2<=e['recording_seconds']<=b+2 and e['id']not in used]
            match=min(candidates,key=lambda e:abs(e['recording_seconds']-(a+b)/2))if candidates else None
            if match:used.add(match['id'])
            rows.append({'clip':clip['id'],'official_event_id':goal['official_event_id'],
                'existing_reviewed_recording_interval_s':[a,b],'matched_event':match['id']if match else None})
    return {'source':'web/data/stats/nbhl-752704.json','source_sha256':sha(path),
        'meaning':'Sanity check only: official goal events with previously assistant-reviewed video timing, using a2-second tolerance. These are development recordings. This does not measure save accuracy or full SOG precision/recall.',
        'reference_goals':len(rows),'matched':len(used),'rows':rows}

def select_records(requests,verify,context,recovery=None):
    """A successful empty answer supersedes the earlier guess; failures do not."""
    records=[];missing=[];fallback=[]
    for request in requests:
        key=request['id'];candidate=(recovery or {}).get(key.replace('-verify-','-context-recovery-'))
        if candidate and candidate['status']=='completed':records.append(candidate);continue
        candidate=context.get(key.replace('-verify-','-context-'))
        if candidate and candidate['status']=='completed':records.append(candidate);continue
        candidate=verify.get(key)
        if candidate and candidate['status']=='completed':records.append(candidate);fallback.append(key)
        else:missing.append(key)
    return records,missing,fallback

def derive(exp):
    protocol=read(exp/'protocol.json');config=read(exp/'context-protocol.json')
    ledgers={phase:read(exp/f'results/{phase}-ledger.json')for phase in ('scan','verify','context')}
    if (exp/'results/context-recovery-ledger.json').exists():ledgers['context-recovery']=read(exp/'results/context-recovery-ledger.json')
    requests=read(exp/'verify-requests.json')['requests'];records,missing,fallback=select_records(
        requests,ledgers['verify']['requests'],ledgers['context']['requests'],ledgers.get('context-recovery',{}).get('requests'))
    events=[];clips=[];clip_reports=[]
    for clip in protocol['clips']:
        c={**clip};c.pop('frame_folder');clips.append(c)
        found=combine_events([r for r in records if r['clip']==c['id']],c,protocol['fps']);events.extend(found)
        covered=set()
        for r in ledgers['scan']['requests'].values():
            if r['clip']==c['id'] and r['status']=='completed':covered.update(range(*r['core']))
        clip_reports.append({'id':c['id'],'scanned_seconds':len(covered)/protocol['fps'],
            'scan_coverage':len(covered)/c['frames'],'totals':event_totals(found),
            'contextual_sog':sum(e['automatic_status']=='sog'and e['basis']=='contextual'for e in found),
            'unprocessed_proposals':[k for k in missing if k.startswith(c['id']+'-')],
            'pixel_verifier_fallback':[k for k in fallback if k.startswith(c['id']+'-')]})
    # Presentation metadata is separate from the frozen inference protocol.
    presentation=read(exp/'presentation.json')
    for c in clips:c.update(presentation[c['id']])
    for event in events:event['team_label']=presentation[event['clip']]['team_labels'].get(event['team_color'],'Unknown team')
    review_path=exp/'event-review-notes.json'
    if review_path.exists():
        notes=read(review_path)['observations']
        for event in events:
            matching=[n['note']for n in notes if n['clip']==event['clip']and n['seconds'][0]<=event['clip_seconds']<=n['seconds'][1]]
            if matching:event['review_note']=' '.join(matching)
    phase_reports={}
    for phase,ledger in ledgers.items():
        rs=list(ledger['requests'].values());phase_events=[]
        for c in protocol['clips']:phase_events.extend(combine_events([r for r in rs if r['clip']==c['id']],c))
        phase_reports[phase]={'requests':len(rs),'statuses':dict(Counter(r['status']for r in rs)),
            'estimated_usd_including_unresolved_reservations':sum(r.get('estimated_usd',r.get('reserved_usd',0))for r in rs),
            'totals':event_totals(phase_events)}
    report={'scope':f'{sum(c["duration_s"]for c in clips)} seconds across {len(clips)} NBHL ball-hockey recordings; development results, not full-game stats.',
        'method_summary':'A video model scans overlapping 20-second windows at 1.875 frames/s, then examines candidate plays at 5 frames/s. The context pass adds native goal-mouth crops at 7.5 frames/s, full views at 2.5 frames/s and up to 30 seconds of aftermath. Counts require an on-target save or a goal with supporting evidence. Contextual calls are labeled; unresolved attempts remain reviewable.',
        'review_summary':'The assistant reviewed all eight minutes on chronological contact sheets before reading event classifications, then inspected selected plays more closely. These observations are not independent human ground truth. Precision and recall have not been established; 100% scanning coverage does not imply 100% event recall.',
        'definition':protocol['definition'],'clips':clip_reports,'totals':event_totals(events),'phases':phase_reports,
        'campaign_budget_usd':config['campaign_budget_usd'],
        'cost_scope':'Conservative OpenAI API estimates, including unresolved request reservations; excludes Modal compute/storage and is not an invoice.',
        'estimated_usd_including_unresolved_reservations':sum(r['estimated_usd_including_unresolved_reservations']for r in phase_reports.values()),
        'unprocessed_proposals':missing,'pixel_verifier_fallback':fallback,
        'goal_sanity_check':goal_sanity_check(events,clips),
        'count_policy_sha256':sha(ROOT/'hockey_tracking/shots.py'),
        'input_sha256':{str(p.relative_to(exp)):sha(p) for p in [exp/'protocol.json',exp/'context-protocol.json',exp/'verify-requests.json',exp/'presentation.json',exp/'count-policy.json',*([review_path]if review_path.exists()else[]),*([exp/'recovery-protocol.json']if 'context-recovery'in ledgers else[]),*[exp/f'results/{phase}-ledger.json'for phase in ledgers]]},
        'limitations':['No independent human event labels or precision/recall measurement.',
          'Counts are incomplete when attempts are missed in the initial scan or remain ambiguous.',
          'Contextual saves can be confused with wide shots or passes; model confidence is not a calibrated probability.',
          'Release times are approximate at the supplied sampling rate; recording time is not game-clock time.',
          'Jersey readings and player attribution are unverified.',
          'Pittsburgh light-gray jerseys retain the original black compatibility code; display labels correct their appearance.',
          'The verification request manifest overrides the initial protocol defaults with stride3 and 30/45-frame context.']}
    return {'schema_version':1,'source_type':'automatic_sequence_model_estimates','fps':protocol['fps'],'clips':clips,'events':events},report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,default=ROOT/'experiments/sog-20261027')
    parser.add_argument('--output',type=Path,default=ROOT/'web/data/sog-20261027')
    args=parser.parse_args();dataset,report=derive(args.experiment);args.output.mkdir(parents=True,exist_ok=True)
    write(args.output/'events.json',dataset);write(args.output/'report.json',report)
    fields=['id','clip','source_id','clip_seconds','recording_seconds','team_color','team_label','shooter_jersey','attempt','outcome','on_target','automatic_status','basis','confidence','count_reason','explanation']
    with (args.output/'events.csv').open('w',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(dataset['events'])
    print(json.dumps({'totals':report['totals'],'estimated_usd':report['estimated_usd_including_unresolved_reservations']}))

if __name__=='__main__':main()
