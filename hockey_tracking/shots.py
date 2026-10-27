"""Conservative shot-on-goal events from temporal visual evidence.

This layer never infers SOG from ball proximity or a goalkeeper pose alone.
Coordinates are image coordinates; timestamps are recording-relative, not game
clock time. Automatic evidence is not independent human ground truth.
"""
from copy import deepcopy
from collections import Counter
import math

OUTCOMES={'goal','save','blocked','miss','post','unknown','not_shot','disallowed_goal'}


def validate_event(event, frames):
    for key in ('release_frame','outcome_frame'):
        if type(event.get(key)) is not int or not 0<=event[key]<frames:
            raise ValueError('Invalid event frame')
    if event['outcome_frame']<event['release_frame']:raise ValueError('Outcome precedes release')
    if event.get('outcome') not in OUTCOMES:raise ValueError('Unknown shot outcome')
    if event.get('attempt') not in ('clear','possible','not_shot'):raise ValueError('Unknown attempt state')
    if event.get('on_target') not in ('yes','no','unknown'):raise ValueError('Unknown target state')
    for key in ('shooter_xy_1000','net_xy_1000'):
        point=event.get(key)
        if point is not None and (len(point)!=2 or not all(isinstance(x,(float,int)) and math.isfinite(x) and 0<=x<=1000 for x in point)):
            raise ValueError('Invalid image coordinate')
    for flag,value in event['evidence'].items():
        if type(value) is not bool:raise ValueError('Evidence flags must be booleans')
    if event.get('confidence') not in (None,'high','medium','low'):raise ValueError('Invalid confidence tier')
    if event.get('basis') not in (None,'direct','contextual','unresolved'):raise ValueError('Invalid evidence basis')
    if event.get('team_color') not in ('red','white','black','green','orange','unknown'):raise ValueError('Invalid team color')


def classify_event(event):
    """Return count category and a concrete reason; no score is a probability."""
    outcome=event['outcome'];e=event['evidence']
    if event['attempt']=='not_shot' or outcome in ('not_shot','disallowed_goal'):
        return 'not_sog','Not a legal attacking shot.'
    if event.get('confidence') in ('medium','low'):
        return 'needs_review','The temporal assessment is not sufficiently confident.'
    cues=set(event.get('cues',[]))
    if (outcome=='goal' and event['on_target']=='yes' and event.get('confidence')=='high'
            and 'center_restart' in cues
            and (e.get('score_change_visible') or e.get('official_goal_signal_visible'))):
        return 'sog','Goal inferred from score/official evidence and the subsequent center restart.'
    if event['attempt']!='clear':return 'needs_review','The attacking release is not established.'
    if outcome in ('blocked','miss','post'):
        return 'not_sog',{'blocked':'Stopped by a skater before reaching goal.','miss':'The attempt missed the goal.','post':'Post or crossbar without a goal.'}[outcome]
    if event['on_target']!='yes':return 'needs_review','The on-target path is not established.'
    if outcome=='save' and e.get('ball_to_goal_visible') and e.get('goalie_save_visible'):
        return 'sog','On-target attacking attempt and goalkeeper save are supported.'
    if outcome=='goal' and (e.get('goal_crossing_visible') or (e.get('score_change_visible') and e.get('official_goal_signal_visible'))):
        return 'sog','Goal supported by crossing or combined score and official evidence.'
    corroboration=cues&{'goalward_path_fragments','rebound_from_goalkeeper','defensive_zone_restart'}
    if (outcome=='save' and event.get('confidence')=='high' and event.get('basis')=='contextual'
            and {'shooting_motion','goalkeeper_interception'}<=cues and corroboration
            and not event.get('alternative_explanation')):
        return 'sog','Save inferred from the shooting motion, keeper interception and corroborating ball-path, rebound or defensive-zone restart evidence.'
    return 'needs_review','The shot outcome lacks sufficient visible evidence.'


def combine_events(records, clip, fps=15):
    """Merge only cross-window duplicates; preserve distinct same-window rebounds."""
    if fps!=15:raise ValueError('These inference frames use15fps')
    candidates=[]
    for record in records:
        if record.get('status')!='completed':continue
        for index,event in enumerate((record.get('answer')or{}).get('events',[])):
            validate_event(event,clip['frames'])
            if not record['core'][0]<=event['release_frame']<record['core'][1]:raise ValueError('Release outside request core')
            if not set(event['evidence_frames'])<=set(record['sample_frames']):raise ValueError('Evidence frame was not supplied')
            candidates.append({**deepcopy(event),'request_id':record['id'],'request_event_index':index})
    candidates.sort(key=lambda e:(e['release_frame'],e['outcome_frame'],e['request_id']))
    groups=[]
    def duplicate(a,b):
        if a['request_id']==b['request_id']:return False
        if abs(a['release_frame']-b['release_frame'])>5 or abs(a['outcome_frame']-b['outcome_frame'])>8:return False
        if a['team_color']!=b['team_color']:return False
        if a.get('shooter_jersey') and b.get('shooter_jersey') and a['shooter_jersey']!=b['shooter_jersey']:return False
        pa,pb=a.get('shooter_xy_1000'),b.get('shooter_xy_1000')
        return pa is not None and pb is not None and math.dist(pa,pb)<=50
    for event in candidates:
        group=next((g for g in groups if all(duplicate(event,x) for x in g)),None)
        if group is None:groups.append([event])
        else:group.append(event)
    events=[]
    for group in groups:
        categories={classify_event(e)[0] for e in group}
        # Disagreement must not silently choose whichever answer gives a SOG.
        chosen=deepcopy(min(group,key=lambda e:(e['release_frame'],e['request_id'])))
        status,reason=classify_event(chosen)
        if len(categories)>1 or len({e['outcome'] for e in group})>1:
            status,reason='needs_review','Overlapping sequence assessments disagree.'
        chosen.update(id=f'{clip["id"]}-{chosen["release_frame"]:04d}',clip=clip['id'],
            source_id=clip['source_id'],clip_seconds=chosen['release_frame']/fps,
            recording_seconds=clip['start_s']+chosen['release_frame']/fps,
            automatic_status=status,count_reason=reason,supporting_requests=sorted({e['request_id'] for e in group}))
        chosen.setdefault('basis','direct' if status=='sog' else 'unresolved')
        events.append(chosen)
    # Two releases can share a sampled frame: preserve them with unique IDs.
    ids=Counter(e['id'] for e in events)
    for index,e in enumerate(events):
        if ids[e['id']]>1:e['id']+=f'-{index}'
    return events


def event_totals(events):
    result={'sog':0,'needs_review':0,'not_sog':0,'teams':{}}
    for event in events:
        status=event['automatic_status'];result[status]+=1
        team=result['teams'].setdefault(event['team_color'],{'sog':0,'goals':0,'saves':0,'needs_review':0})
        if status=='sog':
            team['sog']+=1;team['goals']+=event['outcome']=='goal';team['saves']+=event['outcome']=='save'
        elif status=='needs_review':team['needs_review']+=1
    return result
