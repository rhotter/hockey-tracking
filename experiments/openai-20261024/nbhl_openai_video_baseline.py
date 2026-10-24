"""Direct, sampled-video baseline: full frames and task only, no tracker inputs."""
import base64
import hashlib
import json
import math
import time
from pathlib import Path
from urllib.request import Request, build_opener
from urllib.error import HTTPError, URLError

from run_nbhl_openai_pilot import atomic_json, sha, read_key, NoRedirects, reserve_cost, usage_cost


def obj(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties),
            'additionalProperties': False}


def array(items):
    return {'type': 'array', 'items': items}


S = {'type': 'string'}
N = {'type': 'number'}
NULL_N = {'type': ['number', 'null']}
NULL_S = {'type': ['string', 'null']}
BOX = {'type': 'array', 'items': N, 'minItems': 4, 'maxItems': 4}
POINT = {'type': ['array', 'null'], 'items': N, 'minItems': 2, 'maxItems': 2}
INTERVAL = obj({'start_s': N, 'end_s': N, 'evidence': S})
SCHEMA = obj({
    'summary': S,
    'players': array(obj({
        'player_id': S, 'team': S, 'jersey_number': NULL_S,
        'identity_evidence': S, 'visible_intervals': array(INTERVAL),
        'time_on_floor_s': NULL_N, 'time_on_floor_evidence': S,
        'distance_miles': NULL_N, 'distance_evidence': S,
        'shots': NULL_N, 'shots_on_goal': NULL_N, 'goals': NULL_N,
        'assists': NULL_N, 'stats_evidence': S,
    })),
    'frames': array(obj({
        'frame': {'type': 'integer'},
        'objects': array(obj({'player_id': S, 'bbox': BOX,
                              'visible_floor_point': POINT})),
    })),
    'events': array(obj({
        'type': {'type': 'string', 'enum': ['shot', 'shot_on_goal', 'goal', 'assist',
                                           'substitution', 'faceoff', 'stoppage']},
        'time_s': N, 'game_clock': NULL_S, 'team': NULL_S,
        'player_id': NULL_S, 'related_player_ids': array(S),
        'certainty': {'type': 'string', 'enum': ['clear', 'uncertain']},
        'evidence': S,
    })),
    'possession': array(obj({'team': S, 'start_s': N, 'end_s': N,
                             'certainty': {'type': 'string', 'enum': ['clear', 'uncertain']},
                             'evidence': S})),
    'scoreboard': array(obj({'time_s': N, 'game_clock': NULL_S,
                             'team_a': NULL_S, 'team_b': NULL_S,
                             'score_a': NULL_N, 'score_b': NULL_N,
                             'evidence': S})),
    'limitations': array(S),
})

PROMPT = '''Analyze this NBHL BALL HOCKEY recording end to end, directly from the supplied chronological full frames. Do the whole task yourself: find and track every visible on-floor player, keep player identity stable through the clip where supported, read team and jersey numbers when visible, recover player trajectories, and report time on floor, distance run in miles, shots, shots on goal, goals, assists with their times, and team possession intervals. Read the visible score and game clock, and also assess goals from play itself. No detector boxes, tracks, crops, roster, prior model answers, or annotations are supplied.

The API input is sampled video frames, not a native video or audio stream. Sampling can hide brief events. Use all supplied frames as temporal context. All times in your output are seconds relative to the beginning of this clip, not game-clock or source-video seconds; game_clock is a separate field. Report only this clip's statistics, never extrapolate to the full game.

Give every player a stable arbitrary player_id and use it consistently everywhere. Leave jersey_number null when unreadable; do not infer hidden digits. If identity cannot be resolved, use separate IDs and explain the uncertainty. Include goalkeepers; exclude referees, spectators, and bench players from on-floor tracks. In frames, return one entry for EVERY requested output frame, with every visible player's bbox [left,top,right,bottom] normalized to 0..1000 in that image. The box should enclose the visible person, not their extended stick. Provide visible_floor_point [x,y] only where the actual contact point with the playing surface is visible; otherwise null. Do not invent hidden positions. Empty objects is allowed if no player is visible.

Do not treat being off camera as being off the playing surface. Use null for time_on_floor_s or distance_miles when visibility, rink scale, camera calibration, or identity does not support it. Explain the evidence and what is missing. A null count means not established; zero means you can establish zero. Keep clear events separate from uncertain hypotheses with the certainty field and specific visual evidence. Do not turn a scoreboard update alone into a known scorer or assist. Possession can remain unknown; do not force intervals to cover the whole clip. Return the specified JSON only, with concise evidence strings.'''


def sample_indices(spec):
    count = spec['frame_count']
    outputs = list(range(0, count, spec['output_step_frames']))
    outputs = sorted(set(outputs + [count - 1]))
    inputs = sorted(set(range(0, count, spec['input_step_frames'])) | set(outputs))
    return inputs, outputs


def make_payload(clip, model, protocol, root):
    content = [{'type': 'input_text', 'text': PROMPT + '\n\n' + json.dumps({
        'clip_duration_s': clip['frame_count'] / 15,
        'source_fps': 15, 'input_sampling_step_s': clip['input_step_frames'] / 15,
        'requested_output_frames': clip['output_frames'],
        'coordinate_system': '0..1000 normalized to each supplied image',
    })}]
    for frame in clip['images']:
        path = root/frame['path']
        assert path.resolve().is_relative_to(root.resolve())
        assert sha(path) == frame['sha256']
        content.extend([
            {'type': 'input_text', 'text': f"Frame {frame['frame']}; clip time {frame['frame']/15:.6f}s; image size {frame['width']}x{frame['height']}"},
            {'type': 'input_image', 'detail': 'original',
             'image_url': 'data:image/jpeg;base64,' + base64.b64encode(path.read_bytes()).decode()},
        ])
    return {'model': model['id'], 'input': [{'role': 'user', 'content': content}],
            'reasoning': {'effort': protocol['reasoning_effort']},
            'text': {'format': {'type': 'json_schema', 'name': 'ball_hockey_video_analysis',
                                'strict': True, 'schema': SCHEMA}},
            'max_output_tokens': protocol['max_output_tokens'], 'store': False,
            'truncation': 'disabled', 'service_tier': 'default'}


def post(endpoint, body, key):
    assert endpoint in ['responses', 'responses/input_tokens']
    request = Request('https://api.openai.com/v1/'+endpoint,
                      data=json.dumps(body).encode(), method='POST',
                      headers={'Content-Type': 'application/json', 'Authorization': 'Bearer '+key})
    try:
        with build_opener(NoRedirects).open(request, timeout=900) as response:
            return json.load(response)
    except HTTPError as error:
        raise RuntimeError('OpenAI HTTP status '+str(error.code)) from None
    except (URLError, TimeoutError):
        raise RuntimeError('OpenAI connection failed; request will not be retried') from None


def parse(response):
    if response.get('status') != 'completed':
        return None
    parts = [c for item in response.get('output', []) if item.get('type') == 'message'
             for c in item.get('content', [])]
    if any(c.get('type') == 'refusal' for c in parts):
        return None
    try:
        value = json.loads(''.join(c.get('text', '') for c in parts if c.get('type') == 'output_text'))
        if not isinstance(value, dict) or set(value) != set(SCHEMA['properties']):
            return None
        return value
    except (TypeError, ValueError):
        return None


def validate(value, clip):
    """Validate semantic constraints without changing or repairing predictions."""
    errors = []
    if value is None:
        return ['No completed structured answer']
    players = [r['player_id'] for r in value['players']]
    if len(set(players)) != len(players): errors.append('duplicate player IDs')
    frames = [r['frame'] for r in value['frames']]
    if len(set(frames)) != len(frames): errors.append('duplicate frames')
    if set(frames) != set(clip['output_frames']): errors.append('missing or unexpected output frames')
    for frame in value['frames']:
        ids = [r['player_id'] for r in frame['objects']]
        if len(set(ids)) != len(ids): errors.append('same player twice in one frame')
        if set(ids) - set(players): errors.append('unknown player in frame')
        for row in frame['objects']:
            b = row['bbox']
            if (len(b) != 4 or not all(isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<=1000 for v in b)
                    or b[0]>=b[2] or b[1]>=b[3]): errors.append('invalid box')
    duration = clip['frame_count']/15
    for event in value['events'] + value['scoreboard']:
        if not 0<=event['time_s']<=duration: errors.append('event outside clip')
    for event in value['events']:
        if event['player_id'] is not None and event['player_id'] not in players: errors.append('unknown event player')
        if set(event['related_player_ids'])-set(players): errors.append('unknown related player')
    for interval in value['possession'] + [i for p in value['players'] for i in p['visible_intervals']]:
        if not 0<=interval['start_s']<=interval['end_s']<=duration: errors.append('invalid interval')
    for player in value['players']:
        for key in ['time_on_floor_s','distance_miles','shots','shots_on_goal','goals','assists']:
            v = player[key]
            if v is not None and (not math.isfinite(v) or v<0): errors.append('invalid player metric')
        if player['time_on_floor_s'] is not None and player['time_on_floor_s']>duration: errors.append('floor time exceeds clip')
    return sorted(set(errors))


def run(root, out, protocol, manifest, commit=lambda: None, api=post):
    import fcntl
    key = read_key(None)
    out.mkdir(parents=True, exist_ok=True)
    identity = {'protocol_sha256': sha(root/'protocol.json'), 'manifest_sha256': sha(root/'manifest.json'),
                'runner_sha256': sha(__file__),
                'utility_sha256': sha(Path(__file__).with_name('run_nbhl_openai_pilot.py'))}
    with (out/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = out/'ledger.json'
        ledger = json.loads(path.read_text()) if path.exists() else identity | {'budget_usd': protocol['budget_usd'], 'requests': {}}
        assert all(ledger[k] == v for k,v in identity.items())
        assert ledger['budget_usd'] == protocol['budget_usd']
        for clip in manifest['clips']:
            for model in protocol['models']:
                rid = model['id']+'/'+clip['clip']
                if rid in ledger['requests']: continue
                body = make_payload(clip, model, protocol, root)
                token_body = {k:v for k,v in body.items() if k not in ['store','max_output_tokens','service_tier']}
                count = api('responses/input_tokens', token_body, key)['input_tokens']
                if not 0<count<=protocol['max_input_tokens']:
                    ledger['stop_reason'] = 'input_token_limit'; atomic_json(path,ledger); commit(); return ledger
                reservation = reserve_cost(count, model, protocol)
                if sum(r['cost_ceiling_usd'] for r in ledger['requests'].values())+reservation>protocol['budget_usd']:
                    ledger['stop_reason'] = 'budget'; atomic_json(path,ledger); commit(); return ledger
                record = {'clip': clip['clip'], 'model': model['id'], 'status': 'reserved',
                          'counted_input_tokens': count, 'cost_ceiling_usd': reservation,
                          'request_sha256': hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()}
                ledger['requests'][rid] = record; atomic_json(path,ledger); commit()
                started = time.monotonic()
                try:
                    response = api('responses',body,key)
                except Exception:
                    record['status']='unknown_billing'; atomic_json(path,ledger); commit(); raise
                value = parse(response)
                usage = response.get('usage')
                record.update(status=response.get('status','unknown'), seconds=time.monotonic()-started,
                              response_id=response.get('id'), returned_model=response.get('model'),
                              parsed=value, validation_errors=validate(value,clip), usage=usage,
                              incomplete_details=response.get('incomplete_details'))
                atomic_json(out/(model['id']+'-'+clip['clip']+'-response.json'),response)
                if usage:
                    actual = usage_cost(usage,model)
                    record['cost_ceiling_usd']=actual
                    if actual>reservation+1e-8:
                        ledger['stop_reason']='cost_exceeded_reservation'; atomic_json(path,ledger); commit()
                        raise RuntimeError('Provider usage exceeded reservation')
                atomic_json(path,ledger); commit()
                print(json.dumps({'clip':clip['clip'],'model':model['id'],'status':record['status'],
                                  'valid':value is not None,'validation_errors':record['validation_errors'],
                                  'cost_ceiling_usd':round(sum(r['cost_ceiling_usd'] for r in ledger['requests'].values()),4)}),flush=True)
        ledger['stop_reason']='all_attempted'; atomic_json(path,ledger); commit(); return ledger
