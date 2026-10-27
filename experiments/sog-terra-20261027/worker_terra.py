"""Frozen model-substitution and independent-discovery SOG experiment."""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import time

import worker as baseline
import context_worker as contextual
from hockey_tracking.shots import validate_event
from propose_shot_windows import propose

HERE = Path('/work/experiment')
OUT = baseline.BASE / 'sog-terra-20261027-v1'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def cost(record):
    return record.get('estimated_usd', record.get('reserved_usd', 0))


def inputs(protocol, request, phase, discovery):
    """Preserve the original prompts, image encoding, crops and frame sampling."""
    from PIL import Image, ImageDraw, ImageFont
    clip = next(c for c in protocol['clips'] if c['id'] == request['clip'])
    a, b = request['core']
    config = protocol['context']
    font = ImageFont.load_default(size=18)
    if phase == 'scan':
        sample = list(range(*request['range'], request['stride']))
        full, crop_frames, nets = set(sample), set(), []
        prompt = baseline.PROMPT + '\n' + protocol['definition'] + f'\nCORE [{a}, {b}) at15fps; colors {clip["team_colors"]}. Frame times are clip-relative. Broad discovery pass: retain plausible uncertain attempts for later verification.'
    else:
        full = set(range(max(0, a-30), min(clip['frames'], b+30), config['full_frame_stride']))
        full.update(range(min(clip['frames']-1, b+30), min(clip['frames'], b+config['after_context_seconds']*15), config['after_context_stride']))
        proposals = [e for r in discovery['requests'].values() if r['clip'] == clip['id'] and r['status'] == 'completed' for e in r['answer']['events'] if a <= e['release_frame'] < b]
        nets = [e['net_xy_1000'] for e in proposals if e['net_xy_1000']]
        crop_frames = set(range(max(0, a-6), min(clip['frames'], b+12), config['goal_crop_stride'])) if nets else set()
        sample = sorted(full | crop_frames)
        colors = 'green and light gray (use black for the light-gray team for compatibility).' if clip['id'].startswith('pittsburgh') else str(clip['team_colors'])
        prompt = contextual.PROMPT + f'\nCORE[{a},{b}) at15fps. Clip{clip["id"]}. Team appearance: ' + colors
    content = [{'type': 'input_text', 'text': prompt}]
    hashes, image_hashes, bounds = {}, [], None
    for frame in sample:
        source = baseline.BASE / clip['frame_folder'] / f'{frame+1:06d}.jpg'
        raw = Image.open(source).convert('RGB')
        hashes[str(frame)] = sha(source)
        images = []
        if phase == 'scan':
            width = request['width']
            images.append((raw.resize((width, round(width*raw.height/raw.width))), f'Frame{frame} | {frame/15:.3f}s'))
        else:
            if frame in full:
                width = config['full_frame_width']
                images.append((raw.resize((width, round(raw.height*width/raw.width))), f'FULL frame{frame} | {frame/15:.3f}s' + (' AFTERMATH' if frame >= b+30 else '')))
            if frame in crop_frames:
                x = sum(n[0] for n in nets)/len(nets)*raw.width/1000
                y = sum(n[1] for n in nets)/len(nets)*raw.height/1000
                w = config['goal_crop_width']; h = round(w*9/16)
                left = max(0, min(raw.width-w, round(x-w/2)))
                top = max(0, min(raw.height-h, round(y-h/2)))
                bounds = [left, top, left+w, top+h]
                images.append((raw.crop(tuple(bounds)), f'CROP frame{frame} | bounds {bounds}'))
        for im, title in images:
            sheet = Image.new('RGB', (im.width, im.height+26), 'white')
            sheet.paste(im, (0, 26))
            ImageDraw.Draw(sheet).text((4, 3), title, font=font, fill='black')
            buf = io.BytesIO(); sheet.save(buf, format='JPEG', quality=93 if phase == 'scan' else 95)
            image_hashes.append(hashlib.sha256(buf.getvalue()).hexdigest())
            content.append({'type': 'input_image', 'detail': 'original', 'image_url': 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()})
    metadata = {'sample_frames': sample, 'full_frames': sorted(full), 'crop_frames': sorted(crop_frames),
                'crop_bounds': bounds, 'source_sha256': hashes, 'image_sha256': image_hashes,
                'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
    return content, metadata


def run_phase(protocol, phase, requests, discovery, volume):
    path = OUT / f'{phase}-ledger.json'
    ledger = read(path) if path.exists() else {'phase': phase, 'model': protocol['model'],
        'protocol_sha256': sha(HERE/'protocol.json'), 'worker_sha256': sha(Path(__file__)), 'requests': {}}
    assert ledger['protocol_sha256'] == sha(HERE/'protocol.json')
    assert ledger['worker_sha256'] == sha(Path(__file__))
    other = sum(cost(r) for p in OUT.glob('*-ledger.json') if p != path for r in read(p)['requests'].values())
    key = os.environ['OPENAI_API_KEY']; lock = threading.Lock()
    def persist():
        write(path, ledger); volume.commit()
    def task(request):
        rid = request['id']
        record = {**request, 'status': 'preparing', 'reserved_usd': 0}
        with lock:
            if rid in ledger['requests']:
                return
            ledger['requests'][rid] = record
        started = time.monotonic()
        try:
            content, metadata = inputs(protocol, request, phase, discovery)
            record.update(metadata)
            config = protocol['scan'] if phase == 'scan' else protocol['context']
            schema = baseline.SCHEMA if phase == 'scan' else contextual.SCHEMA
            body = {'model': protocol['model'], 'input': [{'role': 'user', 'content': content}],
                'reasoning': {'effort': config['reasoning']}, 'max_output_tokens': config['max_output_tokens'],
                'text': {'format': {'type': 'json_schema', 'name': 'nbhl_shot_events' if phase == 'scan' else 'contextual_nbhl_shots', 'strict': True, 'schema': schema}},
                'store': False, 'service_tier': 'default'}
            count = contextual.post('responses/input_tokens', {'model': body['model'], 'input': body['input']}, key)['input_tokens']
            reserved = (count*protocol['input_usd_per_million_conservative'] + config['max_output_tokens']*protocol['output_usd_per_million'])/1e6
            with lock:
                record['input_tokens_counted'] = count
                if other + sum(cost(r) for r in ledger['requests'].values()) + reserved > protocol['budget_usd']:
                    record['status'] = 'budget_not_sent'; persist(); return
                record.update(status='reserved', reserved_usd=reserved); persist()
            api_start = time.monotonic()
            response = contextual.post('responses', body, key)
            record.update(api_seconds=time.monotonic()-api_start, response_id=response.get('id'),
                          response_status=response.get('status'), response_model=response.get('model'),
                          incomplete_details=response.get('incomplete_details'), usage=response.get('usage'))
            usage = record['usage']
            if usage:
                record['estimated_usd'] = (usage['input_tokens']*protocol['input_usd_per_million_conservative'] + usage['output_tokens']*protocol['output_usd_per_million'])/1e6
                cached = usage.get('input_tokens_details', {}).get('cached_tokens', 0)
                record['standard_rate_usd'] = ((usage['input_tokens']-cached)*protocol['input_usd_per_million'] + cached*protocol['cached_input_usd_per_million'] + usage['output_tokens']*protocol['output_usd_per_million'])/1e6
            output = ''.join(b.get('text', '') for item in response.get('output', []) if item.get('type') == 'message' for b in item.get('content', []) if b.get('type') == 'output_text')
            record['answer'] = json.loads(output) if response.get('status') == 'completed' else None
            if record['answer'] is not None:
                clip = next(c for c in protocol['clips'] if c['id'] == request['clip'])
                for event in record['answer']['events']:
                    validate_event(event, clip['frames'])
                    assert request['core'][0] <= event['release_frame'] < request['core'][1]
                    assert event['release_frame'] in record['sample_frames'] and event['outcome_frame'] in record['sample_frames']
                    assert set(event['evidence_frames']) <= set(record['sample_frames'])
                record['status'] = 'completed'
            else:
                record['status'] = 'incomplete'
        except Exception as error:
            record.update(status='failed', error_type=type(error).__name__)
            # Only the fixed, credential-free transport status is safe to retain.
            if isinstance(error, RuntimeError) and str(error).startswith(('OpenAI HTTP status ', 'Connection failed;')):
                record['error'] = str(error)
        finally:
            with lock:
                record['seconds'] = time.monotonic()-started; persist()
            print({'phase': phase, 'id': rid, 'status': record['status'], 'cost': round(cost(record), 4)}, flush=True)
    with ThreadPoolExecutor(max_workers=protocol['max_concurrency']) as pool:
        list(pool.map(task, requests))
    return ledger


def run(volume):
    OUT.mkdir(exist_ok=True, parents=True)
    protocol = read(HERE/'protocol.json')
    for name, digest in protocol['baseline_sha256'].items():
        assert sha(Path('/work/baseline')/name) == digest, name
    requests = []
    config = protocol['scan']
    for clip in protocol['clips']:
        for start in range(0, clip['frames'], config['core_frames']):
            end = min(clip['frames'], start+config['core_frames'])
            requests.append({'id': f'{clip["id"]}-scan-{start:04d}', 'clip': clip['id'], 'core': [start, end],
                'range': [max(0, start-config['context_frames']), min(clip['frames'], end+config['context_frames'])],
                'stride': config['stride_frames'], 'width': config['width']})
    scan = run_phase(protocol, 'scan', requests, None, volume)
    own = propose(protocol, scan, sha(OUT/'scan-ledger.json'))
    for r in own['requests']:
        r['id'] = r['id'].replace('-verify-', '-own-context-')
    write(OUT/'own-requests.json', own); volume.commit()
    original_scan = read(Path('/work/baseline/results/scan-ledger.json'))
    matched = read(Path('/work/baseline/verify-requests.json'))['requests']
    for r in matched:
        r['id'] = r['id'].replace('-verify-', '-matched-context-')
    run_phase(protocol, 'matched-context', matched, original_scan, volume)
    run_phase(protocol, 'own-context', own['requests'], scan, volume)
    import tarfile
    with tarfile.open(OUT/'results.tar.gz', 'w:gz') as tar:
        for path in sorted(OUT.glob('*.json')):
            tar.add(path, arcname=path.name)
    volume.commit()
    return {'output': str(OUT), 'estimated_usd': sum(cost(r) for p in OUT.glob('*-ledger.json') for r in read(p)['requests'].values())}
