"""Resumable, budgeted OpenAI image comparison. No labels enter requests.

Only --run sends API requests. --audit verifies the sealed evidence locally.
Outputs and crops belong in the ignored outputs/ folder, never in public Git.
"""
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, HTTPRedirectHandler


SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'decision': {'type': 'string', 'enum': ['same', 'different', 'uncertain']},
        'a_consistent': {'type': 'string', 'enum': ['yes', 'no', 'uncertain']},
        'b_consistent': {'type': 'string', 'enum': ['yes', 'no', 'uncertain']},
        'evidence_type': {'type': 'string', 'enum': ['jersey_number', 'distinctive_details',
                                                   'team_difference', 'uniform_only', 'insufficient']},
        'evidence': {'type': 'string'},
    },
    'required': ['decision', 'a_consistent', 'b_consistent', 'evidence_type', 'evidence'],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
        temp = handle.name
    os.replace(temp, path)


def read_key(path):
    key = os.environ.get('OPENAI_API_KEY', '').strip()
    if not key and path and Path(path).is_file():
        for line in Path(path).read_text().splitlines():
            name, sep, value = line.strip().removeprefix('export ').partition('=')
            if sep and name.strip() == 'OPENAI_API_KEY':
                key = value.strip()
                if len(key) >= 2 and key[0] == key[-1] and key[0] in "\"'":
                    key = key[1:-1]
    if not key:
        raise RuntimeError('OpenAI setup is pending; no API requests were made.')
    return key


def image_path(root, item):
    path = (root/item['path']).resolve()
    if not path.is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError('Image path is outside the sealed evidence directory')
    if sha(path) != item['sha256']:
        raise ValueError('Evidence image hash mismatch')
    return path


def audit(root):
    protocol = json.loads((root/'protocol.json').read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    assert manifest['protocol_sha256'] == sha(root/'protocol.json')
    assert manifest['selection_uses_reference_labels'] is False
    assert manifest['phase'] == 'development-pilot'
    rows = manifest['requests']
    assert len({r['id'] for r in rows}) == len(rows)
    assert len(rows) == 2 * sum(c['edges'] for c in manifest['clips'])
    assert {r['task'] for r in rows} == {'multi', 'reverse'}
    groups = {}
    for row in rows:
        assert set(row) == {'id', 'task', 'clip', 'edge_id', 'images', 'prompt'}
        assert 2 <= len(row['images']) <= 6
        for item in row['images']:
            assert set(item) == {'path', 'sha256'}
            image_path(root, item)
        groups.setdefault(row['edge_id'], {})[row['task']] = row
    assert all(set(g) == {'multi', 'reverse'} for g in groups.values())
    return protocol, manifest


def payload(root, row, spec, protocol):
    content = [{'type': 'input_text', 'text': row['prompt']}]
    for item in row['images']:
        encoded = base64.b64encode(image_path(root, item).read_bytes()).decode('ascii')
        content.append({'type': 'input_image', 'detail': protocol['image_detail'],
                        'image_url': 'data:image/jpeg;base64,' + encoded})
    return {'model': spec['id'], 'input': [{'role': 'user', 'content': content}],
            'reasoning': {'effort': protocol['reasoning_effort']},
            'text': {'format': {'type': 'json_schema', 'name': 'player_comparison',
                                'strict': True, 'schema': SCHEMA}},
            'max_output_tokens': protocol['max_output_tokens'], 'store': False,
            'service_tier': 'default', 'truncation': 'disabled'}


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def post(endpoint, body, key):
    assert endpoint in ['responses', 'responses/input_tokens']
    request = Request('https://api.openai.com/v1/' + endpoint,
                      data=json.dumps(body).encode(), method='POST',
                      headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
    try:
        with build_opener(NoRedirects).open(request, timeout=180) as response:
            return json.load(response)
    except HTTPError as error:
        # Do not print provider error bodies, request headers, or payload images.
        raise RuntimeError('OpenAI HTTP status ' + str(error.code)) from None
    except (URLError, TimeoutError):
        raise RuntimeError('OpenAI connection failed; dispatched requests will not be retried') from None


def reserve_cost(tokens, spec, protocol):
    assert type(tokens) is int and 0 < tokens <= protocol['max_input_tokens'] < 272000
    return (tokens * spec['input_usd_per_million'] * 1.25 +
            protocol['max_output_tokens'] * spec['output_usd_per_million']) / 1e6


def usage_cost(usage, spec):
    # Ignore cache discounts and allow the documented cache-write premium.
    return (usage['input_tokens'] * spec['input_usd_per_million'] * 1.25 +
            usage['output_tokens'] * spec['output_usd_per_million']) / 1e6


def parsed_answer(response):
    if response.get('status') != 'completed':
        return None
    blocks = [c for item in response.get('output', []) if item.get('type') == 'message'
              for c in item.get('content', [])]
    if any(c.get('type') == 'refusal' for c in blocks):
        return None
    text = ''.join(c.get('text', '') for c in blocks if c.get('type') == 'output_text')
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict) or set(value) != set(SCHEMA['required']):
        return None
    for name, field in SCHEMA['properties'].items():
        if not isinstance(value[name], str) or ('enum' in field and value[name] not in field['enum']):
            return None
    return value


def schedule(manifest, protocol):
    # Pair adjacent forward/reverse tasks and alternate models within each task.
    rows = sorted(manifest['requests'], key=lambda r: (r['edge_id'], r['task']))
    return [(row, spec) for row in rows for spec in protocol['models']]


def run(root, out, env_file=None, api=post):
    protocol, manifest = audit(root)
    key = read_key(env_file)
    out.mkdir(parents=True, exist_ok=True)
    with (out/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger_path = out/'ledger.json'
        identity = {'protocol_sha256': sha(root/'protocol.json'),
                    'manifest_sha256': sha(root/'manifest.json'), 'runner_sha256': sha(__file__)}
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else identity | {
            'budget_usd': protocol['budget_usd'], 'requests': {}}
        assert all(ledger[k] == v for k, v in identity.items()), 'Protocol/evidence/runner changed'
        assert ledger['budget_usd'] == protocol['budget_usd']
        for row, spec in schedule(manifest, protocol):
            request_id = spec['id'] + '/' + row['id']
            if request_id in ledger['requests']:
                continue  # Includes reservations whose response/billing is unknown.
            body = payload(root, row, spec, protocol)
            count_body = {k: v for k, v in body.items()
                          if k not in ['max_output_tokens', 'store', 'service_tier']}
            count = api('responses/input_tokens', count_body, key)['input_tokens']
            reservation = reserve_cost(count, spec, protocol)
            spent = sum(r['cost_ceiling_usd'] for r in ledger['requests'].values())
            if spent + reservation > protocol['budget_usd']:
                ledger['stop_reason'] = 'budget'; atomic_json(ledger_path, ledger)
                return ledger
            record = {'model': spec['id'], 'id': row['id'], 'task': row['task'],
                      'edge_id': row['edge_id'], 'clip': row['clip'],
                      'status': 'reserved', 'cost_ceiling_usd': reservation,
                      'counted_input_tokens': count, 'parsed': None,
                      'request_sha256': hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()}
            ledger['requests'][request_id] = record
            atomic_json(ledger_path, ledger)
            start = time.monotonic()
            try:
                response = api('responses', body, key)
            except Exception:
                record['status'] = 'unknown_billing'; atomic_json(ledger_path, ledger)
                raise
            usage = response.get('usage')
            record.update(seconds=time.monotonic()-start, status=response.get('status', 'unknown'),
                          response_id=response.get('id'), returned_model=response.get('model'),
                          parsed=parsed_answer(response), usage=usage,
                          incomplete_details=response.get('incomplete_details'))
            if usage:
                actual = usage_cost(usage, spec)
                record['cost_ceiling_usd'] = actual
                if actual > reservation + 1e-8:
                    ledger['stop_reason'] = 'cost_exceeded_reservation'
                    atomic_json(ledger_path, ledger)
                    raise RuntimeError('Provider usage exceeded reservation; stopped')
            else:
                record['status'] = 'missing_usage'
            atomic_json(ledger_path, ledger)
            print(json.dumps({'model': spec['id'], 'completed': len(ledger['requests']),
                              'cost_ceiling_usd': round(sum(r['cost_ceiling_usd'] for r in ledger['requests'].values()), 4)}), flush=True)
        ledger['stop_reason'] = 'all_attempted'; atomic_json(ledger_path, ledger)
        return ledger


def summarize(ledger):
    from collections import Counter
    results = []
    for model in sorted({r['model'] for r in ledger['requests'].values()}):
        rows = [r for r in ledger['requests'].values() if r['model'] == model]
        groups = {}
        for row in rows:
            groups.setdefault(row['edge_id'], {})[row['task']] = row
        complete = [g for g in groups.values() if set(g) == {'multi', 'reverse'}
                    and all(r.get('parsed') is not None for r in g.values())]
        results.append({'model': model, 'attempted': len(rows),
                        'valid': sum(r.get('parsed') is not None for r in rows),
                        'decisions': dict(Counter((r.get('parsed') or {}).get('decision', 'uncertain') for r in rows)),
                        'order_comparisons': len(complete),
                        'order_agreement': sum(g['multi']['parsed']['decision'] == g['reverse']['parsed']['decision'] for g in complete),
                        'cost_ceiling_usd': sum(r['cost_ceiling_usd'] for r in rows),
                        'seconds': sum(r.get('seconds', 0) for r in rows)})
    return {'models': results, 'stop_reason': ledger.get('stop_reason'),
            'tracking_accuracy_measured': False,
            'limitation': 'Development pilot only. Agreement is not accuracy; unknown cases are not negatives.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--env-file', type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--audit', action='store_true')
    mode.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.audit:
        protocol, manifest = audit(args.evidence)
        print(json.dumps({'verified_requests_per_model': len(manifest['requests']),
                          'models': [s['id'] for s in protocol['models']], 'api_requests_made': 0}))
    else:
        if not args.output:
            parser.error('--output is required with --run')
        ledger = run(args.evidence, args.output, args.env_file)
        summary = summarize(ledger)
        atomic_json(args.output/'summary.json', summary)
        print(json.dumps(summary))


if __name__ == '__main__':
    main()
