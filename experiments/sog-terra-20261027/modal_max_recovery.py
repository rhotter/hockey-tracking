"""Larger output allowance only for known token-limited max responses."""
from pathlib import Path
import modal

HERE = Path(__file__).parent
ROOT = HERE.parents[1] if modal.is_local() else Path('/work')
app = modal.App('sport-scans-sog-terra-max-recovery-20261027')
volume = modal.Volume.from_name('sport-scans-experiments')
image = (modal.Image.debian_slim(python_version='3.12').pip_install('pillow==11.3.0')
    .add_local_dir(HERE, '/work/experiment', ignore=['results', '__pycache__'])
    .add_local_file(HERE/'max-recovery-protocol.json', '/work/recovery-experiment/protocol.json')
    .add_local_dir(ROOT/'experiments/sog-20261027', '/work/baseline', ignore=['__pycache__'])
    .add_local_dir(ROOT/'hockey_tracking', '/work/hockey_tracking', ignore=['__pycache__'])
    .add_local_file(ROOT/'scripts/propose_shot_windows.py', '/work/propose_shot_windows.py')
    .add_local_file(ROOT/'scripts/detect_shots.py', '/work/detect_shots.py')
    .add_local_file(ROOT/'experiments/openai-20261024/run_nbhl_openai_pilot.py', '/work/client.py'))


@app.function(image=image, cpu=4, memory=8192, timeout=7200, volumes={'/data': volume}, max_containers=1)
def execute():
    import json
    import sys
    from urllib.request import Request, build_opener
    from urllib.error import HTTPError, URLError
    sys.path[:0] = ['/work/experiment', '/work/baseline', '/work']
    import worker_terra as worker
    from client import NoRedirects
    volume.reload()
    protocol = worker.read(Path('/work/recovery-experiment/protocol.json'))
    assert protocol['parent_max_protocol_sha256'] == worker.sha(Path('/work/experiment/max-protocol.json'))
    previous = worker.read(worker.OUT/'max-context-ledger.json')['requests']
    assert len(previous) == len(protocol['requests'])
    assert all(r['status'] in ('completed', 'failed', 'incomplete', 'budget_not_sent') for r in previous.values())
    requests = []
    for request in protocol['requests']:
        record = previous[request['id']]
        if (record['status'] == 'incomplete' and record.get('usage')
                and (record.get('incomplete_details') or {}).get('reason') == 'max_output_tokens'):
            requests.append({**request, 'id': request['id'].replace('-max-context-', '-max-recovery-'), 'parent_request_id': request['id']})
    worker.write(worker.OUT/'max-recovery-requests.json', {'requests': requests,
        'parent_ledger_sha256': worker.sha(worker.OUT/'max-context-ledger.json')})
    volume.commit()
    def post(endpoint, body, key):
        assert endpoint in ('responses', 'responses/input_tokens')
        request = Request('https://api.openai.com/v1/'+endpoint, data=json.dumps(body).encode(), method='POST',
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer '+key})
        try:
            with build_opener(NoRedirects).open(request, timeout=protocol['response_timeout_seconds'] if endpoint == 'responses' else 360) as response:
                return json.load(response)
        except HTTPError as error:
            raise RuntimeError('OpenAI HTTP status '+str(error.code)) from None
        except (URLError, TimeoutError):
            raise RuntimeError('Connection failed; request is not retried') from None
    worker.contextual.post = post
    worker.HERE = Path('/work/recovery-experiment')
    ledger = worker.run_phase(protocol, 'max-recovery', requests,
        worker.read(Path('/work/baseline/results/scan-ledger.json')), volume)
    return {'requests': len(ledger['requests']), 'estimated_usd': sum(worker.cost(r) for r in ledger['requests'].values())}


@app.local_entrypoint()
def main(env_file: str = ''):
    import sys
    sys.path.insert(0, str(ROOT/'experiments/openai-20261024'))
    from run_nbhl_openai_pilot import read_key
    secret = modal.Secret.from_dict({'OPENAI_API_KEY': read_key(Path(env_file))})
    print(execute.with_options(secrets=[secret]).remote())
