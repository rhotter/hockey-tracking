"""Run the approved, sealed OpenAI pilot on Modal with a shared budget ledger."""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/data/ball-hockey/reliability-20261018/openai-reconnect-20261024-v1')
app = modal.App('sport-scans-openai-reconnect-pilot')
volume = modal.Volume.from_name('sport-scans-experiments')
image = (modal.Image.debian_slim(python_version='3.12')
         .env({'PYTHONPATH': '/work'})
         .add_local_file(ROOT/'scripts/run_nbhl_openai_pilot.py', '/work/run_nbhl_openai_pilot.py'))


@app.function(image=image, cpu=1, memory=2048, timeout=14400, max_containers=1,
              volumes={'/data': volume})
def run():
    import json
    import tarfile
    import datetime
    from run_nbhl_openai_pilot import run as pilot, post, summarize, atomic_json, sha
    volume.reload()
    root = Path('/tmp/openai-pilot-evidence'); root.mkdir(exist_ok=True)
    export = json.loads((BASE/'evidence-export.json').read_text())
    archive = BASE/'pilot-evidence.tar.gz'
    assert sha(archive) == export['sha256']
    with tarfile.open(archive) as bundle:
        bundle.extractall(root, filter='data')
    output = BASE/'pilot-run'; output.mkdir(exist_ok=True)
    def request(endpoint, body, key):
        # Persist the previous result and current reservation before each call.
        volume.commit()
        return post(endpoint, body, key)
    try:
        ledger = pilot(root, output, api=request)
        result = summarize(ledger)
        result['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic_json(output/'summary.json', result)
        return result
    except Exception as error:
        atomic_json(output/'error.json', {'type': type(error).__name__, 'message': str(error),
                                         'at': datetime.datetime.now(datetime.timezone.utc).isoformat()})
        raise
    finally:
        volume.commit()


@app.local_entrypoint()
def main():
    import json
    import sys
    sys.path.insert(0, str(ROOT/'scripts'))
    from run_nbhl_openai_pilot import read_key
    secret = modal.Secret.from_dict({'OPENAI_API_KEY': read_key(ROOT/'.env.local')})
    result = run.with_options(secrets=[secret]).remote()
    target = ROOT/'notes/ball-hockey/openai-20261024/pilot-summary.json'
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
