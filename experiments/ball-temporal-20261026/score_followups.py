"""Score the frozen replay-training and multiview proposals, including failures."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
from evaluate import evaluate

HERE = Path(__file__).resolve().parent
BUILD = HERE.parents[1]/'build/ball-temporal-20261026'


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    labels = sum([read(HERE/f'{scope}-reference.json')['rows'] for scope in ['fresh', 'extra', 'final-check']], [])
    for source, protocol_file in [('distillation-replay-report', 'distill-replay'),
                                  ('verifier-rerank', 'verifier-rerank'),
                                  ('multiview-transfer', 'multiview-transfer')]:
        path = BUILD/f'{source}.json'; report = read(path)
        assert report['protocol'] == read(HERE/f'{protocol_file}-protocol.json')
        models = {}
        for name, variant in (report.get('models') or report['variants']).items():
            if 'transfer' in variant:
                frames = [{**r, 'candidates': [{'pixel': r['pixel'], 'score': 1.}]
                           if r['accepted'] else []} for r in variant['transfer']]
                point = {'threshold': .5, 'appearance_power': 0}
            else:
                frames = variant['frames']; point = report['protocol']['operating_point']
            models[name] = {f'at_{radius}px': evaluate(frames, labels, point, radius) for radius in [12, 25]}
        output = {**report, 'evaluated_at': datetime.now(timezone.utc).isoformat(),
                  'source_sha256': sha(path), 'evaluation': models}
        destination = HERE/'results'/f'{source}-evaluation.json'
        assert not destination.exists(), 'Do not overwrite sealed reports.'
        destination.write_text(json.dumps(output, separators=(',', ':'))+'\n')
        print(json.dumps({'source': source, 'models': {name: {k: v for k, v in row['at_25px'].items() if k != 'details'} for name, row in models.items()}}, indent=2))


if __name__ == '__main__': main()
