"""Score a model-label distillation experiment without retuning its threshold."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
from evaluate import evaluate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT/'build/ball-temporal-20261026/distillation-report.json'
    report = read(source)
    assert report['protocol'] == read(HERE/'distill-protocol.json')
    assert report['teacher_labels_sha256'] == sha(HERE/'distill-labels.json')
    assert report['transfer_inputs_sha256'] == sha(HERE/'results/distill-transfer-inputs.json')
    labels = sum([read(HERE/f'{scope}-reference.json')['rows']
                  for scope in ['fresh', 'extra', 'final-check']], [])
    # The protocol chooses the model using development scores only.
    selected = max(['image-head', 'temporal-head'], key=lambda name: (
        report['variants'][name]['development']['f1'],
        -report['variants'][name]['development']['fp'], name == 'image-head'))
    metrics = {}
    for name, variant in report['variants'].items():
        frames = [{**r, 'candidates': [{'pixel': r['pixel'], 'score': 1.}]
                   if r['accepted'] else []} for r in variant['transfer']]
        metrics[name] = {f'at_{radius}px': evaluate(
            frames, labels, {'threshold': .5, 'appearance_power': 0}, radius)
            for radius in [12, 25]}
    output = {**report, 'evaluated_at': datetime.now(timezone.utc).isoformat(),
              'source_sha256': sha(source), 'selected_variant': selected,
              'decision': 'Do not promote: the development-selected verifier misses all four positive later-segment teacher labels.',
              'models': metrics}
    destination = HERE/'results/distillation-evaluation.json'
    assert not destination.exists(), 'Do not overwrite a sealed result.'
    destination.write_text(json.dumps(output, separators=(',', ':'))+'\n')
    print(json.dumps({'selected': selected, 'transfer': {
        name: {k: v for k, v in value['at_25px'].items() if k != 'details'}
        for name, value in metrics.items()}}, indent=2))


if __name__ == '__main__': main()
