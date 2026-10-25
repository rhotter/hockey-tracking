"""Audit pair-conditioned VLM clothing-number readings without inventing hidden digits."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def normalize(value):
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 999:
        return value
    return 'invalid'


def counts(rows):
    result = dict(readable=0, correct=0, wrong=0, abstained=0, invalid=0,
                  unknown=0, unsupported_readings=0, unknown_abstained=0)
    errors = []
    for row in rows:
        answer = row['prediction']
        if row['status'] == 'readable':
            result['readable'] += 1
            key = ('correct' if answer == row['number'] else
                   'abstained' if answer is None else 'invalid' if answer == 'invalid' else 'wrong')
            result[key] += 1
            if key == 'wrong':
                errors.append(row)
        else:
            result['unknown'] += 1
            if answer is None:
                result['unknown_abstained'] += 1
            elif answer != 'invalid':
                result['unsupported_readings'] += 1
    answered = result['correct'] + result['wrong']
    result['accuracy_when_answered_on_readable_crops'] = result['correct']/answered if answered else None
    result['recall_on_readable_crops'] = result['correct']/result['readable'] if result['readable'] else None
    return result | {'errors': errors}


def evaluate(predictions, labels, benchmark):
    pairs = {r['id']: r for r in benchmark['pairs']}
    labels_by_id = {r['id']: r for r in labels['observations']}
    occurrences = []; votes = defaultdict(list)
    for row in predictions['results']:
        pair = pairs[row['id']]
        parsed = row.get('parsed')
        for side in ('a', 'b'):
            oid = pair[side]
            answer = normalize(parsed.get('number_'+side)) if isinstance(parsed, dict) else 'invalid'
            votes[oid].append(answer)
            occurrences.append(labels_by_id[oid] | {'pair': row['id'], 'prediction': answer})
    consensus = []
    for oid, label in labels_by_id.items():
        answers = votes[oid]
        numeric = Counter(a for a in answers if isinstance(a, int))
        value, n = numeric.most_common(1)[0] if numeric else (None, 0)
        # Treat null/invalid as a vote against a reading. Repeated contexts are
        # correlated evidence, not repeated independent model samples.
        accepted = n >= 2 and n/len(answers) >= .8
        consensus.append(label | {'prediction': value if accepted else None,
                                 'votes': answers, 'agreeing': n})
    splits = {}
    for split in ['all'] + sorted({r['split'] for r in labels['observations']}):
        splits[split] = {name: counts([r for r in rows if split == 'all' or r['split'] == split])
                         for name, rows in [('pair_occurrences', occurrences), ('conservative_consensus', consensus)]}
    return {'model': predictions['model'], 'variant': predictions.get('variant', 'v1'),
            'quantization': predictions.get('quantization', 'bf16'), 'splits': splits,
            'consensus_rows': consensus, 'independent_human_ground_truth': False,
            'readable_crops': labels['readable'], 'unknown_crops': labels['unknown'],
            'limitations': '97 assistant-reviewed crops. Ambiguous draft readings were withdrawn after an enlarged second pass; the audit is not blinded. Numbers on shorts are not independently verified as jersey numbers. Pair-conditioned readings may use the other crop. Repeated contexts and same-player crops are dependent. Unsupported readings on unknown crops are not scored as false. This is an audit after inference, not an untouched test.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=ROOT/'data/ball-hockey/jersey-number-review-v1.json')
    args = parser.parse_args()
    bp = ROOT/'data/ball-hockey/identity-pairs.json'
    lp = args.labels
    benchmark = json.loads(bp.read_text()); labels = json.loads(lp.read_text())
    assert labels['benchmark_sha256'] == hashlib.sha256(bp.read_bytes()).hexdigest()
    predictions = json.loads(args.predictions.read_text())
    assert predictions['benchmark_sha256'] in (hashlib.sha256(bp.read_bytes()).hexdigest(),
                                               hashlib.sha256(json.dumps(benchmark, indent=2).encode()).hexdigest())
    result = evaluate(predictions, labels, benchmark)
    result['labels_sha256'] = hashlib.sha256(lp.read_bytes()).hexdigest()
    result['label_revision'] = labels.get('schema_version', 1)
    result['label_corrections'] = labels.get('corrections', [])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({s: {m: {k: v for k, v in row.items() if k != 'errors'} for m, row in methods.items()}
                      for s, methods in result['splits'].items()}, indent=2))
