"""Compare matched OpenAI/Qwen decisions without treating agreement as accuracy."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def answer(row):
    value = row.get('parsed') if row else None
    if not isinstance(value, dict) or value.get('decision') not in ['same', 'different', 'uncertain']:
        return None
    return value


def strict_same(row):
    value = answer(row)
    return bool(value and value['decision'] == 'same' and value.get('a_consistent') == 'yes'
                and value.get('b_consistent') == 'yes'
                and value.get('evidence_type') in ['jersey_number', 'distinctive_details'])


def summarize_model(model, rows, expected):
    by_id = {row['id']: row for row in rows}
    assert len(by_id) == len(rows), 'Duplicate response IDs'
    assert set(by_id) <= expected
    edges = sorted({key.rsplit('/', 1)[0] for key in expected})
    forward = [by_id[edge+'/multi'] for edge in edges if edge+'/multi' in by_id]
    valid_forward = [r for r in forward if answer(r)]
    pairs = [(by_id.get(edge+'/multi'), by_id.get(edge+'/reverse')) for edge in edges]
    valid_pairs = [(a, b) for a, b in pairs if answer(a) and answer(b)]
    table = Counter(answer(a)['decision']+' / '+answer(b)['decision'] for a, b in valid_pairs)
    agreement = sum(answer(a)['decision'] == answer(b)['decision'] for a, b in valid_pairs)
    flips = sum({answer(a)['decision'], answer(b)['decision']} == {'same', 'different'} for a, b in valid_pairs)
    stable_decisive = sum(answer(a)['decision'] == answer(b)['decision'] != 'uncertain' for a, b in valid_pairs)
    result = {'model': model, 'expected_requests': len(expected), 'expected_edges': len(edges),
              'attempted_requests': len(rows), 'valid_responses': sum(answer(r) is not None for r in rows),
              'missing_requests': len(expected)-len(rows),
              'invalid_or_unfinished_requests': sum(answer(r) is None for r in rows),
              'forward_decisions': dict(Counter(answer(r)['decision'] for r in valid_forward)),
              'forward_missing_or_invalid': len(edges)-len(valid_forward),
              'valid_order_comparisons': len(valid_pairs), 'order_agreement': agreement,
              'order_disagreement': len(valid_pairs)-agreement, 'direct_same_different_flips': flips,
              'stable_decisive_edges': stable_decisive,
              'symmetric_strict_same_edges': sum(strict_same(a) and strict_same(b) for a, b in valid_pairs),
              'order_table': dict(table), 'identity_accuracy': None}
    if any('cost_ceiling_usd' in r for r in rows):
        result['conservative_list_cost_usd'] = sum(r['cost_ceiling_usd'] for r in rows)
        result['total_inference_seconds'] = sum(r.get('seconds', 0) for r in rows)
        result['input_tokens'] = sum((r.get('usage') or {}).get('input_tokens', 0) for r in rows)
        result['output_tokens'] = sum((r.get('usage') or {}).get('output_tokens', 0) for r in rows)
        result['reasoning_tokens'] = sum(((r.get('usage') or {}).get('output_tokens_details') or {}).get('reasoning_tokens', 0) for r in rows)
        result['returned_models'] = sorted({r['returned_model'] for r in rows if r.get('returned_model')})
    return result


def analyze(evidence, ledger_path, baseline_path):
    manifest_path = evidence/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    ledger = json.loads(ledger_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    protocol = json.loads((evidence/'protocol.json').read_text())
    assert ledger['protocol_sha256'] == baseline['protocol_sha256'] == manifest['protocol_sha256'] == sha(evidence/'protocol.json')
    assert ledger['manifest_sha256'] == sha(manifest_path)
    assert baseline['evidence_sha256'] == manifest['evidence_sha256']
    expected = {r['id'] for r in manifest['requests']}
    assert manifest['selection_uses_reference_labels'] is False
    assert baseline['selection_uses_reference_labels'] is False
    grouped = {spec['id']: [] for spec in protocol['models']}
    for request_id, row in ledger['requests'].items():
        assert request_id == row['model']+'/'+row['id']
        grouped[row['model']].append(row)
    for item in baseline['models']:
        grouped['qwen-'+item['model']] = item['responses']
    summaries = [summarize_model(model, rows, expected) for model, rows in grouped.items()]
    index = {model: {r['id']: r for r in rows} for model, rows in grouped.items()}
    cases = [{'edge_id': edge, 'models': {model: {task: index[model].get(edge+'/'+task)
              for task in ['multi', 'reverse']} for model in grouped}}
             for edge in sorted({r['edge_id'] for r in manifest['requests']})]
    both = [s for s in summaries if s['model'].startswith('gpt-')]
    all_attempted = all(s['missing_requests'] == 0 for s in both)
    all_valid = all(s['missing_requests'] == 0 and s['invalid_or_unfinished_requests'] == 0 for s in both)
    result = {'scope': 'NBHL ball hockey only', 'phase': 'development-pilot',
              'models': summaries, 'all_requested_responses_attempted': all_attempted,
              'all_requested_responses_valid': all_valid, 'stop_reason': ledger.get('stop_reason'),
              'tracking_accuracy_measured': False, 'cases': cases,
              'provenance': {'ledger_sha256': sha(ledger_path), 'manifest_sha256': sha(manifest_path),
                             'protocol_sha256': sha(evidence/'protocol.json'),
                             'qwen_baselines_sha256': sha(baseline_path),
                             'evidence_sha256': manifest['evidence_sha256']},
              'limits': ['Image-order consistency is not identity accuracy. Always show abstentions and stable decisive coverage beside agreement.',
                         '65 hash-selected candidate links across nine reused development clips; no held-out tracking outcome is measured.',
                         'Models see identical original crops and prompts but use different image processing and output/reasoning budgets.',
                         'Cost is a conservative standard-price token estimate, including a cache-write premium; not the provider invoice.',
                         'Unknown or interrupted requests are retained, never silently retried or treated as labeled negatives.']}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--baselines', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.evidence, args.ledger, args.baselines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['cases','limits']}, indent=2))


if __name__ == '__main__':
    main()
