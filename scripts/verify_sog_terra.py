"""Verify the Terra experiment's provenance, costs, alignment and public data."""
import csv
import json
import math
from pathlib import Path
from analyze_sog_terra import BASE, DEST, EXP, ROOT, derive, read, sha, max_view_report
from propose_shot_windows import propose


def main():
    protocol = read(EXP/'protocol.json')
    assert protocol['count_policy_sha256'] == sha(ROOT/'hockey_tracking/shots.py')
    for name, digest in protocol['baseline_sha256'].items():
        assert sha(BASE/name) == digest, name
    baseline_scan = read(BASE/'results/scan-ledger.json')['requests']
    baseline_context = read(BASE/'results/context-ledger.json')['requests']
    phases = {}
    names = ['scan', 'matched-context', 'own-context']
    if (EXP/'results/max-context-ledger.json').exists():
        names.append('max-context')
    if (EXP/'results/max-recovery-ledger.json').exists():
        names.append('max-recovery')
    for phase in names:
        phase_protocol = EXP/('max-recovery-protocol.json' if phase == 'max-recovery' else 'max-protocol.json' if phase == 'max-context' else 'protocol.json')
        ledger = read(EXP/f'results/{phase}-ledger.json')
        phases[phase] = ledger
        assert ledger['protocol_sha256'] == sha(phase_protocol)
        assert ledger['worker_sha256'] == sha(EXP/'worker_terra.py')
        assert ledger['model'] == protocol['model']
        for key, record in ledger['requests'].items():
            assert key == record['id']
            assert record['status'] in ('completed', 'incomplete', 'failed', 'budget_not_sent')
            sample = record['sample_frames']
            assert sample == sorted(set(sample))
            assert set(record['source_sha256']) == set(map(str, sample))
            assert len(record['image_sha256']) == len(record['full_frames'])+len(record['crop_frames'])
            usage = record.get('usage')
            if usage:
                expected = (usage['input_tokens']*protocol['input_usd_per_million_conservative']+usage['output_tokens']*protocol['output_usd_per_million'])/1e6
                assert math.isclose(record['estimated_usd'], expected, abs_tol=1e-10)
                cached = usage.get('input_tokens_details', {}).get('cached_tokens', 0)
                standard = ((usage['input_tokens']-cached)*protocol['input_usd_per_million']+cached*protocol['cached_input_usd_per_million']+usage['output_tokens']*protocol['output_usd_per_million'])/1e6
                assert math.isclose(record['standard_rate_usd'], standard, abs_tol=1e-10)
            if record['status'] == 'completed':
                assert record['response_status'] == 'completed'
                assert record['response_model'].startswith(protocol['model'])
                for event in record['answer']['events']:
                    assert event['release_frame'] in sample and event['outcome_frame'] in sample
            if phase == 'scan':
                original = baseline_scan[key.replace('-scan-', '-')]
                for field in ('core', 'range', 'stride', 'width', 'sample_frames', 'source_sha256'):
                    assert record[field] == original[field], (key, field)
            if phase in ('matched-context', 'max-context', 'max-recovery'):
                original = baseline_context[key.replace('-'+phase+'-', '-context-')]
                for field in ('core', 'sample_frames', 'full_frames', 'crop_frames', 'crop_bounds', 'source_sha256'):
                    assert record[field] == original[field], (key, field)
            if phase in ('max-context', 'max-recovery'):
                high = phases['matched-context']['requests'][key.replace('-'+phase+'-', '-matched-context-')]
                for field in ('image_sha256', 'prompt_sha256'):
                    assert record[field] == high[field], (key, field)
    assert {key.replace('-scan-', '-') for key in phases['scan']['requests']} == set(baseline_scan)
    assert {k.replace('-matched-context-', '-context-') for k in phases['matched-context']['requests']} == set(baseline_context)
    expected = propose(protocol, phases['scan'], sha(EXP/'results/scan-ledger.json'))
    for r in expected['requests']:
        r['id'] = r['id'].replace('-verify-', '-own-context-')
    assert expected == read(EXP/'results/own-requests.json')
    assert {r['id'] for r in expected['requests']} == set(phases['own-context']['requests'])
    if 'max-context' in phases:
        maximum = read(EXP/'max-protocol.json')
        assert maximum['parent_protocol_sha256'] == sha(EXP/'protocol.json')
        assert maximum['selection_events_sha256'] == sha(ROOT/'web/data/sog-20261027/events.json')
        assert {r['id'] for r in maximum['requests']} == set(phases['max-context']['requests'])
        assert maximum['budget_usd'] == protocol['budget_usd']
    if 'max-recovery' in phases:
        recovery = read(EXP/'max-recovery-protocol.json')
        assert recovery['parent_max_protocol_sha256'] == sha(EXP/'max-protocol.json')
        assert recovery['budget_usd'] == protocol['budget_usd']
        expected_requests = []
        for request in recovery['requests']:
            original = phases['max-context']['requests'][request['id']]
            if original['status'] == 'incomplete' and original.get('usage') and (original.get('incomplete_details') or {}).get('reason') == 'max_output_tokens':
                expected_requests.append({**request, 'id': request['id'].replace('-max-context-', '-max-recovery-'), 'parent_request_id': request['id']})
        assert {r['id'] for r in expected_requests} == set(phases['max-recovery']['requests'])
        assert read(EXP/'results/max-recovery-requests.json') == {'requests': expected_requests,
            'parent_ledger_sha256': sha(EXP/'results/max-context-ledger.json')}
    outputs, report = derive()
    assert read(DEST/'events.json') == outputs['own-context']
    assert read(DEST/'matched-events.json') == outputs['matched-context']
    if 'max-context' in outputs:
        assert read(DEST/'max-events.json') == outputs['max-context']
        assert read(DEST/'max-report.json') == max_view_report(outputs, report)
        with (DEST/'max-events.csv').open(newline='') as handle:
            assert [r['id'] for r in csv.DictReader(handle)] == [e['id'] for e in outputs['max-context']['events']]
    assert read(DEST/'report.json') == report
    assert report['estimated_usd_including_unresolved_reservations'] <= protocol['budget_usd']
    with (DEST/'events.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(outputs['own-context']['events'])
    assert [r['id'] for r in rows] == [e['id'] for e in outputs['own-context']['events']]
    for comparison in report['comparison'].values():
        assert len({p['reference_id'] for p in comparison['pairs']}) == comparison['matched']
        assert len({p['candidate_id'] for p in comparison['pairs']}) == comparison['matched']
    print(json.dumps({'requests': {p: len(v['requests']) for p, v in phases.items()},
        'estimated_usd': report['estimated_usd_including_unresolved_reservations'],
        'pipeline_usd_per_minute': report['pipeline_usd_per_video_minute'], 'frozen_baseline_verified': True}))


if __name__ == '__main__':
    main()
