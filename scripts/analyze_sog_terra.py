"""Reproduce Terra SOG outputs, API costs and comparisons without inference."""
import csv
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hockey_tracking.shots import combine_events, event_totals
from detect_shots import read, sha, write, goal_sanity_check

EXP = ROOT/'experiments/sog-terra-20261027'
BASE = ROOT/'experiments/sog-20261027'
DEST = ROOT/'web/data/sog-terra-20261027'


def match_events(reference, candidate, tolerance=2):
    """Maximum one-to-one temporal matching within the same clip and team.

    Prefer more matches, then lower total timing error. Matching ignores count
    status and outcome, so disagreements are not discarded during alignment.
    Neither side is ground truth.
    """
    pairs = []
    groups = {(e['clip'], e['team_color']) for e in reference+candidate}
    for clip, team in sorted(groups):
        left = sorted([e for e in reference if (e['clip'], e['team_color']) == (clip, team)], key=lambda e: (e['clip_seconds'], e['id']))
        right = sorted([e for e in candidate if (e['clip'], e['team_color']) == (clip, team)], key=lambda e: (e['clip_seconds'], e['id']))
        @lru_cache(None)
        def solve(i, j):
            if i == len(left) or j == len(right):
                return (0, 0., ())
            options = [solve(i+1, j), solve(i, j+1)]
            delta = abs(left[i]['clip_seconds']-right[j]['clip_seconds'])
            if delta <= tolerance:
                n, error, matched = solve(i+1, j+1)
                options.append((n+1, error+delta, ((i, j),)+matched))
            return max(options, key=lambda item: (item[0], -item[1]))
        for i, j in solve(0, 0)[2]:
            a, b = left[i], right[j]
            pairs.append({'reference_id': a['id'], 'candidate_id': b['id'], 'clip': clip,
                'reference_seconds': a['clip_seconds'], 'candidate_seconds': b['clip_seconds'],
                'timing_error_seconds': abs(a['clip_seconds']-b['clip_seconds']),
                'reference_status': a['automatic_status'], 'candidate_status': b['automatic_status'],
                'reference_outcome': a['outcome'], 'candidate_outcome': b['outcome'],
                'same_count_category': a['automatic_status'] == b['automatic_status'],
                'same_outcome': a['outcome'] == b['outcome']})
    left_ids = {p['reference_id'] for p in pairs}; right_ids = {p['candidate_id'] for p in pairs}
    return {'meaning': 'Prediction agreement, not accuracy. Maximum one-to-one timing alignment within the same team and clip; outcomes are not used to pick matches.',
        'tolerance_seconds': tolerance, 'pairs': pairs,
        'matched': len(pairs), 'same_count_category': sum(p['same_count_category'] for p in pairs),
        'same_outcome': sum(p['same_outcome'] for p in pairs),
        'reference_only': [e['id'] for e in reference if e['id'] not in left_ids],
        'candidate_only': [e['id'] for e in candidate if e['id'] not in right_ids]}


def usage_priced_cost(usage, protocol):
    """Price disjoint ordinary input, cache reads and cache writes once each."""
    details = usage.get('input_tokens_details') or {}
    cached = details.get('cached_tokens', 0)
    writes = details.get('cache_write_tokens', 0)
    split = details.get('cache_write_tokens_details') or {}
    long_writes = split.get('ttl_12h_tokens', 0)
    short_writes = split.get('ttl_30m_tokens', writes-long_writes)
    ordinary = usage['input_tokens']-cached-writes
    if min(ordinary, cached, short_writes, long_writes) < 0 or short_writes+long_writes != writes:
        raise ValueError('Inconsistent cache token accounting')
    return (ordinary*protocol['input_usd_per_million'] + cached*protocol['cached_input_usd_per_million']
        + short_writes*protocol['input_usd_per_million']*1.25 + long_writes*protocol['input_usd_per_million']*2.5
        + usage['output_tokens']*protocol['output_usd_per_million'])/1e6


def phase_summary(ledger, protocol):
    rows = list(ledger['requests'].values())
    times = [r['api_seconds'] for r in rows if r['status'] == 'completed' and 'api_seconds' in r]
    return {'requests': len(rows), 'statuses': dict(Counter(r['status'] for r in rows)),
        'estimated_usd_including_unresolved_reservations': sum(r.get('estimated_usd', r.get('reserved_usd', 0)) for r in rows),
        'standard_rate_usd_for_reported_usage': sum(usage_priced_cost(r['usage'], protocol) for r in rows if r.get('usage')),
        'legacy_base_rate_estimate_excluding_cache_writes_usd': sum(r.get('standard_rate_usd', 0) for r in rows),
        'unresolved_reserved_usd': sum(r.get('reserved_usd', 0) for r in rows if 'estimated_usd' not in r),
        'input_tokens': sum((r.get('usage') or {}).get('input_tokens', 0) for r in rows),
        'output_tokens': sum((r.get('usage') or {}).get('output_tokens', 0) for r in rows),
        'median_completed_api_seconds': statistics.median(times) if times else None}


def proposed_goal_check(events, clips):
    """Separate recognizing a possible goal from passing the frozen count gate."""
    proposed = [{**e, 'automatic_status': 'sog'} for e in events if e['outcome'] == 'goal']
    result = goal_sanity_check(proposed, clips)
    if result:
        result['meaning'] = 'Development diagnostic: match any predicted goal outcome before confidence/evidence counting rules. This does not make withheld goals counted SOG, and does not establish accuracy or recall.'
    return result


def derive(exp=EXP):
    protocol = read(exp/'protocol.json')
    ledgers = {phase: read(exp/f'results/{phase}-ledger.json') for phase in ('scan', 'matched-context', 'own-context')}
    if (exp/'results/max-context-ledger.json').exists():
        ledgers['max-context'] = read(exp/'results/max-context-ledger.json')
    if (exp/'results/max-recovery-ledger.json').exists():
        ledgers['max-recovery'] = read(exp/'results/max-recovery-ledger.json')
    presentation = read(BASE/'presentation.json')
    clips = [{**{k: v for k, v in c.items() if k != 'frame_folder'}, **presentation[c['id']]} for c in protocol['clips']]
    phases = {name: phase_summary(ledger, protocol) for name, ledger in ledgers.items()}
    outputs = {}
    effective_max = []
    if 'max-context' in ledgers:
        recoveries = ledgers.get('max-recovery', {}).get('requests', {})
        for key, row in ledgers['max-context']['requests'].items():
            recovery = recoveries.get(key.replace('-max-context-', '-max-recovery-'))
            effective_max.append(recovery if recovery and recovery['status'] == 'completed' else row)
    for phase in (name for name in ledgers if name not in ('scan', 'max-recovery')):
        events = []
        for clip in clips:
            rows = effective_max if phase == 'max-context' else ledgers[phase]['requests'].values()
            records = [r for r in rows if r['clip'] == clip['id']]
            found = combine_events(records, clip)
            for event in found:
                event['team_label'] = clip['team_labels'].get(event['team_color'], 'Unknown team')
            events.extend(found)
        outputs[phase] = {'schema_version': 1, 'source_type': 'automatic_sequence_model_estimates',
            'model': protocol['model'], 'fps': 15, 'clips': clips, 'events': events}
        if phase == 'max-context':
            outputs[phase]['scope'] = 'Maximum-reasoning diagnostic on nine Astra-selected candidate windows only. Not a complete independent scan.'
    events = outputs['own-context']['events']
    minutes = sum(c['duration_s'] for c in clips)/60
    pipeline_cost = sum(phases[p]['estimated_usd_including_unresolved_reservations'] for p in ('scan', 'own-context'))
    campaign_cost = sum(p['estimated_usd_including_unresolved_reservations'] for p in phases.values())
    baseline = read(ROOT/'web/data/sog-20261027/events.json')['events']
    baseline_report = read(ROOT/'web/data/sog-20261027/report.json')
    own_requests = read(exp/'results/own-requests.json')['requests']
    baseline_sog = [e for e in baseline if e['automatic_status'] == 'sog']
    covered_baseline = [e['id'] for e in baseline_sog if any(r['clip'] == e['clip'] and r['core'][0] <= e['release_frame'] < r['core'][1] for r in own_requests)]
    original_context = read(BASE/'results/context-ledger.json')['requests']
    paired_requests = []
    for key, row in ledgers['matched-context']['requests'].items():
        original = original_context[key.replace('-matched-context-', '-context-')]
        if row['status'] == original['status'] == 'completed':
            paired_requests.append({'terra_request': key, 'astra_request': original['id'],
                'astra_conservative_usd': original['estimated_usd'], 'terra_conservative_usd': row['estimated_usd'],
                'astra_response_seconds': original['seconds'], 'terra_response_seconds': row['api_seconds']})
    paired_cost = {'requests': len(paired_requests), 'pairs': paired_requests,
        'astra_conservative_usd': sum(p['astra_conservative_usd'] for p in paired_requests),
        'terra_conservative_usd': sum(p['terra_conservative_usd'] for p in paired_requests),
        'astra_median_response_seconds': statistics.median(p['astra_response_seconds'] for p in paired_requests) if paired_requests else None,
        'terra_median_response_seconds': statistics.median(p['terra_response_seconds'] for p in paired_requests) if paired_requests else None,
        'meaning': 'Same prompt, source frames, crops, sampling and reasoning/output settings; only requests completed by both original context passes. Timing excludes frame preparation and token-count requests. Separate, non-simultaneous runs; this is not a controlled latency benchmark.'}
    clip_reports = []
    for clip in clips:
        covered = set()
        for r in ledgers['scan']['requests'].values():
            if r['clip'] == clip['id'] and r['status'] == 'completed':
                covered.update(range(*r['core']))
        found = [e for e in events if e['clip'] == clip['id']]
        clip_reports.append({'id': clip['id'], 'scanned_seconds': len(covered)/15,
            'scan_coverage': len(covered)/clip['frames'], 'totals': event_totals(found),
            'contextual_sog': sum(e['automatic_status'] == 'sog' and e['basis'] == 'contextual' for e in found)})
    report = {'scope': f'{minutes:g} minutes from the same three NBHL recordings; development model comparison.',
        'model': protocol['model'], 'method_summary': 'Terra 6.1 scans chronological 20-second windows at 1.875 fps, then re-examines its own proposed attempts with full views at 2.5 fps, native goal crops at 7.5 fps and up to 30 seconds of aftermath. The shot-counting rules are frozen from the Astra experiment. No Astra event answers are supplied to this pipeline.',
        'review_summary': 'These are experimental model predictions on previously examined development clips. Agreement with Astra is not ground truth. The existing three New Jersey goals provide a limited sanity check; SOG precision and recall remain unmeasured.',
        'definition': protocol['definition'], 'clips': clip_reports, 'totals': event_totals(events), 'phases': phases,
        'pipeline_estimated_usd': pipeline_cost, 'pipeline_usd_per_video_minute': pipeline_cost/minutes,
        'pipeline_standard_rate_usd_for_reported_usage': sum(phases[p]['standard_rate_usd_for_reported_usage'] for p in ('scan', 'own-context')),
        'estimated_usd_including_unresolved_reservations': campaign_cost, 'campaign_budget_usd': protocol['budget_usd'],
        'cost_scope': 'Pipeline cost includes Terra scan and its own context pass. Matched-context and max experiments are additional evaluation expense, not pipeline cost. Conservative estimates include unresolved reservations and a 25% input-rate margin. Standard-rate totals are recomputed from reported ordinary input, cache reads and cache writes, using the default 30-minute cache-write rate when no lifetime breakdown is supplied. The raw ledger standard_rate_usd field is a legacy base-rate estimate that omits cache-write premiums; do not use it as billed cost. Neither estimate is an invoice. Modal compute/storage is excluded.',
        'pricing_sources': protocol['pricing_sources']+['https://developers.openai.com/api/docs/guides/prompt-caching'],
        'unprocessed_proposals': [r['id'] for r in ledgers['own-context']['requests'].values() if r['status'] != 'completed'],
        'goal_sanity_check': goal_sanity_check(events, clips),
        'proposed_goal_sanity_check': proposed_goal_check(events, clips),
        'matched_context_goal_sanity_check': goal_sanity_check(outputs['matched-context']['events'], clips),
        'matched_context_proposed_goal_sanity_check': proposed_goal_check(outputs['matched-context']['events'], clips),
        'matched_context_totals': event_totals(outputs['matched-context']['events']),
        'paired_completed_context_cost_and_latency': paired_cost,
        'comparison': {phase: match_events(baseline, outputs[phase]['events'], protocol['evaluation']['event_match_tolerance_seconds']) for phase in ('matched-context', 'own-context')},
        'baseline_totals': baseline_report['totals'],
        'baseline_events_sha256': sha(ROOT/'web/data/sog-20261027/events.json'),
        'baseline_report_sha256': sha(ROOT/'web/data/sog-20261027/report.json'),
        'baseline_sog_within_terra_proposal_cores': {'covered': covered_baseline,
            'outside': [e['id'] for e in baseline_sog if e['id'] not in covered_baseline],
            'meaning': 'Whether each Astra-estimated SOG release falls within a Terra review core; proposal coverage of another model’s predictions, not shot recall.'},
        'baseline_campaign_usd_per_video_minute': baseline_report['estimated_usd_including_unresolved_reservations']/minutes,
        'baseline_scan_plus_context_recovery_usd_per_video_minute': sum(baseline_report['phases'][p]['estimated_usd_including_unresolved_reservations'] for p in ('scan', 'context', 'context-recovery'))/minutes,
        'baseline_comparison_note': 'Astra campaign includes its earlier strict verification and recovery experiments; its scan+context+recovery subtotal omits verification expense although one window used verifier fallback. Terra matched-context uses Astra proposals for a controlled diagnostic, not independent discovery.',
        'input_sha256': {str(p.relative_to(exp)): sha(p) for p in [exp/'protocol.json', exp/'worker_terra.py', exp/'modal_run.py', *([exp/'max-protocol.json', exp/'modal_max.py'] if 'max-context' in ledgers else []), *([exp/'max-recovery-protocol.json', exp/'modal_max_recovery.py'] if 'max-recovery' in ledgers else []), *sorted((exp/'results').glob('*.json'))]},
        'count_policy_sha256': sha(ROOT/'hockey_tracking/shots.py'),
        'limitations': ['No independent human SOG labels, precision or recall.', 'More counted events or agreement with Astra does not establish higher accuracy.',
            'Initial scanning can miss attempts. Failed windows are missing, not zero.', 'Counts are estimates and shooter identities remain unverified.',
            'Matched-context reuses Astra proposals; only own-context tests independent Terra discovery.', 'No ball/player tracking costs are included.']}
    if 'max-context' in ledgers:
        config = read(exp/'max-protocol.json')
        requests = config['requests']
        def selected(event):
            return any(r['clip'] == event['clip'] and r['core'][0] <= event['release_frame'] < r['core'][1] for r in requests)
        selected_baseline = [e for e in baseline if selected(e)]
        selected_high = [e for e in outputs['matched-context']['events'] if selected(e)]
        high_ids = {r['id'].replace('-max-context-', '-matched-context-') for r in requests}
        high_cost = sum(r.get('estimated_usd', r.get('reserved_usd', 0)) for k, r in ledgers['matched-context']['requests'].items() if k in high_ids)
        max_events = outputs['max-context']['events']
        report['max_reasoning_diagnostic'] = {'scope': config['scope'], 'requests': len(requests),
            'completed_windows': sum(r['status'] == 'completed' for r in effective_max),
            'missing_windows': [r['id'] for r in effective_max if r['status'] != 'completed'],
            'totals': event_totals(max_events), 'high_reasoning_same_windows_totals': event_totals(selected_high),
            'astra_same_windows_totals': event_totals(selected_baseline),
            'estimated_usd': sum(phases[p]['estimated_usd_including_unresolved_reservations'] for p in ('max-context', 'max-recovery') if p in phases),
            'high_reasoning_same_windows_estimated_usd': high_cost,
            'goal_sanity_check': goal_sanity_check(max_events, clips), 'proposed_goal_sanity_check': proposed_goal_check(max_events, clips),
            'vs_high': match_events(selected_high, max_events), 'vs_astra': match_events(selected_baseline, max_events),
            'note': 'Added after examining high-reasoning outputs. Maximum reasoning and an increased output allowance are changed together. Initial max calls allow 16384 tokens; known token-limited responses can receive a separately versioned 65536-token recovery with a 900-second timeout. Both costs remain included. Conditional development diagnostic, not a fresh accuracy test or a measured maximum-reasoning full pipeline.'}
    return outputs, report


def max_view_report(outputs, report):
    dataset = outputs['max-context']
    completed = report['max_reasoning_diagnostic']['completed_windows']
    return {'scope': dataset['scope'], 'model': dataset['model'],
        'coverage_note': f'{completed} of nine Astra-selected play windows completed review across the recordings. These are not segment totals; missing answers do not mean zero shots.',
        'method_summary': 'Terra 6.1 at maximum reasoning examines the same frames and crops as the high-reasoning matched test. Initial calls allow 16,384 output tokens; known token-limited responses can receive a 65,536-token recovery. The nine windows were selected because Astra previously counted SOG there. No event labels are supplied to inference; this is a conditional diagnostic, not independent discovery.',
        'review_summary': report['review_summary'], 'definition': report['definition'],
        'clips': [{'id': c['id'], 'totals': event_totals([e for e in dataset['events'] if e['clip'] == c['id']])} for c in dataset['clips']],
        'totals': event_totals(dataset['events']), 'diagnostic_estimated_usd': report['max_reasoning_diagnostic']['estimated_usd'],
        'diagnostic': report['max_reasoning_diagnostic'], 'phases': {p: report['phases'][p] for p in ('max-context', 'max-recovery') if p in report['phases']},
        'input_sha256': report['input_sha256'], 'count_policy_sha256': report['count_policy_sha256'],
        'limitations': report['limitations'], 'cost_scope': 'Cost of these nine contextual reviews only; excludes discovery, other experimental passes and Modal. Not a per-minute pipeline estimate.'}


def main():
    outputs, report = derive()
    DEST.mkdir(parents=True, exist_ok=True)
    write(DEST/'events.json', outputs['own-context'])
    write(DEST/'matched-events.json', outputs['matched-context'])
    if 'max-context' in outputs:
        write(DEST/'max-events.json', outputs['max-context'])
        write(DEST/'max-report.json', max_view_report(outputs, report))
    write(DEST/'report.json', report)
    fields = ['id', 'clip', 'source_id', 'clip_seconds', 'recording_seconds', 'team_color', 'team_label', 'shooter_jersey', 'attempt', 'outcome', 'on_target', 'automatic_status', 'basis', 'confidence', 'count_reason', 'explanation']
    for name, phase in [('events', 'own-context'), *([('max-events', 'max-context')] if 'max-context' in outputs else [])]:
        with (DEST/f'{name}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
            writer.writeheader(); writer.writerows(outputs[phase]['events'])
    print(json.dumps({'totals': report['totals'], 'pipeline_usd_per_minute': report['pipeline_usd_per_video_minute'], 'campaign_usd': report['estimated_usd_including_unresolved_reservations']}))


if __name__ == '__main__':
    main()
