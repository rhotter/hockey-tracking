"""Explicit schema compatibility control, added before viewing accuracy scores.

Qwen often emits digit strings instead of JSON integers. Preserve the original
strict parser results and separately evaluate lossless decimal normalization.
"""
import copy
import json
import re
from pathlib import Path

POLICIES = {
    'jersey_veto_normalized': 'jersey_veto',
    'jersey_support_normalized': 'jersey_support',
    'multi_and_jersey_veto_normalized': 'multi_and_jersey_veto',
}


def normalized_responses(results):
    output = copy.deepcopy(results)
    converted = 0
    for row in output:
        parsed = row.get('parsed')
        if row.get('task') != 'number' or not isinstance(parsed, dict):
            continue
        readings = parsed.get('readings')
        if not isinstance(readings, list):
            continue
        for reading in readings:
            if not isinstance(reading, dict):
                continue
            value = reading.get('number')
            if isinstance(value, str) and re.fullmatch(r'(?:0|[1-9][0-9]?)', value):
                reading['number'] = int(value)
                converted += 1
    return output, converted


def extend(phase, model):
    from nbhl_vlm_reconnect import OUT, datasets, apply_policy, dump, sha
    from nbhl_tracking_fourhour import tracklet_evidence, link_candidates, evaluate
    from nbhl_tracking_reference_audit import corrected_report
    folder = OUT/phase/model
    full = json.loads((folder/'evaluation.json').read_text())
    manifest = json.loads((OUT/phase/'evidence.json').read_text())
    prediction = json.loads((folder/'predictions.json').read_text())
    responses, converted = normalized_responses(prediction['results'])
    responses = {r['id']:r for r in responses}
    corrections = {r['pair_id']:r for r in json.loads(Path('/work/tracking-four-hour-reference-corrections-v1.json').read_text())['corrections']}
    reports = []
    for meta, features, raw, _, raw_sha in datasets(phase, labels=True):
        clip = next(c for c in manifest['clips'] if c['clip']==raw['clip'])
        assert raw_sha == clip['raw_tracks_sha256']
        evidence = tracklet_evidence(meta, features, raw)
        edges = link_candidates(meta, features, evidence)
        for policy, original in POLICIES.items():
            linked = apply_policy(raw, evidence, edges, clip, responses, original)
            row = evaluate(meta, linked)
            if phase=='development':
                row = corrected_report(row, corrections)
            row.update(policy=policy, source_id=meta['summary']['source_id'], links=len(linked['links']))
            reports.append(row)
            dump(folder/'tracks'/policy/(raw['clip']+'.json'), linked)
    full['clips'].extend(reports)
    for policy in POLICIES:
        for scope in (['all'] if phase=='development' else ['short','stress']):
            selected = [r for r in reports if r['policy']==policy and (scope=='all' or ('stress' in r['clip'])==(scope=='stress'))]
            full['totals'].append({'policy':policy, 'scope':scope,
                'counts':{k:sum(r['counts'][k] for r in selected) for k in selected[0]['counts']},
                'fragments':sum(r['fragments'] for r in selected), 'links':sum(r['links'] for r in selected)})
    full['normalization'] = {'converted_digit_strings':converted, 'amendment_sha256':sha('/work/vlm-number-amendment.json'),
                             'original_evaluation_sha256':sha(folder/'evaluation.json')}
    dump(folder/'evaluation-extended.json', full)
    return {k:v for k,v in full.items() if k!='clips'}
