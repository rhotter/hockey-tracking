"""Reproduce published SOG decisions, coverage, provenance and cost accounting."""
import csv
import hashlib
import json
from pathlib import Path
from detect_shots import derive,read,sha
from propose_shot_windows import propose

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/sog-20261027'

def main():
    dataset,report=derive(EXP);site=ROOT/'web/data/sog-20261027'
    assert read(site/'events.json')==dataset
    assert read(site/'report.json')==report
    assert read(EXP/'count-policy.json')['sha256']==sha(ROOT/'hockey_tracking/shots.py')
    assert len(dataset['events'])==len({e['id']for e in dataset['events']})
    with (site/'events.csv').open()as f:rows=list(csv.DictReader(f))
    assert len(rows)==len(dataset['events'])
    for row,event in zip(rows,dataset['events']):
        assert all(row[k]==(''if event.get(k)is None else str(event[k]))for k in row)
    protocol=read(EXP/'protocol.json');config=read(EXP/'context-protocol.json')
    assert read(EXP/'verify-requests.json')['source_scan_sha256']==sha(EXP/'results/scan-ledger.json')
    assert read(EXP/'verify-requests.json')==propose(protocol,read(EXP/'results/scan-ledger.json'),sha(EXP/'results/scan-ledger.json'))
    frames={c['id']:c['frames']for c in protocol['clips']};source_hashes={}
    for phase in report['phases']:
        ledger=read(EXP/f'results/{phase}-ledger.json')
        assert ledger['protocol_sha256']==sha(EXP/'protocol.json')
        assert not any(r['status']=='reserved'for r in ledger['requests'].values()),'Inference is still pending'
        if phase=='context':
            assert ledger['context_protocol_sha256']==sha(EXP/'context-protocol.json')
            assert ledger['worker_sha256']==sha(EXP/'context_worker.py')
        if phase=='verify':assert ledger['worker_sha256']==sha(EXP/'worker.py')
        if phase=='context-recovery':
            assert ledger['recovery_protocol_sha256']==sha(EXP/'recovery-protocol.json')
            assert ledger['worker_sha256']==sha(EXP/'recovery_worker.py')
            prior=read(EXP/'results/context-ledger.json')['requests']
            assert all(prior[k.replace('-context-recovery-','-context-')]['status']=='incomplete'for k in ledger['requests'])
        for r in ledger['requests'].values():
            assert 0<=r['core'][0]<r['core'][1]<=frames[r['clip']]
            assert set(map(int,r['source_sha256']))==set(r['sample_frames'])
            for frame,value in r['source_sha256'].items():
                assert len(value)==64 and all(c in '0123456789abcdef'for c in value)
                key=(r['clip'],int(frame));assert source_hashes.get(key,value)==value;source_hashes[key]=value
            if r.get('usage'):
                expected=(r['usage']['input_tokens']*config['input_usd_per_million_conservative']+r['usage']['output_tokens']*config['output_usd_per_million'])/1e6
                assert abs(r['estimated_usd']-expected)<1e-9
            else:assert r['status']!='completed'
    assert sum(report['phases'][p]['estimated_usd_including_unresolved_reservations']for p in ('scan','verify'))<=protocol['budget_usd']
    assert report['estimated_usd_including_unresolved_reservations']<=config['campaign_budget_usd']
    assert read(EXP/'event-review-notes.json')['source_manifest_sha256']==sha(EXP/'event-review-source-manifest.json')
    for row in read(EXP/'event-review-source-manifest.json'):
        assert source_hashes.get((row['clip'],row['frame']),row['source_sha256'])==row['source_sha256']
    assets=read(ROOT/'site-assets.json')['files']
    assert all(c['video']in assets for c in dataset['clips'])
    for c in report['clips']:assert 0<=c['scan_coverage']<=1
    print(json.dumps({'events':len(dataset['events']),'totals':report['totals'],
        'frame_hashes_cross_checked':len(source_hashes),'estimated_usd':round(report['estimated_usd_including_unresolved_reservations'],2)}))

if __name__=='__main__':main()
