"""Retain only automatically verified detector seeds; reuse their unchanged SAM tracks."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import argparse
import math
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BUILD=ROOT/'build/ball-temporal-20261026'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,separators=(',',':'))+'\n')


def gate(seeds,tracks,ledger):
    by_frame={}
    for request in ledger['requests'].values():
        if request['status']!='completed':continue
        for row in request['answer']['samples']:
            assert row['frame'] not in by_frame
            by_frame[row['frame']]=row
    accepted=[];reviews=[]
    for seed in seeds['seeds']:
        row=by_frame.get(seed['frame'],{'frame':seed['frame'],'verdict':'not_verified','support':'unclear','reason':'No completed verification.'})
        reviews.append({**row,'pixel':seed['pixel'],'source_sha256':seed['source_sha256']})
        if row['verdict']=='visible_ball':
            accepted.append({**seed,'support':row['support'],'evidence':seed['evidence']+' Sequence verification: '+row['reason']})
    keep={s['frame']for s in accepted}
    assert len(keep)==len(accepted)
    return accepted,[t for t in tracks['tracks']if t['seed_frame']in keep],reviews


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='detector',choices=['detector','yolo'])
    args=parser.parse_args(); source=args.source
    seedpath=BUILD/f'tracking-seeds-dense-{source}-gpu.json'
    trackpath=BUILD/f'short-sam3-box6-dense-{source}-gpu.json'
    ledgerpath=BUILD/('seed-verification-ledger.json' if source=='detector' else 'yolo-verification-ledger.json')
    seeds,tracks,ledger=map(read,[seedpath,trackpath,ledgerpath])
    assert tracks['complete'] and tracks['seed_file_sha256']==sha(seedpath)
    assert ledger['protocol']['seed_sha256']==sha(seedpath)
    assert len(ledger['requests'])==math.ceil(len(seeds['seeds'])/5),'Wait for every request, including withheld statuses.'
    accepted,kept,rows=gate(seeds,tracks,ledger)
    batch=f'dense-{source}-verified-gpu'
    filtered={**seeds,'batch':batch,'seeds':accepted,'parent_seed_sha256':sha(seedpath),'verification_ledger_sha256':sha(ledgerpath),
        'selection':'Only completed visible_ball sequence verdicts; coordinates unchanged. Support updated from model review, not manual labels.'}
    dest=BUILD/f'tracking-seeds-{batch}.json';write(dest,filtered)
    filtered_tracks={**tracks,'protocol':{**tracks['protocol'],'batch':batch},'tracks':kept,'seed_file_sha256':sha(dest),
        'parent_track_sha256':sha(trackpath),'derivation':'Subset of unchanged original SAM outputs; no new inference.'}
    write(BUILD/f'short-sam3-box6-{batch}.json',filtered_tracks)
    cost=sum(r.get('estimated_usd',r.get('reserved_usd',0))for r in ledger['requests'].values())
    report={'created_at':datetime.now(timezone.utc).isoformat(),'protocol':ledger['protocol'],
        'source_sha256':{'seeds':sha(seedpath),'tracks':sha(trackpath),'ledger':sha(ledgerpath)},
        'counts':dict(Counter(r['verdict']for r in rows)),'retained_seeds':len(accepted),'estimated_usd':cost,
        'previous_verification_reserved_usd':.8456 if source=='detector' else 0,
        'total_openai_estimate_usd':(41.229175+.8456 if source=='detector' else read(HERE/'results/detector-seed-verification.json')['total_openai_estimate_usd'])+cost,'rows':rows}
    write(HERE/f'results/{source}-seed-verification.json',report)
    print(json.dumps({k:report[k]for k in ['counts','retained_seeds','estimated_usd','total_openai_estimate_usd']}))


if __name__=='__main__':main()
