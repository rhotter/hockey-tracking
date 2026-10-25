"""Independently recount sparse new-recording results and verify provenance."""
from collections import Counter
from pathlib import Path
import datetime
import hashlib
import json
import sys
from verify_roboflow_results import recount

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/new-games-20261025'
DATA=ROOT/'web/data/reliability/new-games-20261025'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assets=read(ROOT/'site-assets.json')['files']
    for name,digest in read(DATA/'export-manifest.json')['files'].items():
        if name.endswith('.json'):assert sha(ROOT/'web'/name)==digest
        else:assert assets[name]==digest
    sys.path.insert(0,str(EXP/'scripts'))
    from seal_reference import pairs_for
    protocol=read(EXP/'data/protocol.json');reference=read(EXP/'data/reference.json');result=read(DATA/'results.json')
    assert result['clips']==protocol['clips'] and result['sources']==protocol['sources']
    assert result['frames']==sum(c['duration_s']*c['fps']for c in protocol['clips'])
    assert [m['method']for m in result['methods']]==protocol['methods']
    assert reference['status']=='sealed_before_inference'
    assert result['reference_sha256']==sha(EXP/'data/reference.json')==sha(DATA/'reference.json')
    assert result['protocol_sha256']==reference['protocol_sha256']==sha(EXP/'data/protocol.json')==sha(DATA/'protocol.json')
    assert datetime.datetime.fromisoformat(protocol['frozen_at'])<datetime.datetime.fromisoformat(reference['sealed_at'])
    assert not(set(s['id']for s in protocol['sources'])&set(protocol['previous_source_ids']))
    assert reference['pairs']==pairs_for(reference['observations'])
    freeze=read(EXP/'data/code-freeze.json')
    assert freeze['reference_sha256']==sha(EXP/'data/reference.json')
    for name,digest in freeze['code_sha256'].items():assert sha(EXP/'scripts'/name)==digest,name
    assert all(s['confirmed_ball_hockey']for s in read(DATA/'source-review.json')['sources'])
    assert result['source_review_sha256']==sha(DATA/'source-review.json')==sha(EXP/'data/source-review.json')
    assert reference['visual_notes_sha256']==sha(EXP/'data/visual-notes.json')
    reviews={s['source_id']:s for s in read(DATA/'source-review.json')['sources']}
    samples={(f['clip'],f['frame']):f for f in reference['frames']}
    for reservation in result['reservations']:
        reviewed=reviews[reservation['source_id']]
        assert reviewed['raw_sha256']==reservation['raw_sha256']
        assert reviewed['raw_bytes']==reservation['raw_bytes']
        for clip in reservation['clips']:
            for frame in clip['samples']:
                assert samples[clip['clip'],frame['frame']]['source_frame_sha256']==frame['sha256']
    assert {(r['task'],r['clip'])for r in result['execution']}=={(task,c['clip'])for task in ['detect','track','sam','embed']for c in protocol['clips']}
    for run in result['execution']:
        assert run['started_at']>reference['sealed_at']
        assert run['reference_sha256']==result['reference_sha256'] and run['protocol_sha256']==result['protocol_sha256']
        assert run['adapter_sha256']==freeze['code_sha256']['new_games.py']
    shape={c['clip']:(c['height'],c['width'])for r in result['reservations']for c in r['clips']}
    observations={o['id']:o for o in reference['observations']};checked=0
    for method in result['methods']:
        totals=Counter();category={key:Counter()for key in ['up_to_2s','over_2s']}
        for c in protocol['clips']:
            name=c['clip']+'-'+method['method']+'.json'
            pred=read(DATA/'sampled-predictions'/name);report=read(DATA/'reports'/name)
            assert {k:v for k,v in report.items()if k!='pairs'}==next(r for r in result['per_clip']if r['clip']==c['clip']and r['method']==method['method'])
            truth={p['id']:p for p in reference['pairs']if p['clip']==c['clip']}
            for pair in report['pairs']:
                for key in ['a','b','clip','truth']:assert pair[key]==truth[pair['id']][key]
            counts,outcomes=recount(reference,pred,c['clip']);assert counts==report['counts']
            assert {p['id']:p['predicted_same']for p in report['pairs']}==outcomes
            for cat in category:
                pairs=[p for p in reference['pairs']if p['clip']==c['clip']and(abs(observations[p['a']]['frame']-observations[p['b']]['frame'])<=30)==(cat=='up_to_2s')]
                value={'same_total':sum(p['truth']=='same'for p in pairs),'correct':sum(p['truth']=='same'and outcomes[p['id']]is True for p in pairs),'different_total':sum(p['truth']=='different'for p in pairs),'wrong':sum(p['truth']=='different'and outcomes[p['id']]is True for p in pairs),'unknown_same':sum(p['truth']=='uncertain'and outcomes[p['id']]is True for p in pairs)}
                assert value==report[cat];category[cat].update(value)
            replay=read(ROOT/'web/media/new-games-20261025'/name)
            assert replay['source_tracks_sha256']==report['tracks_sha256']
            assert len(replay['frames'])==c['duration_s']*15 and replay['fps']==15 and replay['source_id']==c['source_id']
            assert replay['fragment_ids_only']and not replay['interpolated_positions']and not replay['manually_corrected']
            assert replay['start_s']==c['start_s'] and replay['method']==method['method']
            for boxes in replay['frames']:
                assert len({b[0]for b in boxes})==len(boxes)
                for tid,x,y,u,v in boxes:
                    assert type(tid)is int and tid>0 and 0<=x<u<=10000 and 0<=y<v<=10000
            h,w=shape[c['clip']]
            for row in pred['frames']:
                boxes=[[replay['track_id_map'][o['track_id']],*[round(min(1,max(0,v/s))*10000)for v,s in zip(o['bbox'],[w,h,w,h])]]for o in row['objects']]
                assert boxes==replay['frames'][row['frame']]
            totals.update(counts);checked+=1
        assert dict(totals)==method['counts']
        for cat in category:assert dict(category[cat])==method[cat]
    for c in protocol['clips']:
        raw=read(ROOT/'web/media/new-games-20261025'/(c['clip']+'-existing-sam2-local.json'))
        linked=read(ROOT/'web/media/new-games-20261025'/(c['clip']+'-existing-sam2-osnet-qwen2b.json'))
        assert [[b[1:]for b in row]for row in raw['frames']]==[[b[1:]for b in row]for row in linked['frames']]
    from analyze_new_games import compute
    assert read(DATA/'analysis.json')==compute()
    assert sha(DATA/'diagnostic-scene-review.json')==sha(EXP/'data/diagnostic-scene-review.json')
    diagnosis=read(DATA/'diagnostic-scene-review.json')
    crossing={p['id']for p in reference['pairs']if p['clip']==diagnosis['clip']and p['truth']=='same'and any(min(observations[p['a']]['frame'],observations[p['b']]['frame'])<f<=max(observations[p['a']]['frame'],observations[p['b']]['frame'])for f in diagnosis['reset_frames'])}
    assert crossing==set(diagnosis['known_same_pairs_crossing_reset'])
    print(json.dumps({'reports_independently_recounted':checked,'new_recordings':len(protocol['sources']),'frames':result['frames'],'reference_seal_verified':True,'inference_code_unchanged_since_seal':True}))

if __name__=='__main__':main()
