"""Seal proposal-free manual visual notes; ambiguous identities stay unknown."""
import datetime
import hashlib
import itertools
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def pairs_for(observations):
    co_visible=set()
    for a,b in itertools.combinations(observations,2):
        if a['clip']==b['clip']and a['frame']==b['frame']and a.get('identity')and b.get('identity'):
            assert a['identity']!=b['identity'],'Duplicate identity in one frame'
            co_visible.add((a['clip'],*sorted([a['identity'],b['identity']])))
    pairs=[]
    for a,b in itertools.combinations(observations,2):
        if a['clip']!=b['clip']:continue
        aa,bb=a.get('identity'),b.get('identity')
        if aa and aa==bb:truth='same';reason='visually reviewed identity continuity'
        elif a['frame']==b['frame']:truth='different';reason='simultaneously visible people'
        elif a.get('team')and b.get('team')and a['team']!=b['team']:truth='different';reason='different team uniforms'
        elif aa and bb and (a['clip'],*sorted([aa,bb]))in co_visible:truth='different';reason='identity groups are co-visible elsewhere in this clip'
        else:truth='uncertain';reason='insufficient identity evidence'
        pairs.append({'id':a['id']+'__'+b['id'],'clip':a['clip'],'a':a['id'],'b':b['id'],'truth':truth,'reason':reason})
    return pairs

def main():
    target=ROOT/'data/reference.json'
    assert not target.exists(),'Reference already sealed; never overwrite after inference'
    protocol=read(ROOT/'data/protocol.json');notes=read(ROOT/'data/visual-notes.json')
    review=read(ROOT/'data/source-review.json')
    assert review['protocol_sha256']==sha(ROOT/'data/protocol.json')
    assert review['inference_performed']is False and all(s['confirmed_ball_hockey']for s in review['sources'])
    assert notes['proposal_free']is True
    clips={c['clip']:c for c in protocol['clips']};observations=[];frames=[]
    for row in notes['frames']:
        name=row['clip'];f=row['frame'];assert f in clips[name]['reference_frames']
        w,h=row['native_size'];scale=row.get('coordinate_scale_to_native',1)
        for i,item in enumerate(row['observations']):
            x,y,u,v=[float(n)*scale for n in item['bbox']]
            assert 0<=x<u<=w and 0<=y<v<=h
            observations.append(item|{'id':f'{name}/{f:04d}/{i:02d}','clip':name,'frame':f,'bbox':[x,y,u,v]})
        frames.append({k:v for k,v in row.items()if k not in ['observations','coordinate_scale_to_native']})
    expected={(c['clip'],f)for c in protocol['clips']for f in c['reference_frames']}
    assert len(frames)==len(expected) and {(r['clip'],r['frame'])for r in frames}==expected
    reference={'status':'sealed_before_inference','sealed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':sha(ROOT/'data/protocol.json'),'visual_notes_sha256':sha(ROOT/'data/visual-notes.json'),'frames':frames,'observations':observations,'pairs':pairs_for(observations),'annotation_source':'Assistant visual review of raw frames, with no detector proposals or tracking predictions. No independent human verification.','scope':'Selected visible players only. Not exhaustive detector ground truth. Local reference identity groups are not roster identities.'}
    target.write_text(json.dumps(reference,indent=2)+'\n')
    freeze={'sealed_at':reference['sealed_at'],'protocol_sha256':reference['protocol_sha256'],'reference_sha256':sha(target),'code_sha256':{p.name:sha(p)for p in sorted((ROOT/'scripts').glob('*.py'))}}
    (ROOT/'data/code-freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
    print(json.dumps({'observations':len(observations),'pairs':len(reference['pairs']),'same':sum(p['truth']=='same'for p in reference['pairs']),'different':sum(p['truth']=='different'for p in reference['pairs']),'unknown':sum(p['truth']=='uncertain'for p in reference['pairs']),'reference_sha256':sha(target)}))

if __name__=='__main__':main()
